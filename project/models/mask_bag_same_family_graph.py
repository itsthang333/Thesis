from __future__ import annotations

"""Fixed S3 same-family graph smoothing for frozen selector-cache logits."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_critical_relation_training import _padded_batch
from models.mask_bag_orbit_relation_training import _orbit_inputs
from models.mask_bag_relational_selector import (
    build_family_overlap_graph_from_pairwise,
    smooth_candidate_logits,
)
from models.rad_dino_mask_bag_mil import smooth_mil_pool


@dataclass(frozen=True)
class SameFamilyGraphConfig:
    minimum_iou: float = 0.25
    minimum_containment: float = 0.50
    alpha: float = 0.50
    iterations: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_iou <= 1.0:
            raise ValueError("minimum_iou must lie in [0,1]")
        if not 0.0 <= self.minimum_containment <= 1.0:
            raise ValueError("minimum_containment must lie in [0,1]")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must lie in [0,1)")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")


def _padded_graph(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    valid: torch.Tensor,
    *,
    config: SameFamilyGraphConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, candidates = valid.shape
    iou = torch.zeros((batch, candidates, candidates), dtype=torch.float32, device=device)
    containment = torch.zeros_like(iou)
    family_ids = torch.full(
        (batch, candidates),
        -1,
        dtype=torch.long,
        device=device,
    )
    for row, record_index in enumerate(indices):
        record = records[int(record_index)]
        count = int(valid[row].sum().item())
        record_iou = np.asarray(record["pairwise_iou"], dtype=np.float32)
        record_containment = np.asarray(
            record["pairwise_containment"], dtype=np.float32
        )
        record_families = np.asarray(record["family_ids"], dtype=np.int64)
        if (
            record_iou.shape != (count, count)
            or record_containment.shape != (count, count)
            or record_families.shape != (count,)
            or np.any(record_families < 0)
        ):
            raise ValueError("S3 pairwise/family cache alignment mismatch")
        iou[row, :count, :count] = torch.from_numpy(record_iou).to(device)
        containment[row, :count, :count] = torch.from_numpy(
            record_containment
        ).to(device)
        family_ids[row, :count] = torch.from_numpy(record_families).to(device)
    adjacency = build_family_overlap_graph_from_pairwise(
        iou,
        containment,
        valid,
        family_ids,
        minimum_iou=config.minimum_iou,
        minimum_containment=config.minimum_containment,
    )
    return adjacency, family_ids


def score_same_family_graph_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    *,
    bag_temperature: float,
    graph_config: SameFamilyGraphConfig,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score records without fitting and return reproducible GT-blind diagnostics."""

    if not records or batch_size <= 0 or bag_temperature <= 0:
        raise ValueError("S3 scoring requires records and positive controls")
    frozen_base_scorer.requires_grad_(False).eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        original, flipped, valid, _ = _padded_batch(records, indices, device)
        adjacency, family_ids = _padded_graph(
            records,
            indices,
            valid,
            config=graph_config,
            device=device,
        )
        with torch.inference_mode():
            _, base = _orbit_inputs(original, flipped, valid, frozen_base_scorer)
            _, swapped_base = _orbit_inputs(
                flipped,
                original,
                valid,
                frozen_base_scorer,
            )
            alpha_zero = smooth_candidate_logits(
                base,
                valid,
                adjacency,
                alpha=0.0,
                iterations=graph_config.iterations,
            )
            smoothed = smooth_candidate_logits(
                base,
                valid,
                adjacency,
                alpha=graph_config.alpha,
                iterations=graph_config.iterations,
            )
            swapped = smooth_candidate_logits(
                swapped_base,
                valid,
                adjacency,
                alpha=graph_config.alpha,
                iterations=graph_config.iterations,
            )
            bag_logits = smooth_mil_pool(
                smoothed,
                valid,
                temperature=bag_temperature,
            )
            base_bag_logits = smooth_mil_pool(
                base,
                valid,
                temperature=bag_temperature,
            )
        for row, record_index in enumerate(indices):
            count = int(valid[row].sum().item())
            row_adjacency = adjacency[row, :count, :count]
            row_families = family_ids[row, :count]
            identity = torch.eye(count, dtype=torch.bool, device=device)
            off_diagonal = row_adjacency.bool() & ~identity
            isolated = ~off_diagonal.any(dim=1)
            cross_family = off_diagonal & (
                row_families[:, None] != row_families[None, :]
            )
            graph_symmetric = torch.equal(row_adjacency, row_adjacency.T)
            alpha_zero_exact = torch.equal(
                alpha_zero[row, :count],
                base[row, :count],
            )
            isolated_exact = torch.equal(
                smoothed[row, :count][isolated],
                base[row, :count][isolated],
            )
            view_swap_exact = torch.equal(
                smoothed[row, :count],
                swapped[row, :count],
            )
            output.append(
                {
                    "image_id": records[int(record_index)]["image_id"],
                    "candidate_logits": smoothed[row, :count].float().cpu().numpy(),
                    "base_candidate_logits": base[row, :count].float().cpu().numpy(),
                    "bag_logit": float(bag_logits[row].item()),
                    "bag_probability": float(torch.sigmoid(bag_logits[row]).item()),
                    "base_bag_logit": float(base_bag_logits[row].item()),
                    "base_bag_probability": float(
                        torch.sigmoid(base_bag_logits[row]).item()
                    ),
                    "candidate_count": count,
                    "view_swap_exact": bool(view_swap_exact),
                    "alpha_zero_identity_exact": bool(alpha_zero_exact),
                    "graph_symmetric": bool(graph_symmetric),
                    "cross_family_edge_count": int(cross_family.sum().item() // 2),
                    "non_self_edge_count": int(off_diagonal.sum().item() // 2),
                    "isolated_candidate_count": int(isolated.sum().item()),
                    "isolated_logits_exact": bool(isolated_exact),
                }
            )
    return output


__all__ = [
    "SameFamilyGraphConfig",
    "score_same_family_graph_records",
]
