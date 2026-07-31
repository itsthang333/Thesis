from __future__ import annotations

"""Image-label-only R3 critical-relation residual fitting.

The immutable selector-cache descriptors and the frozen Geometry-v3 scorer are
the only inputs.  This module deliberately has no dataset, annotation,
segmentation-quality, lesion-size, or validation-selection interface.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_relational_selector import CriticalRelationResidual
from models.rad_dino_mask_bag_mil import (
    aligned_candidate_consistency_loss,
    image_bag_loss,
    self_guided_instance_loss,
    smooth_mil_pool,
)


@dataclass(frozen=True)
class CriticalRelationTrainingConfig:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    instance_loss_weight: float = 0.25
    consistency_weight: float = 0.10
    instance_warmup_epochs: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer controls are invalid")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least two")
        if self.instance_loss_weight < 0 or self.consistency_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if not 0 <= self.instance_warmup_epochs < self.epochs:
            raise ValueError("instance warm-up must precede the final epoch")


def initial_critical_relation_state(
    *, descriptor_dim: int, hidden_dim: int, seed: int
) -> dict[str, torch.Tensor]:
    """Create one deterministic CPU initial state shared by every device."""

    torch.manual_seed(seed)
    module = CriticalRelationResidual(descriptor_dim, hidden_dim)
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _padded_batch(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [records[int(index)] for index in indices]
    if not selected:
        raise ValueError("critical-relation batch cannot be empty")
    maximum = max(len(record["descriptors"]) for record in selected)
    descriptor_dim = int(np.asarray(selected[0]["descriptors"]).shape[1])
    original = np.zeros((len(selected), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(original)
    valid = np.zeros((len(selected), maximum), dtype=bool)
    labels = np.zeros(len(selected), dtype=np.float32)
    for row, record in enumerate(selected):
        count = len(record["candidate_indices"])
        arrays = (
            np.asarray(record["descriptors"], dtype=np.float32),
            np.asarray(record["flipped_descriptors"], dtype=np.float32),
        )
        if (
            arrays[0].shape != (count, descriptor_dim)
            or arrays[1].shape != arrays[0].shape
            or count <= 0
            or not all(np.isfinite(array).all() for array in arrays)
        ):
            raise ValueError("critical-relation descriptor alignment mismatch")
        original[row, :count] = arrays[0]
        flipped[row, :count] = arrays[1]
        valid[row, :count] = True
        labels[row] = float(record["label"])
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (original, flipped, valid, labels)
    )


def audit_zero_initialization_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: CriticalRelationResidual,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    """Verify exact identity and freeze base flip critical-index agreement."""

    if not records or batch_size <= 0:
        raise ValueError("initial audit requires records and a positive batch size")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter.eval()
    agreements = 0
    candidates = 0
    audited = 0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            indices = np.arange(start, min(start + batch_size, len(records)))
            original, flipped, valid, _ = _padded_batch(records, indices, device)
            original_base, _ = frozen_base_scorer.score_descriptors(original, valid)
            flipped_base, _ = frozen_base_scorer.score_descriptors(flipped, valid)
            original_combined, original_critical, original_residual = adapter(
                original, original_base, valid
            )
            flipped_combined, flipped_critical, flipped_residual = adapter(
                flipped, flipped_base, valid
            )
            if (
                not torch.equal(original_combined, original_base)
                or not torch.equal(flipped_combined, flipped_base)
                or not torch.count_nonzero(original_residual).item() == 0
                or not torch.count_nonzero(flipped_residual).item() == 0
            ):
                raise RuntimeError("R3 zero initialization is not exact identity")
            agreements += int((original_critical == flipped_critical).sum().item())
            audited += len(indices)
            candidates += int(valid.sum().item())
    return {
        "records": audited,
        "candidates": candidates,
        "zero_residual_exact": True,
        "combined_equals_frozen_base_exact": True,
        "base_flip_critical_agreement_count": agreements,
        "base_flip_critical_agreement": agreements / audited,
    }


def train_critical_relation_adapter(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    *,
    descriptor_dim: int,
    bag_temperature: float,
    training_config: CriticalRelationTrainingConfig,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
) -> tuple[CriticalRelationResidual, list[dict[str, float]]]:
    """Fit exactly one residual while the accepted base scorer stays frozen."""

    if not records:
        raise ValueError("critical-relation training records cannot be empty")
    if bag_temperature <= 0:
        raise ValueError("bag_temperature must be positive")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter = CriticalRelationResidual(
        descriptor_dim, training_config.hidden_dim
    ).to(device)
    adapter.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        adapter.train()
        order = np.random.default_rng(training_config.seed + epoch).permutation(
            len(records)
        )
        sums = {"total": 0.0, "image": 0.0, "instance": 0.0, "consistency": 0.0}
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            original, flipped, valid, labels = _padded_batch(records, indices, device)
            with torch.inference_mode():
                original_base, _ = frozen_base_scorer.score_descriptors(original, valid)
                flipped_base, _ = frozen_base_scorer.score_descriptors(flipped, valid)
            original_logits, _, original_residual = adapter(
                original, original_base, valid
            )
            flipped_logits, _, flipped_residual = adapter(
                flipped, flipped_base, valid
            )
            if not torch.equal(
                original_logits, original_base + original_residual
            ) or not torch.equal(flipped_logits, flipped_base + flipped_residual):
                raise RuntimeError("R3 residual changed frozen-base semantics")
            original_bag = smooth_mil_pool(
                original_logits, valid, temperature=bag_temperature
            )
            flipped_bag = smooth_mil_pool(
                flipped_logits, valid, temperature=bag_temperature
            )
            image_loss = 0.5 * (
                image_bag_loss(original_bag, labels)
                + image_bag_loss(flipped_bag, labels)
            )
            if epoch > training_config.instance_warmup_epochs:
                instance_loss = 0.5 * (
                    self_guided_instance_loss(original_logits, valid, labels)
                    + self_guided_instance_loss(flipped_logits, valid, labels)
                )
            else:
                instance_loss = original_logits.sum() * 0.0
            consistency = aligned_candidate_consistency_loss(
                original_logits, flipped_logits, valid
            )
            total = (
                image_loss
                + training_config.instance_loss_weight * instance_loss
                + training_config.consistency_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total),
                ("image", image_loss),
                ("instance", instance_loss),
                ("consistency", consistency),
            ):
                sums[key] += float(value.detach().item())
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / batches for key, value in sums.items()},
            }
        )
    if any(parameter.requires_grad for parameter in frozen_base_scorer.parameters()):
        raise RuntimeError("R3 base scorer became trainable")
    return adapter, history


def score_critical_relation_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: CriticalRelationResidual,
    *,
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score immutable aligned views and expose GT-blind ranking diagnostics."""

    if batch_size <= 0 or bag_temperature <= 0:
        raise ValueError("scoring controls must be positive")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        original, flipped, valid, _ = _padded_batch(records, indices, device)
        with torch.inference_mode():
            original_base, _ = frozen_base_scorer.score_descriptors(original, valid)
            flipped_base, _ = frozen_base_scorer.score_descriptors(flipped, valid)
            original_logits, original_critical, _ = adapter(
                original, original_base, valid
            )
            flipped_logits, flipped_critical, _ = adapter(
                flipped, flipped_base, valid
            )
            logits = 0.5 * (original_logits + flipped_logits)
            bag_logits = smooth_mil_pool(logits, valid, temperature=bag_temperature)
            original_selected = original_logits.masked_fill(~valid, -torch.inf).argmax(1)
            flipped_selected = flipped_logits.masked_fill(~valid, -torch.inf).argmax(1)
        for row, record_index in enumerate(indices):
            count = int(valid[row].sum().item())
            output.append(
                {
                    "image_id": records[int(record_index)]["image_id"],
                    "candidate_logits": logits[row, :count].float().cpu().numpy(),
                    "bag_logit": float(bag_logits[row].item()),
                    "bag_probability": float(torch.sigmoid(bag_logits[row]).item()),
                    "candidate_count": count,
                    "base_critical_agreement": bool(
                        original_critical[row] == flipped_critical[row]
                    ),
                    "final_selected_agreement": bool(
                        original_selected[row] == flipped_selected[row]
                    ),
                }
            )
    return output


__all__ = [
    "CriticalRelationTrainingConfig",
    "audit_zero_initialization_records",
    "initial_critical_relation_state",
    "score_critical_relation_records",
    "train_critical_relation_adapter",
]
