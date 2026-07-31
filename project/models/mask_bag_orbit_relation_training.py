from __future__ import annotations

"""Image-label-only R4 orbit-averaged critical-relation fitting."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_critical_relation_training import (
    _padded_batch,
    initial_critical_relation_state,
)
from models.mask_bag_relational_selector import CriticalRelationResidual
from models.rad_dino_mask_bag_mil import (
    image_bag_loss,
    self_guided_instance_loss,
    smooth_mil_pool,
)


@dataclass(frozen=True)
class OrbitRelationTrainingConfig:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    instance_loss_weight: float = 0.25
    instance_warmup_epochs: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer controls are invalid")
        if self.hidden_dim < 2 or self.instance_loss_weight < 0:
            raise ValueError("model/loss controls are invalid")
        if not 0 <= self.instance_warmup_epochs < self.epochs:
            raise ValueError("instance warm-up must precede the final epoch")


def orbit_average(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("orbit-average tensors must be aligned and finite")
    return 0.5 * (first + second)


def _orbit_inputs(
    original: torch.Tensor,
    flipped: torch.Tensor,
    valid: torch.Tensor,
    frozen_base_scorer: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    descriptors = orbit_average(original, flipped)
    with torch.inference_mode():
        original_base, _ = frozen_base_scorer.score_descriptors(original, valid)
        flipped_base, _ = frozen_base_scorer.score_descriptors(flipped, valid)
        base = orbit_average(original_base, flipped_base)
    return descriptors, base


def audit_orbit_initialization_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: CriticalRelationResidual,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, int | bool]:
    if not records or batch_size <= 0:
        raise ValueError("R4 initial audit requires records and positive batch size")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter.eval()
    audited = 0
    candidates = 0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            indices = np.arange(start, min(start + batch_size, len(records)))
            original, flipped, valid, _ = _padded_batch(records, indices, device)
            descriptors, base = _orbit_inputs(original, flipped, valid, frozen_base_scorer)
            swapped_descriptors, swapped_base = _orbit_inputs(
                flipped, original, valid, frozen_base_scorer
            )
            combined, critical, residual = adapter(descriptors, base, valid)
            swapped, swapped_critical, swapped_residual = adapter(
                swapped_descriptors, swapped_base, valid
            )
            if (
                not torch.equal(combined, base)
                or torch.count_nonzero(residual).item() != 0
                or not torch.equal(swapped, combined)
                or not torch.equal(swapped_residual, residual)
                or not torch.equal(swapped_critical, critical)
            ):
                raise RuntimeError("R4 zero identity or view-swap invariance failed")
            audited += len(indices)
            candidates += int(valid.sum().item())
    return {
        "records": audited,
        "candidates": candidates,
        "zero_residual_exact": True,
        "combined_equals_averaged_frozen_base_exact": True,
        "view_swap_candidate_logits_exact": True,
        "view_swap_critical_indices_exact": True,
    }


def train_orbit_relation_adapter(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    *,
    descriptor_dim: int,
    bag_temperature: float,
    training_config: OrbitRelationTrainingConfig,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
) -> tuple[CriticalRelationResidual, list[dict[str, float]]]:
    if not records or bag_temperature <= 0:
        raise ValueError("R4 training records/temperature are invalid")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter = CriticalRelationResidual(descriptor_dim, training_config.hidden_dim).to(device)
    adapter.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        adapter.train()
        order = np.random.default_rng(training_config.seed + epoch).permutation(len(records))
        sums = {"total": 0.0, "image": 0.0, "instance": 0.0}
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            original, flipped, valid, labels = _padded_batch(records, indices, device)
            descriptors, base = _orbit_inputs(original, flipped, valid, frozen_base_scorer)
            logits, _, residual = adapter(descriptors, base, valid)
            if not torch.equal(logits, base + residual):
                raise RuntimeError("R4 residual changed averaged-base semantics")
            bag_logits = smooth_mil_pool(logits, valid, temperature=bag_temperature)
            image_loss = image_bag_loss(bag_logits, labels)
            if epoch > training_config.instance_warmup_epochs:
                instance_loss = self_guided_instance_loss(logits, valid, labels)
            else:
                instance_loss = logits.sum() * 0.0
            total = image_loss + training_config.instance_loss_weight * instance_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (("total", total), ("image", image_loss), ("instance", instance_loss)):
                sums[key] += float(value.detach().item())
            batches += 1
        history.append(
            {"epoch": float(epoch), **{key: value / batches for key, value in sums.items()}}
        )
    if any(parameter.requires_grad for parameter in frozen_base_scorer.parameters()):
        raise RuntimeError("R4 base scorer became trainable")
    return adapter, history


def score_orbit_relation_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: CriticalRelationResidual,
    *,
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if batch_size <= 0 or bag_temperature <= 0:
        raise ValueError("R4 scoring controls must be positive")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        original, flipped, valid, _ = _padded_batch(records, indices, device)
        with torch.inference_mode():
            descriptors, base = _orbit_inputs(original, flipped, valid, frozen_base_scorer)
            swapped_descriptors, swapped_base = _orbit_inputs(flipped, original, valid, frozen_base_scorer)
            logits, critical, _ = adapter(descriptors, base, valid)
            swapped_logits, swapped_critical, _ = adapter(
                swapped_descriptors, swapped_base, valid
            )
            swap_exact = torch.all(logits == swapped_logits, dim=1) & (critical == swapped_critical)
            bag_logits = smooth_mil_pool(logits, valid, temperature=bag_temperature)
        for row, record_index in enumerate(indices):
            count = int(valid[row].sum().item())
            output.append(
                {
                    "image_id": records[int(record_index)]["image_id"],
                    "candidate_logits": logits[row, :count].float().cpu().numpy(),
                    "bag_logit": float(bag_logits[row].item()),
                    "bag_probability": float(torch.sigmoid(bag_logits[row]).item()),
                    "candidate_count": count,
                    "view_swap_exact": bool(swap_exact[row].item()),
                }
            )
    return output


__all__ = [
    "OrbitRelationTrainingConfig",
    "audit_orbit_initialization_records",
    "initial_critical_relation_state",
    "orbit_average",
    "score_orbit_relation_records",
    "train_orbit_relation_adapter",
]
