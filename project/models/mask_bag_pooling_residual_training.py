from __future__ import annotations

"""Matched descriptor-residual training for the S1 pooling experiment."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.mask_bag_relational_selector import family_balanced_smooth_mil_pool
from models.mask_bag_residual_objective import ResidualObjectiveConfig
from models.rad_dino_mask_bag_mil import smooth_mil_pool


POOL_MODES = ("standard", "family_balanced")


class DescriptorOnlyResidual(nn.Module):
    """Zero-initialized descriptor correction shared by both S1 arms."""

    def __init__(self, descriptor_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if descriptor_dim <= 0 or hidden_dim < 2:
            raise ValueError("descriptor dimensions must be positive")
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError("descriptors have an invalid shape")
        if base_logits.shape != descriptors.shape[:2]:
            raise ValueError("base logits must align with descriptors")
        if candidate_valid.shape != base_logits.shape:
            raise ValueError("candidate validity must align with logits")
        if not torch.isfinite(descriptors).all() or not torch.isfinite(
            base_logits
        ).all():
            raise ValueError("residual inputs must be finite")
        valid = candidate_valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("every bag must contain a valid candidate")
        residual = self.network(descriptors).squeeze(-1)
        residual = residual * valid.to(residual.dtype)
        combined = (base_logits + residual) * valid.to(base_logits.dtype)
        return combined, residual


@dataclass(frozen=True)
class PoolingResidualTrainingConfig:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer controls are invalid")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least two")


def _pool(
    logits: torch.Tensor,
    valid: torch.Tensor,
    family_ids: torch.Tensor,
    *,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    if mode == "standard":
        return smooth_mil_pool(logits, valid, temperature=temperature)
    if mode == "family_balanced":
        pooled, _family_logits = family_balanced_smooth_mil_pool(
            logits,
            valid,
            family_ids,
            temperature=temperature,
        )
        return pooled
    raise ValueError(f"unsupported pooling mode: {mode}")


def pooling_residual_objective(
    original_base_logits: torch.Tensor,
    flipped_base_logits: torch.Tensor,
    original_residual: torch.Tensor,
    flipped_residual: torch.Tensor,
    candidate_valid: torch.Tensor,
    family_ids: torch.Tensor,
    image_labels: torch.Tensor,
    config: ResidualObjectiveConfig,
    *,
    pool_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply identical residual losses with only the bag pool changed."""

    shape = original_base_logits.shape
    if (
        original_base_logits.ndim != 2
        or flipped_base_logits.shape != shape
        or original_residual.shape != shape
        or flipped_residual.shape != shape
        or candidate_valid.shape != shape
        or family_ids.shape != shape
    ):
        raise ValueError("S1 objective tensors must share shape [B,N]")
    labels = image_labels.to(dtype=original_base_logits.dtype).reshape(-1)
    if labels.shape != (shape[0],):
        raise ValueError("image labels must have shape [B]")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all() or (family_ids[valid] < 0).any():
        raise ValueError("valid candidates require nonnegative family IDs")
    values = (
        original_base_logits,
        flipped_base_logits,
        original_residual,
        flipped_residual,
        labels,
    )
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("S1 objective inputs must be finite")

    original_logits = original_base_logits.detach() + original_residual
    flipped_logits = flipped_base_logits.detach() + flipped_residual
    original_bag = _pool(
        original_logits,
        valid,
        family_ids,
        mode=pool_mode,
        temperature=config.bag_temperature,
    )
    flipped_bag = _pool(
        flipped_logits,
        valid,
        family_ids,
        mode=pool_mode,
        temperature=config.bag_temperature,
    )
    image_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(original_bag, labels)
        + F.binary_cross_entropy_with_logits(flipped_bag, labels)
    )
    consistency = F.smooth_l1_loss(
        torch.sigmoid(original_logits[valid]),
        torch.sigmoid(flipped_logits[valid]),
    )
    drift = 0.5 * (
        original_residual[valid].square().mean()
        + flipped_residual[valid].square().mean()
    )
    total = (
        image_loss
        + config.consistency_weight * consistency
        + config.residual_drift_weight * drift
    )
    return total, {
        "image": image_loss,
        "consistency": consistency,
        "residual_drift": drift,
        "original_bag_logit_mean": original_bag.mean(),
        "flipped_bag_logit_mean": flipped_bag.mean(),
    }


def _padded_batch(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    selected = [records[int(index)] for index in indices]
    maximum = max(len(record["descriptors"]) for record in selected)
    descriptor_dim = int(np.asarray(selected[0]["descriptors"]).shape[1])
    original = np.zeros((len(selected), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(original)
    valid = np.zeros((len(selected), maximum), dtype=bool)
    families = np.full((len(selected), maximum), -1, dtype=np.int64)
    labels = np.zeros(len(selected), dtype=np.float32)
    for row_index, record in enumerate(selected):
        descriptor = np.asarray(record["descriptors"], dtype=np.float32)
        flipped_descriptor = np.asarray(
            record["flipped_descriptors"], dtype=np.float32
        )
        family = np.asarray(record["family_ids"], dtype=np.int64)
        count = len(descriptor)
        if (
            descriptor.shape != (count, descriptor_dim)
            or flipped_descriptor.shape != descriptor.shape
            or family.shape != (count,)
            or np.any(family < 0)
            or not np.isfinite(descriptor).all()
            or not np.isfinite(flipped_descriptor).all()
        ):
            raise ValueError("S1 record arrays are invalid")
        original[row_index, :count] = descriptor
        flipped[row_index, :count] = flipped_descriptor
        valid[row_index, :count] = True
        families[row_index, :count] = family
        labels[row_index] = float(record["label"])
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (original, flipped, valid, families, labels)
    )


def train_pooling_residual_adapter(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    *,
    descriptor_dim: int,
    pool_mode: str,
    objective_config: ResidualObjectiveConfig,
    training_config: PoolingResidualTrainingConfig,
    device: torch.device,
    initial_adapter_state: Mapping[str, torch.Tensor],
) -> tuple[DescriptorOnlyResidual, list[dict[str, float]]]:
    """Fit one member of the matched standard/family-balanced pair."""

    if not records or pool_mode not in POOL_MODES:
        raise ValueError("S1 records/pool mode are invalid")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter = DescriptorOnlyResidual(
        descriptor_dim=descriptor_dim,
        hidden_dim=training_config.hidden_dim,
    ).to(device)
    adapter.load_state_dict(initial_adapter_state, strict=True)
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
        sums = {
            "total": 0.0,
            "image": 0.0,
            "consistency": 0.0,
            "residual_drift": 0.0,
        }
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            original, flipped, valid, families, labels = _padded_batch(
                records,
                indices,
                device,
            )
            with torch.inference_mode():
                original_base, _ = frozen_base_scorer.score_descriptors(
                    original, valid
                )
                flipped_base, _ = frozen_base_scorer.score_descriptors(
                    flipped, valid
                )
            original_logits, original_residual = adapter(
                original, original_base, valid
            )
            flipped_logits, flipped_residual = adapter(
                flipped, flipped_base, valid
            )
            total, details = pooling_residual_objective(
                original_base,
                flipped_base,
                original_residual,
                flipped_residual,
                valid,
                families,
                labels,
                objective_config,
                pool_mode=pool_mode,
            )
            if not torch.equal(
                original_logits, original_base + original_residual
            ) or not torch.equal(
                flipped_logits, flipped_base + flipped_residual
            ):
                raise RuntimeError("S1 residual changed frozen-base semantics")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key in sums:
                value = total if key == "total" else details[key]
                sums[key] += float(value.detach().item())
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / batches for key, value in sums.items()},
            }
        )
    if any(parameter.requires_grad for parameter in frozen_base_scorer.parameters()):
        raise RuntimeError("S1 base scorer became trainable")
    return adapter, history


def score_pooling_residual_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: DescriptorOnlyResidual,
    *,
    pool_mode: str,
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return all candidate logits for one matched S1 arm."""

    if batch_size <= 0 or pool_mode not in POOL_MODES:
        raise ValueError("S1 scoring controls are invalid")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        original, flipped, valid, families, _labels = _padded_batch(
            records,
            indices,
            device,
        )
        with torch.inference_mode():
            original_base, _ = frozen_base_scorer.score_descriptors(
                original, valid
            )
            flipped_base, _ = frozen_base_scorer.score_descriptors(
                flipped, valid
            )
            original_logits, _ = adapter(original, original_base, valid)
            flipped_logits, _ = adapter(flipped, flipped_base, valid)
            logits = 0.5 * (original_logits + flipped_logits)
            bag_logits = _pool(
                logits,
                valid,
                families,
                mode=pool_mode,
                temperature=bag_temperature,
            )
        for row, record_index in enumerate(indices):
            count = int(valid[row].sum().item())
            output.append(
                {
                    "image_id": records[int(record_index)]["image_id"],
                    "candidate_logits": logits[row, :count]
                    .float()
                    .cpu()
                    .numpy(),
                    "bag_logit": float(bag_logits[row].item()),
                    "bag_probability": float(torch.sigmoid(bag_logits[row]).item()),
                    "candidate_count": count,
                }
            )
    return output


__all__ = [
    "DescriptorOnlyResidual",
    "POOL_MODES",
    "PoolingResidualTrainingConfig",
    "pooling_residual_objective",
    "score_pooling_residual_records",
    "train_pooling_residual_adapter",
]
