from __future__ import annotations

"""Direct instance self-training primitives for the prospective S7 arm.

The module is dataset agnostic. It accepts frozen candidate descriptors,
frozen base candidate logits, image-level labels, and GT-blind family IDs. No
API accepts segmentation targets or validation metrics.
"""

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.mask_bag_label_granularity import center_valid_candidates


@dataclass(frozen=True)
class GlobalLocalInstanceConfig:
    descriptor_dim: int = 1156
    hidden_dim: int = 128
    dropout: float = 0.10
    start_positive_mass: float = 0.50
    target_positive_mass: float = 0.15
    mass_transition_epochs: int = 20
    total_epochs: int = 40
    consistency_weight: float = 0.10
    residual_drift_weight: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.descriptor_dim <= 0 or self.hidden_dim < 2:
            raise ValueError("descriptor and hidden dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.target_positive_mass < 1.0:
            raise ValueError("target positive mass must lie in (0,1)")
        if not self.target_positive_mass <= self.start_positive_mass < 1.0:
            raise ValueError("start positive mass must be in [target,1)")
        if self.mass_transition_epochs <= 0:
            raise ValueError("mass transition epochs must be positive")
        if self.total_epochs < self.mass_transition_epochs:
            raise ValueError("total epochs cannot precede the mass transition")
        if min(self.consistency_weight, self.residual_drift_weight) < 0.0:
            raise ValueError("loss weights must be nonnegative")


class GlobalLocalInstanceResidual(nn.Module):
    """Scalar candidate residual with exact Geometry-v3 identity at init."""

    def __init__(self, config: GlobalLocalInstanceConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.descriptor_dim),
            nn.Linear(config.descriptor_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.output = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        centered = center_valid_candidates(descriptors, candidate_valid)
        residual = self.output(self.trunk(centered)).squeeze(-1)
        return residual.masked_fill(~candidate_valid.bool(), 0.0)


def adaptive_positive_mass(
    epoch: int,
    config: GlobalLocalInstanceConfig,
) -> float:
    """External-source schedule: 0.50 -> 0.15, then fixed."""

    if epoch < 0 or epoch > config.total_epochs:
        raise ValueError("epoch lies outside the frozen training schedule")
    fraction = min(float(epoch) / float(config.mass_transition_epochs), 1.0)
    return float(
        config.start_positive_mass
        + fraction * (config.target_positive_mass - config.start_positive_mass)
    )


def equal_family_candidate_weights(family_ids: Sequence[Any]) -> np.ndarray:
    """Give one image unit mass, split equally over families and candidates."""

    values = np.asarray(list(family_ids), dtype=object).reshape(-1)
    if values.size == 0:
        raise ValueError("a candidate bag cannot be empty")
    normalized = [str(value) for value in values]
    families = sorted(set(normalized))
    if any(value == "" for value in normalized):
        raise ValueError("candidate family IDs cannot be empty")
    result = np.zeros(values.size, dtype=np.float64)
    for family in families:
        positions = np.asarray(
            [index for index, value in enumerate(normalized) if value == family],
            dtype=np.int64,
        )
        result[positions] = 1.0 / (len(families) * positions.size)
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("family/candidate weights do not sum to one image")
    # Keep the audit/projection weights in float64. Training may cast them to
    # the model dtype later, but the frozen global marginal must not inherit a
    # float32 summation error before the deterministic projection.
    return result


def project_weighted_sigmoid_mass(
    logits: np.ndarray,
    weights: np.ndarray,
    *,
    target_mass: float,
    iterations: int = 96,
) -> tuple[np.ndarray, float, float]:
    """Entropy-project binary probabilities to one weighted global marginal.

    The I-projection for binary logits is an additive scalar bias. Bisection is
    used in float64 so the result is deterministic and independent of GPU math.
    """

    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.size == 0 or scores.shape != mass.shape:
        raise ValueError("projection logits and weights must be aligned and nonempty")
    if not np.isfinite(scores).all() or not np.isfinite(mass).all():
        raise ValueError("projection inputs must be finite")
    if (mass <= 0.0).any():
        raise ValueError("projection weights must be strictly positive")
    if not np.isclose(mass.sum(), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("projection weights must sum to one")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target mass must lie in (0,1)")
    if iterations < 32:
        raise ValueError("projection needs at least 32 bisection iterations")

    def sigmoid(values: np.ndarray) -> np.ndarray:
        result = np.empty_like(values, dtype=np.float64)
        nonnegative = values >= 0.0
        result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
        exponent = np.exp(values[~nonnegative])
        result[~nonnegative] = exponent / (1.0 + exponent)
        return result

    lower = -128.0 - float(np.max(np.abs(scores)))
    upper = 128.0 + float(np.max(np.abs(scores)))
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        realized = float(np.dot(mass, sigmoid(scores + midpoint)))
        if realized < target_mass:
            lower = midpoint
        else:
            upper = midpoint
    bias = 0.5 * (lower + upper)
    probabilities = sigmoid(scores + bias)
    realized = float(np.dot(mass, probabilities))
    if not math.isclose(realized, target_mass, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError("weighted positive-mass projection did not converge")
    return probabilities.astype(np.float32), float(bias), realized


def build_global_local_soft_targets(
    logits_by_bag: Sequence[np.ndarray],
    labels: Sequence[int],
    family_ids_by_bag: Sequence[Sequence[Any]],
    *,
    target_mass: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Build all-instance soft targets from image labels only.

    The global projection is over positive-bag candidates with equal image,
    family and candidate mass. The local constraint then forces the current
    maximum-logit candidate in each positive bag to target one. Negative-bag
    candidate targets are exact zeros.
    """

    if not (
        len(logits_by_bag) == len(labels) == len(family_ids_by_bag)
        and len(labels) > 0
    ):
        raise ValueError("target-builder bag inputs must be aligned and nonempty")
    bag_logits: list[np.ndarray] = []
    bag_weights: list[np.ndarray] = []
    positive_indices: list[int] = []
    for bag_index, (logits, label, family_ids) in enumerate(
        zip(logits_by_bag, labels, family_ids_by_bag)
    ):
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("every candidate bag must contain finite logits")
        if int(label) not in (0, 1):
            raise ValueError("image labels must be binary")
        weights = equal_family_candidate_weights(family_ids)
        if weights.shape != values.shape:
            raise ValueError("family IDs must align with candidate logits")
        bag_logits.append(values)
        bag_weights.append(weights)
        if int(label) == 1:
            positive_indices.append(bag_index)
    if not positive_indices:
        raise ValueError("global/local targets require at least one positive bag")

    positive_bag_count = len(positive_indices)
    flat_logits = np.concatenate([bag_logits[index] for index in positive_indices])
    flat_weights = np.concatenate(
        [bag_weights[index] / positive_bag_count for index in positive_indices]
    ).astype(np.float64)
    projected, bias, projected_mass = project_weighted_sigmoid_mass(
        flat_logits,
        flat_weights,
        target_mass=target_mass,
    )

    targets = [np.zeros_like(values, dtype=np.float32) for values in bag_logits]
    cursor = 0
    local_top_indices: dict[int, int] = {}
    for bag_index in positive_indices:
        count = bag_logits[bag_index].size
        targets[bag_index] = projected[cursor : cursor + count].copy()
        cursor += count
        top = int(np.argmax(bag_logits[bag_index]))
        targets[bag_index][top] = np.float32(1.0)
        local_top_indices[bag_index] = top
    if cursor != projected.size:
        raise RuntimeError("projected target reconstruction is incomplete")

    realized_after_local = 0.0
    for bag_index in positive_indices:
        realized_after_local += float(
            np.dot(
                bag_weights[bag_index].astype(np.float64),
                targets[bag_index].astype(np.float64),
            )
        ) / positive_bag_count
    diagnostics: dict[str, Any] = {
        "bags": len(labels),
        "positive_bags": positive_bag_count,
        "negative_bags": len(labels) - positive_bag_count,
        "candidates": int(sum(values.size for values in bag_logits)),
        "positive_bag_candidates": int(flat_logits.size),
        "target_positive_mass": float(target_mass),
        "projection_bias": float(bias),
        "projected_mass_before_local": float(projected_mass),
        "realized_mass_after_local": float(realized_after_local),
        "locally_forced_candidates": positive_bag_count,
        "local_top_indices": local_top_indices,
    }
    return targets, bag_weights, diagnostics


def combined_instance_logits(
    base_candidate_logits: torch.Tensor,
    residuals: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    if base_candidate_logits.shape != residuals.shape:
        raise ValueError("base candidate logits and residuals must align")
    if candidate_valid.shape != residuals.shape:
        raise ValueError("candidate validity must align with candidate logits")
    if not torch.isfinite(base_candidate_logits).all() or not torch.isfinite(
        residuals
    ).all():
        raise ValueError("candidate logits must be finite")
    return (base_candidate_logits + residuals).masked_fill(
        ~candidate_valid.bool(), 0.0
    )


def global_local_instance_losses(
    *,
    original_logits: torch.Tensor,
    flipped_logits: torch.Tensor,
    original_residuals: torch.Tensor,
    flipped_residuals: torch.Tensor,
    soft_targets: torch.Tensor,
    candidate_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
    config: GlobalLocalInstanceConfig,
) -> dict[str, torch.Tensor]:
    tensors = (
        flipped_logits,
        original_residuals,
        flipped_residuals,
        soft_targets,
        candidate_weights,
        candidate_valid,
    )
    if any(tensor.shape != original_logits.shape for tensor in tensors):
        raise ValueError("all instance-loss tensors must align")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every instance-loss bag must be nonempty")
    if not torch.isfinite(original_logits).all() or not torch.isfinite(
        flipped_logits
    ).all():
        raise ValueError("instance logits must be finite")
    if ((soft_targets < 0.0) | (soft_targets > 1.0)).any():
        raise ValueError("soft targets must lie in [0,1]")
    if (candidate_weights[valid] <= 0.0).any() or (
        candidate_weights[~valid] != 0.0
    ).any():
        raise ValueError("candidate weights must be positive exactly on valid entries")
    row_mass = candidate_weights.sum(dim=1)
    if not torch.allclose(row_mass, torch.ones_like(row_mass), atol=1.0e-6, rtol=0.0):
        raise ValueError("candidate weights must sum to one per image")

    first = F.binary_cross_entropy_with_logits(
        original_logits,
        soft_targets.to(original_logits.dtype),
        reduction="none",
    )
    second = F.binary_cross_entropy_with_logits(
        flipped_logits,
        soft_targets.to(flipped_logits.dtype),
        reduction="none",
    )
    weights = candidate_weights.to(original_logits.dtype)
    instance = (0.5 * (first + second) * weights).sum(dim=1).mean()
    consistency = (
        F.smooth_l1_loss(
            original_residuals,
            flipped_residuals,
            reduction="none",
        )
        * weights
    ).sum(dim=1).mean()
    drift = (
        0.5
        * (original_residuals.square() + flipped_residuals.square())
        * weights
    ).sum(dim=1).mean()
    total = (
        instance
        + config.consistency_weight * consistency
        + config.residual_drift_weight * drift
    )
    return {
        "total": total,
        "instance": instance,
        "consistency": consistency,
        "drift": drift,
    }


__all__ = [
    "GlobalLocalInstanceConfig",
    "GlobalLocalInstanceResidual",
    "adaptive_positive_mass",
    "build_global_local_soft_targets",
    "combined_instance_logits",
    "equal_family_candidate_weights",
    "global_local_instance_losses",
    "project_weighted_sigmoid_mass",
]
