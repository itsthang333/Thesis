from __future__ import annotations

"""Image-label-only label-granularity residuals for mask-bag selection.

The module has no dataset, image, annotation, or evaluation dependency.  It
accepts frozen candidate descriptors and frozen binary candidate logits, then
implements a matched coarse/hierarchical MIL pair.  No API accepts instance or
segmentation targets.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.rad_dino_mask_bag_mil import smooth_mil_pool


BENIGN_SUBTYPE_INDICES = tuple(range(7))
MALIGNANT_SUBTYPE_INDICES = (7, 8)
SUBTYPE_COUNT = 9


@dataclass(frozen=True)
class LabelGranularityConfig:
    descriptor_dim: int = 1156
    hidden_dim: int = 128
    subtype_count: int = SUBTYPE_COUNT
    bag_temperature: float = 0.20
    dropout: float = 0.10
    hierarchy_binary_weight: float = 1.0 / 3.0
    hierarchy_pathology_weight: float = 1.0 / 3.0
    hierarchy_subtype_weight: float = 1.0 / 3.0
    consistency_weight: float = 0.10
    residual_drift_weight: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.descriptor_dim <= 0 or self.hidden_dim < 2:
            raise ValueError("descriptor and hidden dimensions must be positive")
        if self.subtype_count != SUBTYPE_COUNT:
            raise ValueError("BTXRD label-granularity arm requires nine tumor subtypes")
        if self.bag_temperature <= 0:
            raise ValueError("bag_temperature must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        hierarchy_total = (
            self.hierarchy_binary_weight
            + self.hierarchy_pathology_weight
            + self.hierarchy_subtype_weight
        )
        if not math.isclose(hierarchy_total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("hierarchy loss weights must sum to one")
        if min(
            self.hierarchy_binary_weight,
            self.hierarchy_pathology_weight,
            self.hierarchy_subtype_weight,
            self.consistency_weight,
            self.residual_drift_weight,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")


def _validate_bag_inputs(
    values: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    expected_rank: int,
) -> torch.Tensor:
    if values.ndim != expected_rank:
        raise ValueError(f"values must have rank {expected_rank}")
    if candidate_valid.shape != values.shape[:2]:
        raise ValueError("candidate validity must align with the first two dimensions")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every bag must contain at least one valid candidate")
    if not torch.isfinite(values).all():
        raise ValueError("bag values must be finite")
    return valid


def center_valid_candidates(
    values: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Subtract the valid-candidate mean and zero all padded positions."""

    valid = _validate_bag_inputs(values, candidate_valid, expected_rank=3)
    weights = valid[:, :, None].to(values.dtype)
    count = weights.sum(dim=1, keepdim=True)
    mean = (values * weights).sum(dim=1, keepdim=True) / count
    return (values - mean) * weights


def normalized_logmeanexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[dim] <= 0:
        raise ValueError("log-mean-exp requires a nonempty dimension")
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


class LabelGranularityResidual(nn.Module):
    """Nine-column candidate residual with an exactly-zero initial output."""

    def __init__(self, config: LabelGranularityConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.descriptor_dim),
            nn.Linear(config.descriptor_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.output = nn.Linear(config.hidden_dim, config.subtype_count)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        centered = center_valid_candidates(descriptors, candidate_valid)
        raw = self.output(self.trunk(centered))
        return center_valid_candidates(raw, candidate_valid)


def coarse_candidate_logits(
    base_candidate_logits: torch.Tensor,
    subtype_residuals: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    valid = _validate_bag_inputs(subtype_residuals, candidate_valid, expected_rank=3)
    if subtype_residuals.shape[-1] != SUBTYPE_COUNT:
        raise ValueError("subtype residual count mismatch")
    if base_candidate_logits.shape != subtype_residuals.shape[:2]:
        raise ValueError("base candidate logits must align with residuals")
    result = base_candidate_logits + subtype_residuals.mean(dim=-1)
    return result.masked_fill(~valid, 0.0)


def subtype_bag_logits(
    base_candidate_logits: torch.Tensor,
    subtype_residuals: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    valid = _validate_bag_inputs(subtype_residuals, candidate_valid, expected_rank=3)
    if base_candidate_logits.shape != subtype_residuals.shape[:2]:
        raise ValueError("base candidate logits must align with residuals")
    batch, _candidates, classes = subtype_residuals.shape
    candidate = base_candidate_logits[:, :, None] + subtype_residuals
    pooled = smooth_mil_pool(
        candidate.transpose(1, 2).reshape(batch * classes, -1),
        valid[:, None, :].expand(-1, classes, -1).reshape(batch * classes, -1),
        temperature=temperature,
    )
    return pooled.reshape(batch, classes)


def pathology_bag_logits(subtype_logits: torch.Tensor) -> torch.Tensor:
    if subtype_logits.ndim != 2 or subtype_logits.shape[1] != SUBTYPE_COUNT:
        raise ValueError("subtype logits must have shape [B,9]")
    benign = normalized_logmeanexp(
        subtype_logits[:, BENIGN_SUBTYPE_INDICES], dim=1
    )
    malignant = normalized_logmeanexp(
        subtype_logits[:, MALIGNANT_SUBTYPE_INDICES], dim=1
    )
    return torch.stack((benign, malignant), dim=1)


def entropy_route_strength(subtype_logits: torch.Tensor) -> torch.Tensor:
    if subtype_logits.ndim != 2 or subtype_logits.shape[1] != SUBTYPE_COUNT:
        raise ValueError("subtype logits must have shape [B,9]")
    probabilities = torch.softmax(subtype_logits, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    strength = 1.0 - entropy / math.log(SUBTYPE_COUNT)
    return strength.clamp(0.0, 1.0)


def entropy_routed_candidate_logits(
    base_candidate_logits: torch.Tensor,
    subtype_residuals: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = _validate_bag_inputs(subtype_residuals, candidate_valid, expected_rank=3)
    bag_logits = subtype_bag_logits(
        base_candidate_logits,
        subtype_residuals,
        valid,
        temperature=temperature,
    )
    predicted = bag_logits.argmax(dim=1)
    route = entropy_route_strength(bag_logits)
    coarse = subtype_residuals.mean(dim=-1)
    selected = subtype_residuals.gather(
        2,
        predicted[:, None, None].expand(-1, subtype_residuals.shape[1], 1),
    ).squeeze(-1)
    routed = base_candidate_logits + coarse + route[:, None] * (selected - coarse)
    return routed.masked_fill(~valid, 0.0), bag_logits, predicted, route


def inverse_sqrt_subtype_weights(counts: torch.Tensor) -> torch.Tensor:
    values = counts.to(dtype=torch.float64)
    if values.shape != (SUBTYPE_COUNT,) or not torch.isfinite(values).all():
        raise ValueError("subtype counts must be a finite length-nine vector")
    if (values <= 0).any():
        raise ValueError("every subtype must occur in the training split")
    weights = values.rsqrt()
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32, device=counts.device)


def binary_bag_loss(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    tumor_labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bag_logits = smooth_mil_pool(
        candidate_logits,
        candidate_valid,
        temperature=temperature,
    )
    labels = tumor_labels.to(dtype=bag_logits.dtype).reshape_as(bag_logits)
    return F.binary_cross_entropy_with_logits(bag_logits, labels), bag_logits


def label_granularity_losses(
    *,
    base_candidate_logits: torch.Tensor,
    residuals: torch.Tensor,
    flipped_residuals: torch.Tensor,
    candidate_valid: torch.Tensor,
    tumor_labels: torch.Tensor,
    tumor_type_labels: torch.Tensor,
    subtype_class_weights: torch.Tensor,
    config: LabelGranularityConfig,
    hierarchical: bool,
) -> dict[str, torch.Tensor]:
    if residuals.shape != flipped_residuals.shape:
        raise ValueError("original and flipped residuals must align")
    valid = _validate_bag_inputs(residuals, candidate_valid, expected_rank=3)
    labels = tumor_labels.reshape(-1).long()
    types = tumor_type_labels.reshape(-1).long()
    if labels.shape[0] != residuals.shape[0] or types.shape != labels.shape:
        raise ValueError("image labels must align with candidate bags")
    if not torch.equal(labels.bool(), types.ne(0)):
        raise ValueError("tumor and tumor-type labels are inconsistent")
    if types.min() < 0 or types.max() > SUBTYPE_COUNT:
        raise ValueError("tumor-type labels must lie in [0,9]")

    coarse = coarse_candidate_logits(base_candidate_logits, residuals, valid)
    binary, binary_logits = binary_bag_loss(
        coarse,
        valid,
        labels,
        temperature=config.bag_temperature,
    )
    valid3 = valid[:, :, None].expand_as(residuals)
    consistency = F.smooth_l1_loss(
        residuals[valid3], flipped_residuals[valid3]
    )
    drift = residuals[valid3].square().mean()
    zero = residuals.sum() * 0.0
    pathology = zero
    subtype = zero
    if hierarchical:
        positive = labels.bool()
        if not positive.any():
            raise ValueError("hierarchical batches must contain a tumor image")
        subtype_logits = subtype_bag_logits(
            base_candidate_logits,
            residuals,
            valid,
            temperature=config.bag_temperature,
        )[positive]
        subtype_targets = types[positive] - 1
        subtype = F.cross_entropy(
            subtype_logits,
            subtype_targets,
            weight=subtype_class_weights.to(subtype_logits.device, subtype_logits.dtype),
        )
        pathology_targets = types[positive].ge(8).long()
        pathology = F.cross_entropy(
            pathology_bag_logits(subtype_logits), pathology_targets
        )
        supervised = (
            config.hierarchy_binary_weight * binary
            + config.hierarchy_pathology_weight * pathology
            + config.hierarchy_subtype_weight * subtype
        )
    else:
        supervised = binary
    total = (
        supervised
        + config.consistency_weight * consistency
        + config.residual_drift_weight * drift
    )
    return {
        "total": total,
        "binary": binary,
        "pathology": pathology,
        "subtype": subtype,
        "consistency": consistency,
        "drift": drift,
        "binary_bag_logits": binary_logits,
    }


__all__ = [
    "BENIGN_SUBTYPE_INDICES",
    "LabelGranularityConfig",
    "LabelGranularityResidual",
    "MALIGNANT_SUBTYPE_INDICES",
    "SUBTYPE_COUNT",
    "binary_bag_loss",
    "center_valid_candidates",
    "coarse_candidate_logits",
    "entropy_route_strength",
    "entropy_routed_candidate_logits",
    "inverse_sqrt_subtype_weights",
    "label_granularity_losses",
    "normalized_logmeanexp",
    "pathology_bag_logits",
    "subtype_bag_logits",
]
