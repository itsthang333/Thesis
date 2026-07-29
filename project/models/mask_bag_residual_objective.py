from __future__ import annotations

"""Image-label-only objective for baseline-preserving selector adapters."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.rad_dino_mask_bag_mil import smooth_mil_pool


@dataclass(frozen=True)
class ResidualObjectiveConfig:
    bag_temperature: float = 0.20
    consistency_weight: float = 0.10
    residual_drift_weight: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.bag_temperature <= 0:
            raise ValueError("bag_temperature must be positive")
        if self.consistency_weight < 0 or self.residual_drift_weight < 0:
            raise ValueError("loss weights must be nonnegative")


def residual_arm_objective(
    original_base_logits: torch.Tensor,
    flipped_base_logits: torch.Tensor,
    original_residual: torch.Tensor,
    flipped_residual: torch.Tensor,
    candidate_valid: torch.Tensor,
    image_labels: torch.Tensor,
    config: ResidualObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a conservative adapter loss with no inferred instance targets.

    Positive and negative supervision enters only through the image-level bag
    label. Original/flip aligned candidates receive a consistency constraint,
    and the learned correction is kept near the independently frozen scorer.
    """

    shape = original_base_logits.shape
    if original_base_logits.ndim != 2 or flipped_base_logits.shape != shape:
        raise ValueError("original and flipped base logits must share shape [B,N]")
    if original_residual.shape != shape or flipped_residual.shape != shape:
        raise ValueError("residuals must align with candidate logits")
    if candidate_valid.shape != shape:
        raise ValueError("candidate_valid must align with candidate logits")
    labels = image_labels.to(dtype=original_base_logits.dtype).reshape(-1)
    if labels.shape != (shape[0],):
        raise ValueError("image_labels must have shape [B]")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every bag must contain at least one valid candidate")
    for value in (
        original_base_logits,
        flipped_base_logits,
        original_residual,
        flipped_residual,
        labels,
    ):
        if not torch.isfinite(value).all():
            raise ValueError("objective inputs must be finite")

    # Detaching here makes the baseline freeze a property of the objective,
    # rather than relying only on the caller's optimizer construction.
    original_logits = original_base_logits.detach() + original_residual
    flipped_logits = flipped_base_logits.detach() + flipped_residual
    original_bag = smooth_mil_pool(
        original_logits,
        valid,
        temperature=config.bag_temperature,
    )
    flipped_bag = smooth_mil_pool(
        flipped_logits,
        valid,
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


__all__ = ["ResidualObjectiveConfig", "residual_arm_objective"]
