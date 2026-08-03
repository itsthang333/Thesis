"""CAM-conditioned descriptor primitives for a Geometry-v3 successor.

This module is dataset- and annotation-agnostic.  It consumes only frozen
token maps, class-agnostic proposal geometry and a frozen image-label CAM.  It
does not load BTXRD data, segmentation targets, validation metadata or test
inputs, and it does not modify the accepted Geometry-v3 implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    proposal_context_grid_weights,
)


def cam_conditioned_extension_dim(config: MaskBagMILConfig) -> int:
    """Return the fixed CAM-core/interior/ring extension width."""

    return 3 * config.token_layers * config.token_dim


def _validate_prompt_maps(
    prompt_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
) -> None:
    if prompt_maps.ndim != 3:
        raise ValueError("prompt_maps must have shape [B,H,W]")
    if candidate_masks.ndim != 4:
        raise ValueError("candidate_masks must have shape [B,N,H,W]")
    if prompt_maps.shape != (
        candidate_masks.shape[0],
        candidate_masks.shape[2],
        candidate_masks.shape[3],
    ):
        raise ValueError("prompt_maps must align with candidate masks")
    if not torch.isfinite(prompt_maps).all():
        raise ValueError("prompt_maps must be finite")
    if bool(((prompt_maps < 0) | (prompt_maps > 1)).any()):
        raise ValueError("prompt_maps must lie in [0,1]")


def _weighted_token_mean(
    weights: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    denominator = weights.sum(dim=-1).clamp_min(1.0)
    pooled = torch.einsum("bnp,blpd->bnld", weights, tokens)
    return pooled / denominator[:, :, None, None]


def cam_conditioned_descriptor_extension(
    token_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
    prompt_maps: torch.Tensor,
    candidate_valid: torch.Tensor,
    config: MaskBagMILConfig,
    *,
    content_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool threshold-free CAM-conditioned features for frozen proposals.

    Per frozen token layer the returned descriptor concatenates the mean over
    (1) candidate times CAM, (2) candidate times one-minus-CAM, and (3) the
    accepted exterior ring times CAM.  Candidate validity/order is inherited
    exactly from Geometry-v3's shared proposal/context primitive.
    """

    if token_maps.ndim != 5:
        raise ValueError("token_maps must have shape [B,L,H,W,D]")
    batch, layers, grid_height, grid_width, channels = token_maps.shape
    if layers != config.token_layers or channels != config.token_dim:
        raise ValueError("token layout differs from the frozen configuration")
    if candidate_valid.shape != candidate_masks.shape[:2]:
        raise ValueError("candidate_valid must align with candidate masks")
    if candidate_masks.shape[0] != batch:
        raise ValueError("token and candidate batches differ")
    if not torch.isfinite(token_maps).all():
        raise ValueError("token_maps must be finite")
    _validate_prompt_maps(prompt_maps, candidate_masks)

    proposal, context, valid = proposal_context_grid_weights(
        candidate_masks,
        candidate_valid,
        grid_height=grid_height,
        grid_width=grid_width,
        minimum_grid_mass=config.minimum_grid_mass,
        context_radius=config.context_radius,
        content_masks=content_masks,
    )
    prompt = F.interpolate(
        prompt_maps[:, None].float(),
        size=(grid_height, grid_width),
        mode="area",
    )[:, 0].clamp(0.0, 1.0)

    core = proposal * prompt[:, None]
    low_cam_interior = proposal * (1.0 - prompt[:, None])
    positive_exterior = context * prompt[:, None]
    valid_weight = valid[:, :, None, None].to(proposal.dtype)
    core = core * valid_weight
    low_cam_interior = low_cam_interior * valid_weight
    positive_exterior = positive_exterior * valid_weight

    tokens = token_maps.reshape(
        batch,
        layers,
        grid_height * grid_width,
        channels,
    )
    candidates = candidate_masks.shape[1]
    pooled = [
        _weighted_token_mean(
            weights.reshape(batch, candidates, grid_height * grid_width),
            tokens,
        ).flatten(start_dim=2)
        for weights in (core, low_cam_interior, positive_exterior)
    ]
    extension = torch.cat(pooled, dim=-1)
    extension = extension * valid[:, :, None].to(extension.dtype)
    expected = (batch, candidates, cam_conditioned_extension_dim(config))
    if extension.shape != expected:
        raise RuntimeError("unexpected CAM-conditioned descriptor shape")
    if not torch.isfinite(extension).all():
        raise RuntimeError("CAM-conditioned descriptors are non-finite")
    return extension, valid


__all__ = [
    "cam_conditioned_descriptor_extension",
    "cam_conditioned_extension_dim",
]
