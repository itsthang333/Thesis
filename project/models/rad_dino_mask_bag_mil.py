from __future__ import annotations

"""Mask-proposal MIL primitives for image-label-only BTXRD localization.

The module is deliberately dataset- and annotation-agnostic. Candidate masks
are treated as class-agnostic spatial proposals; frozen RAD-DINO token maps and
image-level labels are the only learned semantic signal.  Validation masks are
never accepted by any API in this file.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MaskBagMILConfig:
    token_dim: int = 128
    token_layers: int = 3
    hidden_dim: int = 256
    metadata_dim: int = 4
    bag_temperature: float = 0.20
    context_radius: int = 2
    minimum_grid_mass: float = 0.25

    def __post_init__(self) -> None:
        for name in ("token_dim", "token_layers", "metadata_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least two")
        if self.bag_temperature <= 0:
            raise ValueError("bag_temperature must be positive")
        if self.context_radius < 1:
            raise ValueError("context_radius must be at least one")
        if self.minimum_grid_mass <= 0:
            raise ValueError("minimum_grid_mass must be positive")

    @property
    def descriptor_dim(self) -> int:
        # Per layer: proposal mean, local-context mean and their difference.
        return 3 * self.token_layers * self.token_dim + self.metadata_dim


def project_direct_resize_masks_to_square(
    candidate_masks: torch.Tensor,
    *,
    padded_side: int,
    content_box: tuple[int, int, int, int],
    output_size: int,
) -> torch.Tensor:
    """Map direct-resize proposal masks into a square-padded encoder frame.

    Candidate-gallery masks use normalized coordinates of the unpadded source
    radiograph, whereas RAD-DINO tokens use a centered square-padded source.
    This function performs the exact continuous coordinate change before mask
    pooling. Bilinear sampling preserves fractional support at the content
    boundary; pixels outside the source-image content box are zero.
    """

    if candidate_masks.ndim != 3:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    if not torch.isfinite(candidate_masks).all():
        raise ValueError("candidate_masks must be finite")
    if int(padded_side) <= 0 or int(output_size) <= 0:
        raise ValueError("padded_side and output_size must be positive")
    x0, y0, x1, y1 = (int(value) for value in content_box)
    if not (0 <= x0 < x1 <= int(padded_side)):
        raise ValueError("content_box has invalid horizontal bounds")
    if not (0 <= y0 < y1 <= int(padded_side)):
        raise ValueError("content_box has invalid vertical bounds")

    device = candidate_masks.device
    coordinates = (
        torch.arange(output_size, dtype=torch.float32, device=device) + 0.5
    ) * (float(padded_side) / float(output_size))
    source_x = (coordinates - float(x0)) / float(x1 - x0)
    source_y = (coordinates - float(y0)) / float(y1 - y0)
    grid_y, grid_x = torch.meshgrid(source_y, source_x, indexing="ij")
    grid = torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1)
    grid = grid[None].expand(candidate_masks.shape[0], -1, -1, -1)
    projected = F.grid_sample(
        candidate_masks[:, None].float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]
    return projected.clamp_(0.0, 1.0)


def _validate_inputs(
    token_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_metadata: torch.Tensor,
    candidate_valid: torch.Tensor,
    config: MaskBagMILConfig,
    content_masks: torch.Tensor | None,
) -> None:
    if token_maps.ndim != 5:
        raise ValueError("token_maps must have shape [B,L,H,W,D]")
    if candidate_masks.ndim != 4:
        raise ValueError("candidate_masks must have shape [B,N,H,W]")
    if candidate_metadata.ndim != 3:
        raise ValueError("candidate_metadata must have shape [B,N,K]")
    if candidate_valid.ndim != 2:
        raise ValueError("candidate_valid must have shape [B,N]")
    batch, layers, _height, _width, channels = token_maps.shape
    mask_batch, candidates, _mask_height, _mask_width = candidate_masks.shape
    if batch != mask_batch:
        raise ValueError("Token and candidate-mask batch sizes differ")
    if layers != config.token_layers or channels != config.token_dim:
        raise ValueError("Token layout differs from the frozen configuration")
    if candidate_metadata.shape != (batch, candidates, config.metadata_dim):
        raise ValueError("Candidate metadata shape differs from the configuration")
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("Candidate-valid shape differs from candidate masks")
    if content_masks is not None:
        if content_masks.ndim != 3 or content_masks.shape != (
            batch,
            candidate_masks.shape[-2],
            candidate_masks.shape[-1],
        ):
            raise ValueError(
                "content_masks must have shape [B,H,W] matching candidate masks"
            )
        if not torch.isfinite(content_masks).all():
            raise ValueError("content_masks must be finite")
        if not (content_masks > 0).flatten(start_dim=1).any(dim=1).all():
            raise ValueError("Every content mask must contain valid image support")
    if not torch.isfinite(token_maps).all():
        raise ValueError("token_maps must be finite")
    if not torch.isfinite(candidate_masks).all():
        raise ValueError("candidate_masks must be finite")
    if not torch.isfinite(candidate_metadata).all():
        raise ValueError("candidate_metadata must be finite")


def mask_pool_descriptors(
    token_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_metadata: torch.Tensor,
    candidate_valid: torch.Tensor,
    config: MaskBagMILConfig,
    *,
    content_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool proposal, local-context and contrast features from token grids.

    Area interpolation preserves the fractional coverage of small masks when
    mapping them to the token grid. Invalid/padded proposals are zeroed and do
    not contribute to a bag loss.
    """

    _validate_inputs(
        token_maps,
        candidate_masks,
        candidate_metadata,
        candidate_valid,
        config,
        content_masks,
    )
    batch, layers, grid_h, grid_w, channels = token_maps.shape
    candidates = candidate_masks.shape[1]
    masks = F.interpolate(
        candidate_masks.float().reshape(
            batch * candidates,
            1,
            candidate_masks.shape[-2],
            candidate_masks.shape[-1],
        ),
        size=(grid_h, grid_w),
        mode="area",
    ).reshape(batch, candidates, grid_h, grid_w)
    if content_masks is None:
        content = torch.ones(
            (batch, grid_h, grid_w),
            dtype=masks.dtype,
            device=masks.device,
        )
    else:
        content = F.interpolate(
            content_masks[:, None].float(),
            size=(grid_h, grid_w),
            mode="area",
        )[:, 0].clamp(0.0, 1.0)
    masks = masks * content[:, None]
    valid = candidate_valid.bool() & (
        masks.sum(dim=(-2, -1)) >= config.minimum_grid_mass
    )
    masks = masks * valid[:, :, None, None].to(masks.dtype)

    kernel = 2 * config.context_radius + 1
    dilated = F.max_pool2d(
        masks.reshape(batch * candidates, 1, grid_h, grid_w),
        kernel_size=kernel,
        stride=1,
        padding=config.context_radius,
    ).reshape(batch, candidates, grid_h, grid_w)
    context = (dilated - masks).clamp_min(0.0) * content[:, None]

    tokens = token_maps.reshape(batch, layers, grid_h * grid_w, channels)
    mask_weights = masks.reshape(batch, candidates, grid_h * grid_w)
    context_weights = context.reshape(batch, candidates, grid_h * grid_w)
    mask_denominator = mask_weights.sum(dim=-1).clamp_min(1.0)
    context_denominator = context_weights.sum(dim=-1).clamp_min(1.0)
    inside = torch.einsum("bnp,blpd->bnld", mask_weights, tokens)
    inside = inside / mask_denominator[:, :, None, None]
    outside = torch.einsum("bnp,blpd->bnld", context_weights, tokens)
    outside = outside / context_denominator[:, :, None, None]
    contrast = inside - outside

    descriptors = torch.cat(
        [
            inside.flatten(start_dim=2),
            outside.flatten(start_dim=2),
            contrast.flatten(start_dim=2),
            candidate_metadata.float(),
        ],
        dim=-1,
    )
    descriptors = descriptors * valid[:, :, None].to(descriptors.dtype)
    if descriptors.shape != (batch, candidates, config.descriptor_dim):
        raise RuntimeError("Unexpected mask-pooled descriptor shape")
    return descriptors, valid


def smooth_mil_pool(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Normalized LogSumExp MIL pooling with fail-closed empty bags."""

    if candidate_logits.ndim != 2 or candidate_valid.shape != candidate_logits.shape:
        raise ValueError("Candidate logits and validity must share shape [B,N]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every image bag must contain at least one valid proposal")
    scaled = candidate_logits / temperature
    masked = scaled.masked_fill(~valid, -torch.inf)
    counts = valid.sum(dim=1).to(candidate_logits.dtype)
    return temperature * (torch.logsumexp(masked, dim=1) - counts.log())


def image_bag_loss(bag_logits: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
    labels = image_labels.to(dtype=bag_logits.dtype).reshape_as(bag_logits)
    return F.binary_cross_entropy_with_logits(bag_logits, labels)


def self_guided_instance_loss(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    image_labels: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Self-guided MIL loss with uncertain positive instances left unlabeled.

    All proposals from image-level-negative bags are reliable negatives. For a
    positive bag only its current highest-scoring proposal receives a detached
    positive target; the remaining proposals are intentionally ignored rather
    than being mislabeled as background.
    """

    if candidate_logits.ndim != 2 or candidate_valid.shape != candidate_logits.shape:
        raise ValueError("Candidate logits and validity must share shape [B,N]")
    if negative_weight < 0 or positive_weight < 0:
        raise ValueError("Instance-loss weights must be nonnegative")
    labels = image_labels.reshape(-1).bool()
    if labels.numel() != candidate_logits.shape[0]:
        raise ValueError("Image-label batch size differs from candidate bags")
    valid = candidate_valid.bool()
    terms: list[torch.Tensor] = []
    if (~labels).any():
        negative_logits = candidate_logits[~labels][valid[~labels]]
        if negative_logits.numel():
            targets = torch.zeros_like(negative_logits)
            terms.append(
                negative_weight
                * F.binary_cross_entropy_with_logits(negative_logits, targets)
            )
    positive_rows = torch.nonzero(labels, as_tuple=False).reshape(-1)
    for row in positive_rows.tolist():
        if not valid[row].any():
            raise ValueError("Positive bag has no valid proposal")
        scores = candidate_logits[row].masked_fill(~valid[row], -torch.inf)
        winner = int(scores.detach().argmax().item())
        terms.append(
            positive_weight
            * F.binary_cross_entropy_with_logits(
                candidate_logits[row, winner],
                torch.ones_like(candidate_logits[row, winner]),
            )
        )
    if not terms:
        return candidate_logits.sum() * 0.0
    return torch.stack(terms).mean()


def aligned_candidate_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    if logits_a.shape != logits_b.shape or logits_a.shape != candidate_valid.shape:
        raise ValueError("Aligned candidate logits/validity must share shape")
    valid = candidate_valid.bool()
    if not valid.any():
        return (logits_a.sum() + logits_b.sum()) * 0.0
    return F.smooth_l1_loss(
        torch.sigmoid(logits_a[valid]),
        torch.sigmoid(logits_b[valid]),
    )


def winner_take_all_map(
    candidate_logits: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_valid: torch.Tensor,
    bag_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the highest-scoring class-agnostic proposal and its index."""

    if candidate_logits.ndim != 2 or candidate_valid.shape != candidate_logits.shape:
        raise ValueError("Candidate logits and validity must share shape [B,N]")
    if candidate_masks.ndim != 4 or candidate_masks.shape[:2] != candidate_logits.shape:
        raise ValueError("candidate_masks must have shape [B,N,H,W]")
    if bag_logits.shape != (candidate_logits.shape[0],):
        raise ValueError("bag_logits must have shape [B]")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every image bag must contain at least one valid proposal")
    winners = candidate_logits.masked_fill(~valid, -torch.inf).argmax(dim=1)
    batch_indices = torch.arange(candidate_logits.shape[0], device=candidate_logits.device)
    masks = candidate_masks.float()[batch_indices, winners]
    maps = masks * torch.sigmoid(bag_logits)[:, None, None]
    return maps.clamp(0.0, 1.0), winners


class RadDinoMaskBagMIL(nn.Module):
    def __init__(self, config: MaskBagMILConfig) -> None:
        super().__init__()
        self.config = config
        self.scorer = nn.Sequential(
            nn.LayerNorm(config.descriptor_dim),
            nn.Linear(config.descriptor_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def score_descriptors(
        self,
        descriptors: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.config.descriptor_dim:
            raise ValueError("descriptors must have shape [B,N,descriptor_dim]")
        if candidate_valid.shape != descriptors.shape[:2]:
            raise ValueError("candidate_valid must align with descriptors")
        valid = candidate_valid.bool()
        candidate_logits = self.scorer(descriptors).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~valid, 0.0)
        bag_logits = smooth_mil_pool(
            candidate_logits,
            valid,
            temperature=self.config.bag_temperature,
        )
        return candidate_logits, bag_logits

    def forward(
        self,
        token_maps: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_metadata: torch.Tensor,
        candidate_valid: torch.Tensor,
        content_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptors, valid = mask_pool_descriptors(
            token_maps,
            candidate_masks,
            candidate_metadata,
            candidate_valid,
            self.config,
            content_masks=content_masks,
        )
        candidate_logits, bag_logits = self.score_descriptors(descriptors, valid)
        return candidate_logits, bag_logits, valid


__all__ = [
    "MaskBagMILConfig",
    "RadDinoMaskBagMIL",
    "aligned_candidate_consistency_loss",
    "image_bag_loss",
    "mask_pool_descriptors",
    "project_direct_resize_masks_to_square",
    "self_guided_instance_loss",
    "smooth_mil_pool",
    "winner_take_all_map",
]
