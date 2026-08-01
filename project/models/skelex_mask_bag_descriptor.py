from __future__ import annotations

"""Frozen SKELEX token extraction and exact-support proposal pooling.

The primitives are annotation-agnostic. They accept only full-image tensors,
class-agnostic proposal masks, and non-semantic proposal metadata.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


SKELEX_GRID_SIZE = 14
SKELEX_HIDDEN_SIZE = 1024
SKELEX_PATCHES = SKELEX_GRID_SIZE * SKELEX_GRID_SIZE
SELECTED_HIDDEN_LAYERS = (8, 16, 24)


@dataclass(frozen=True)
class SkelexDescriptorConfig:
    token_dim: int = 128
    token_layers: int = len(SELECTED_HIDDEN_LAYERS)
    metadata_dim: int = 4
    context_radius: int = 2
    support_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.token_dim <= 0 or self.token_layers <= 0 or self.metadata_dim <= 0:
            raise ValueError("SKELEX descriptor dimensions must be positive")
        if self.context_radius < 1 or self.support_epsilon <= 0:
            raise ValueError("SKELEX pooling controls are invalid")

    @property
    def descriptor_dim(self) -> int:
        return 3 * self.token_layers * self.token_dim + self.metadata_dim


class SkelexProjectedMultiLayerEncoder(nn.Module):
    """Expose fixed, unmasked ViT-MAE patch grids in their original order."""

    def __init__(self, encoder: nn.Module, projection: torch.Tensor) -> None:
        super().__init__()
        if projection.shape[0] != SKELEX_HIDDEN_SIZE:
            raise ValueError("SKELEX projection input dimension must be 1024")
        self.encoder = encoder
        self.register_buffer("projection", projection.float(), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        # ViTMAE sorts this noise before masking. With mask_ratio=0 and a
        # strictly increasing vector, no patch is removed or permuted.
        noise = torch.arange(
            SKELEX_PATCHES,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )[None].expand(batch, -1)
        output = self.encoder(
            pixel_values=pixel_values,
            noise=noise,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != 25:
            raise RuntimeError("SKELEX must expose embedding plus 24 hidden states")
        selected: list[torch.Tensor] = []
        for layer_index in SELECTED_HIDDEN_LAYERS:
            hidden = hidden_states[layer_index]
            if hidden.shape != (batch, SKELEX_PATCHES + 1, SKELEX_HIDDEN_SIZE):
                raise RuntimeError(
                    f"Unexpected SKELEX layer-{layer_index} shape {tuple(hidden.shape)}"
                )
            # Hugging Face ViTMAE appends one CLS token after patch masking.
            # Only the following 196 spatial tokens may be reshaped as a grid.
            patches = hidden[:, 1:].reshape(
                batch,
                SKELEX_GRID_SIZE,
                SKELEX_GRID_SIZE,
                SKELEX_HIDDEN_SIZE,
            ).float()
            selected.append(F.normalize(patches @ self.projection, dim=-1))
        return torch.stack(selected, dim=1)


def exact_fractional_mask_pool_descriptors(
    token_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_metadata: torch.Tensor,
    candidate_valid: torch.Tensor,
    config: SkelexDescriptorConfig,
    *,
    content_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool exact immutable candidates without a coarse-grid mass cutoff."""

    if token_maps.ndim != 5:
        raise ValueError("token_maps must have shape [B,L,H,W,D]")
    if candidate_masks.ndim != 4:
        raise ValueError("candidate_masks must have shape [B,N,H,W]")
    if candidate_valid.shape != candidate_masks.shape[:2]:
        raise ValueError("candidate_valid must align with candidate_masks")
    batch, layers, grid_h, grid_w, channels = token_maps.shape
    candidates = candidate_masks.shape[1]
    if (layers, channels) != (config.token_layers, config.token_dim):
        raise ValueError("token layout differs from frozen SKELEX descriptor config")
    if candidate_metadata.shape != (batch, candidates, config.metadata_dim):
        raise ValueError("candidate metadata shape mismatch")
    for name, values in (
        ("token_maps", token_maps),
        ("candidate_masks", candidate_masks),
        ("candidate_metadata", candidate_metadata),
    ):
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite")

    masks = F.interpolate(
        candidate_masks.float().reshape(
            batch * candidates, 1, candidate_masks.shape[-2], candidate_masks.shape[-1]
        ),
        size=(grid_h, grid_w),
        mode="area",
    ).reshape(batch, candidates, grid_h, grid_w).clamp_(0.0, 1.0)
    if content_masks is None:
        content = torch.ones(
            (batch, grid_h, grid_w), dtype=masks.dtype, device=masks.device
        )
    else:
        if content_masks.shape != (
            batch,
            candidate_masks.shape[-2],
            candidate_masks.shape[-1],
        ):
            raise ValueError("content_masks must align with candidate_masks")
        if not torch.isfinite(content_masks).all():
            raise ValueError("content_masks must be finite")
        content = F.interpolate(
            content_masks[:, None].float(),
            size=(grid_h, grid_w),
            mode="area",
        )[:, 0].clamp_(0.0, 1.0)
    masks = masks * content[:, None]
    support_mass = masks.sum(dim=(-2, -1))
    valid = candidate_valid.bool()
    if (support_mass[valid] <= config.support_epsilon).any():
        raise RuntimeError(
            "An immutable candidate has zero SKELEX-grid support; silent dropping is forbidden"
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
    proposal_weights = masks.reshape(batch, candidates, grid_h * grid_w)
    context_weights = context.reshape(batch, candidates, grid_h * grid_w)
    inside = torch.einsum("bnp,blpd->bnld", proposal_weights, tokens)
    inside = inside / proposal_weights.sum(dim=-1).clamp_min(
        config.support_epsilon
    )[:, :, None, None]
    outside = torch.einsum("bnp,blpd->bnld", context_weights, tokens)
    outside = outside / context_weights.sum(dim=-1).clamp_min(
        config.support_epsilon
    )[:, :, None, None]
    descriptors = torch.cat(
        (
            inside.flatten(start_dim=2),
            outside.flatten(start_dim=2),
            (inside - outside).flatten(start_dim=2),
            candidate_metadata.float(),
        ),
        dim=-1,
    )
    descriptors = descriptors * valid[:, :, None].to(descriptors.dtype)
    if descriptors.shape != (batch, candidates, config.descriptor_dim):
        raise RuntimeError("Unexpected SKELEX descriptor shape")
    if not torch.isfinite(descriptors).all():
        raise RuntimeError("SKELEX descriptors are non-finite")
    return descriptors, valid, support_mass


__all__ = [
    "SELECTED_HIDDEN_LAYERS",
    "SKELEX_GRID_SIZE",
    "SKELEX_HIDDEN_SIZE",
    "SKELEX_PATCHES",
    "SkelexDescriptorConfig",
    "SkelexProjectedMultiLayerEncoder",
    "exact_fractional_mask_pool_descriptors",
]
