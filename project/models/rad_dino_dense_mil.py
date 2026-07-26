from __future__ import annotations

"""Small image-label-only dense MIL head for frozen RAD-DINO patch tokens.

The encoder is deliberately kept outside this module.  The head receives
``[B, H, W, D]`` patch tokens and emits a dense logit map; image-level
supervision is applied only after deterministic Log-Sum-Exp pooling.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DenseMILConfig:
    input_dim: int = 768
    temperature: float = 0.20
    consistency_weight: float = 0.10


class DenseMILHead(nn.Module):
    """Layer-normalized linear patch scorer with no image-specific parameters."""

    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B, 1, H, W]``."""
        values = torch.as_tensor(patch_tokens)
        if values.ndim == 4:
            if values.shape[-1] != self.norm.normalized_shape[0]:
                raise ValueError("Patch-token embedding dimension does not match head")
            values = self.projection(self.norm(values)).squeeze(-1).unsqueeze(1)
            return values
        if values.ndim == 3:
            if values.shape[-1] != self.norm.normalized_shape[0]:
                raise ValueError("Patch-token embedding dimension does not match head")
            logits = self.projection(self.norm(values)).squeeze(-1)
            return logits
        raise ValueError("patch_tokens must have shape [B,H,W,D] or [B,N,D]")


def logsumexp_pool(
    logits: torch.Tensor,
    *,
    temperature: float = DenseMILConfig.temperature,
) -> torch.Tensor:
    """Smooth-max pooling that preserves signal from a small positive region."""
    values = torch.as_tensor(logits)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError("logits must have shape [B,H,W] or [B,1,H,W]")
    if temperature <= 0 or not torch.isfinite(values).all():
        raise ValueError("temperature must be positive and logits finite")
    flat = values.flatten(1)
    scaled = flat / float(temperature)
    return float(temperature) * torch.logsumexp(scaled, dim=1) - float(
        temperature
    ) * torch.log(torch.tensor(float(flat.shape[1]), device=values.device))


def dense_mil_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = DenseMILConfig.temperature,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary image-label loss and pooled image logits."""
    values = torch.as_tensor(logits)
    targets = torch.as_tensor(labels, device=values.device, dtype=values.dtype).view(-1)
    pooled = logsumexp_pool(values, temperature=temperature)
    if pooled.shape != targets.shape:
        raise ValueError("labels must contain one binary target per image")
    loss = F.binary_cross_entropy_with_logits(pooled, targets)
    return loss, pooled


def resize_probability_map(
    logits: torch.Tensor,
    *,
    output_size: int,
) -> torch.Tensor:
    """Convert patch logits to a bounded square probability map."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    values = torch.as_tensor(logits)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,H,W] or [B,H,W]")
    return torch.sigmoid(
        F.interpolate(values, size=(output_size, output_size), mode="bilinear", align_corners=False)
    )


def merge_full_and_tiles(
    full_map: torch.Tensor,
    tile_maps: torch.Tensor,
    *,
    tile_boxes: tuple[tuple[int, int, int, int], ...],
    image_size: int,
    full_weight: float = 0.5,
) -> torch.Tensor:
    """Merge fixed overlapping tile maps without image-specific normalization."""
    full = torch.as_tensor(full_map)
    tiles = torch.as_tensor(tile_maps)
    if full.ndim == 3:
        full = full.unsqueeze(1)
    if tiles.ndim == 3:
        tiles = tiles.unsqueeze(1)
    if full.ndim != 4 or tiles.ndim != 4 or full.shape[1] != 1:
        raise ValueError("full_map/tile_maps must have one channel")
    if tiles.shape[1] != 1 or tiles.shape[0] != len(tile_boxes):
        raise ValueError("tile map count differs from tile layout")
    if not 0.0 <= full_weight <= 1.0:
        raise ValueError("full_weight must lie in [0,1]")
    canvas = torch.zeros(
        (full.shape[0], 1, image_size, image_size),
        dtype=full.dtype,
        device=full.device,
    )
    coverage = torch.zeros_like(canvas)
    for index, (x0, y0, x1, y1) in enumerate(tile_boxes):
        if not (0 <= x0 < x1 <= image_size and 0 <= y0 < y1 <= image_size):
            raise ValueError("tile box lies outside image canvas")
        tile = tiles[index]
        if tile.shape[-2:] != (y1 - y0, x1 - x0):
            raise ValueError("tile map shape differs from tile box")
        canvas[:, :, y0:y1, x0:x1] += tile
        coverage[:, :, y0:y1, x0:x1] += 1.0
    if torch.any(coverage <= 0):
        raise ValueError("tile layout does not cover the full image")
    merged = canvas / coverage
    return float(full_weight) * full + (1.0 - float(full_weight)) * merged
