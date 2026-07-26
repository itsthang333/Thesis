from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


@dataclass(frozen=True)
class SquareProjection:
    padded_side: int
    content_box: tuple[int, int, int, int]


def pad_to_square(
    image: Image.Image,
    *,
    fill: int | tuple[int, int, int] = 0,
) -> tuple[Image.Image, SquareProjection]:
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    padded = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)
    return padded, SquareProjection(
        padded_side=side,
        content_box=(left, top, left + width, top + height),
    )


def make_noise_bank(
    *,
    num_masks: int,
    num_patches: int,
    seed: int,
) -> torch.Tensor:
    """Build the frozen random-ranking tensors consumed by ViT-MAE masking."""
    if num_masks <= 0 or num_patches <= 0:
        raise ValueError("num_masks and num_patches must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.rand((num_masks, num_patches), generator=generator)
    # Ties would make the argsort contract dependent on implementation details.
    if any(torch.unique(row).numel() != num_patches for row in noise):
        raise RuntimeError("Frozen MAE noise bank contains a tie")
    return noise


def noise_bank_sha256(noise: torch.Tensor) -> str:
    if noise.ndim != 2 or not torch.isfinite(noise).all():
        raise ValueError("Noise bank must be finite and two-dimensional")
    payload = {
        "dtype": "float32",
        "shape": list(noise.shape),
        "bytes_sha256": hashlib.sha256(
            noise.detach().cpu().contiguous().float().numpy().tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def patchify(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must have shape [batch, channels, height, width]")
    batch, channels, height, width = pixel_values.shape
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = pixel_values.reshape(
        batch,
        channels,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
    return patches.reshape(batch, grid_h * grid_w, patch_size * patch_size * channels)


def unpatchify(
    patches: torch.Tensor,
    *,
    patch_size: int,
    channels: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, patches, values]")
    batch, count, values = patches.shape
    if count != grid_height * grid_width:
        raise ValueError("Patch count does not match the requested grid")
    if values != patch_size * patch_size * channels:
        raise ValueError("Patch vector length does not match patch_size/channels")
    image = patches.reshape(
        batch,
        grid_height,
        grid_width,
        patch_size,
        patch_size,
        channels,
    )
    image = image.permute(0, 5, 1, 3, 2, 4).contiguous()
    return image.reshape(
        batch,
        channels,
        grid_height * patch_size,
        grid_width * patch_size,
    )


def expand_patch_mask(
    mask: torch.Tensor,
    *,
    patch_size: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[1] != grid_height * grid_width:
        raise ValueError("MAE mask shape does not match the requested grid")
    expanded = mask.reshape(mask.shape[0], 1, grid_height, grid_width)
    return expanded.repeat_interleave(patch_size, 2).repeat_interleave(patch_size, 3)


def accumulate_masked_squared_error(
    *,
    prediction_patches: torch.Tensor,
    pixel_values: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pixel masked squared-error sum and observation coverage."""
    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must be four-dimensional")
    batch, channels, height, width = pixel_values.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    reconstruction = unpatchify(
        prediction_patches,
        patch_size=patch_size,
        channels=channels,
        grid_height=grid_h,
        grid_width=grid_w,
    )
    if reconstruction.shape != pixel_values.shape:
        raise ValueError("Reconstruction shape differs from input")
    pixel_mask = expand_patch_mask(
        patch_mask,
        patch_size=patch_size,
        grid_height=grid_h,
        grid_width=grid_w,
    ).to(dtype=pixel_values.dtype)
    squared_error = (reconstruction.float() - pixel_values.float()).square().mean(dim=1)
    return squared_error * pixel_mask[:, 0], pixel_mask[:, 0]


def project_square_map(
    values: np.ndarray,
    projection: SquareProjection,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Square map must be finite and two-dimensional")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output dimensions must be positive")
    square = torch.from_numpy(values.astype(np.float32, copy=False))[None, None]
    square = F.interpolate(
        square,
        size=(projection.padded_side, projection.padded_side),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    x0, y0, x1, y1 = projection.content_box
    if not (
        0 <= x0 < x1 <= projection.padded_side
        and 0 <= y0 < y1 <= projection.padded_side
    ):
        raise ValueError("Content box lies outside the padded square")
    content = square[y0:y1, x0:x1][None, None]
    resized = F.interpolate(
        content,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy().astype(np.float32, copy=False)


def radiograph_foreground_mask(
    image: Image.Image,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    gray = image.convert("L").resize((output_width, output_height), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.float32)
    threshold = max(8.0, float(np.percentile(values, 99.0)) * 0.08)
    foreground = values > threshold
    if not foreground.any():
        return np.ones((output_height, output_width), dtype=bool)
    return foreground


def robust_foreground_normalize(
    values: np.ndarray,
    foreground: np.ndarray,
    *,
    low_percentile: float = 5.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    if values.ndim != 2 or foreground.shape != values.shape:
        raise ValueError("Map and foreground mask must be aligned 2D arrays")
    if not np.isfinite(values).all():
        raise ValueError("Error map contains non-finite values")
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Invalid normalization percentiles")
    valid = values[foreground]
    if valid.size == 0:
        raise ValueError("Foreground mask is empty")
    low, high = np.percentile(valid, [low_percentile, high_percentile])
    result = np.zeros_like(values, dtype=np.float32)
    if float(high) <= float(low) + 1e-12:
        return result
    result[foreground] = np.clip(
        (values[foreground] - float(low)) / (float(high) - float(low)),
        0.0,
        1.0,
    )
    return result


def validate_complete_mask_coverage(
    masks: Sequence[torch.Tensor],
    *,
    num_patches: int,
) -> torch.Tensor:
    if not masks:
        raise ValueError("At least one mask is required")
    coverage = torch.zeros(num_patches, dtype=torch.int64)
    for mask in masks:
        if mask.ndim != 1 or mask.numel() != num_patches:
            raise ValueError("Mask does not match num_patches")
        coverage += (mask.detach().cpu() > 0.5).to(torch.int64)
    if int(coverage.min()) <= 0:
        raise ValueError("Frozen mask bank leaves at least one patch never reconstructed")
    return coverage
