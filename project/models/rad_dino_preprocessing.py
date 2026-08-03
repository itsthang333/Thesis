from __future__ import annotations

"""Minimal, immutable image geometry used by the final RAD-DINO selector."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps


RAD_DINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307]).view(3, 1, 1)
RAD_DINO_STD = torch.tensor([0.2583, 0.2583, 0.2583]).view(3, 1, 1)


@dataclass(frozen=True)
class SquareProjection:
    padded_side: int
    content_box: tuple[int, int, int, int]


def pad_to_square(image: Image.Image) -> tuple[Image.Image, SquareProjection]:
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    padded = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
    return padded, SquareProjection(
        padded_side=side,
        content_box=(left, top, left + width, top + height),
    )


def raw_and_normalized_square(
    image: Image.Image,
    *,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor, SquareProjection]:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    square, projection = pad_to_square(image)
    resized = square.resize((input_size, input_size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    raw = torch.from_numpy(values).permute(2, 0, 1)
    normalized = (raw - RAD_DINO_MEAN) / RAD_DINO_STD
    return raw, normalized, projection
