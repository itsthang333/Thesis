from __future__ import annotations

import random

import numpy as np
from PIL import Image

from btxrd_wsss.data.tiling import extract_tiles
from btxrd_wsss.types import Tile


def resize_long_side(image: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[-2:]
    scale = long_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = np.asarray(Image.fromarray(image).resize(target, Image.Resampling.BICUBIC))
    return resized, scale


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC))


def pad_to_multiple(image: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    if multiple < 1:
        raise ValueError("multiple must be positive")
    height, width = image.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if not (pad_h or pad_w):
        return image, (height, width)
    pads = ((0, pad_h), (0, pad_w)) if image.ndim == 2 else ((0, 0), (0, pad_h), (0, pad_w))
    mode = "reflect" if min(height, width) > 1 else "edge"
    return np.pad(image, pads, mode=mode), (height, width)


def sample_native_tiles(
    image: np.ndarray,
    image_id: str,
    sizes: list[int],
    overlap: float,
    count: int,
    rng: random.Random,
) -> list[Tile]:
    all_tiles = [
        tile
        for size in sizes
        for tile in extract_tiles(image, image_id=image_id, tile_size=size, overlap=overlap)
    ]
    if len(all_tiles) <= count:
        return all_tiles
    # Bias sampling toward textured regions without removing random exploration.
    contrast = np.asarray([float(np.std(tile.pixels)) for tile in all_tiles])
    contrast = contrast - contrast.min() + 1e-6
    contrast = contrast / contrast.sum()
    weights = 0.7 * contrast + 0.3 / len(all_tiles)
    chosen: list[Tile] = []
    available = list(range(len(all_tiles)))
    for _ in range(count):
        position = rng.choices(available, weights=[weights[i] for i in available], k=1)[0]
        chosen.append(all_tiles[position])
        available.remove(position)
    return chosen
