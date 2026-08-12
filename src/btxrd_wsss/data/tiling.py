from __future__ import annotations

import math

import numpy as np

from btxrd_wsss.types import Tile


def _starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    values = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def tile_layout(
    height: int, width: int, tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    if height <= 0 or width <= 0 or tile_size <= 0:
        raise ValueError("Image and tile dimensions must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must lie in [0,1)")
    return [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in _starts(height, tile_size, overlap)
        for x in _starts(width, tile_size, overlap)
    ]


def extract_tiles(
    image: np.ndarray,
    *,
    image_id: str,
    tile_size: int,
    overlap: float,
    pad_mode: str = "reflect",
) -> list[Tile]:
    values = np.asarray(image)
    if values.ndim not in (2, 3):
        raise ValueError("image must be [H,W] or [C,H,W]")
    height, width = values.shape[-2:]
    result: list[Tile] = []
    for box in tile_layout(height, width, tile_size, overlap):
        x0, y0, x1, y1 = box
        patch = values[..., y0:y1, x0:x1]
        pad_h, pad_w = tile_size - patch.shape[-2], tile_size - patch.shape[-1]
        if pad_h or pad_w:
            pads = ((0, pad_h), (0, pad_w)) if patch.ndim == 2 else ((0, 0), (0, pad_h), (0, pad_w))
            mode = pad_mode if min(patch.shape[-2:]) > 1 else "edge"
            patch = np.pad(patch, pads, mode=mode)
        result.append(Tile(image_id=image_id, scale=tile_size, box=box, pixels=patch))
    return result


def cosine_window(height: int, width: int, epsilon: float = 1e-3) -> np.ndarray:
    y = np.sin(np.linspace(0, math.pi, height, dtype=np.float32))
    x = np.sin(np.linspace(0, math.pi, width, dtype=np.float32))
    return np.maximum(np.outer(y, x), epsilon)


def blend_tile_maps(
    tile_maps: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    native_shape: tuple[int, int],
) -> np.ndarray:
    if len(tile_maps) != len(boxes) or not tile_maps:
        raise ValueError("tile_maps and boxes must have the same non-zero length")
    height, width = native_shape
    channels = 1 if tile_maps[0].ndim == 2 else tile_maps[0].shape[0]
    total = np.zeros((channels, height, width), dtype=np.float32)
    weight = np.zeros((1, height, width), dtype=np.float32)
    for tile_map, (x0, y0, x1, y1) in zip(tile_maps, boxes, strict=True):
        values = np.asarray(tile_map, dtype=np.float32)
        if values.ndim == 2:
            values = values[None]
        crop_h, crop_w = y1 - y0, x1 - x0
        values = values[:, :crop_h, :crop_w]
        if values.shape != (channels, crop_h, crop_w):
            raise ValueError("Tile map shape does not match its native box")
        local_weight = cosine_window(crop_h, crop_w)[None]
        total[:, y0:y1, x0:x1] += values * local_weight
        weight[:, y0:y1, x0:x1] += local_weight
    if np.any(weight <= 0):
        raise ValueError("Tile layout does not cover the native canvas")
    blended = total / weight
    return blended[0] if channels == 1 else blended
