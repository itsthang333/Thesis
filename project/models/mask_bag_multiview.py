from __future__ import annotations

"""GT-free fixed-tile geometry for higher-resolution proposal descriptors."""

import numpy as np


def overlapping_corner_tiles(
    *,
    image_height: int,
    image_width: int,
    crop_fraction: float,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return top-left/right and bottom-left/right overlapping rectangles."""

    if image_height <= 1 or image_width <= 1:
        raise ValueError("image dimensions must exceed one pixel")
    if not 0.5 < crop_fraction < 1.0:
        raise ValueError("crop_fraction must lie strictly between 0.5 and 1")
    tile_height = min(
        image_height,
        max(1, int(np.ceil(image_height * crop_fraction))),
    )
    tile_width = min(
        image_width,
        max(1, int(np.ceil(image_width * crop_fraction))),
    )
    bottom = image_height - tile_height
    right = image_width - tile_width
    return (
        (0, 0, tile_width, tile_height),
        (right, 0, image_width, tile_height),
        (0, bottom, tile_width, image_height),
        (right, bottom, image_width, image_height),
    )


def candidate_tile_mass_retention(
    candidate_masks: np.ndarray,
    tile_boxes: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    """Measure the fraction of every candidate retained by every tile."""

    masks = np.asarray(candidate_masks, dtype=np.float32)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    if not np.isfinite(masks).all() or np.any(masks < 0):
        raise ValueError("candidate masks must be finite and nonnegative")
    total_mass = masks.sum(axis=(1, 2))
    if np.any(total_mass <= 0):
        raise ValueError("candidate masks must have positive mass")
    if len(tile_boxes) == 0:
        raise ValueError("tile_boxes must be nonempty")

    height, width = masks.shape[1:]
    retained: list[np.ndarray] = []
    for x0, y0, x1, y1 in tile_boxes:
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("tile lies outside the candidate-mask frame")
        retained.append(masks[:, y0:y1, x0:x1].sum(axis=(1, 2)) / total_mass)
    result = np.stack(retained, axis=1).astype(np.float32)
    if np.any(result < 0) or np.any(result > 1.0 + 1.0e-6):
        raise RuntimeError("tile mass retention lies outside [0,1]")
    return np.clip(result, 0.0, 1.0)


def maximum_retention_tile_weights(
    retention: np.ndarray,
    *,
    absolute_tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Average all equally best tiles to preserve transformation equivariance."""

    values = np.asarray(retention, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("retention must have shape [N,T]")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("retention must be finite and lie in [0,1]")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be nonnegative")
    maximum = values.max(axis=1, keepdims=True)
    tied = np.abs(values - maximum) <= absolute_tolerance
    weights = tied.astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def horizontal_flip_tile_mapping(
    tile_boxes: tuple[tuple[int, int, int, int], ...],
    *,
    image_width: int,
) -> np.ndarray:
    """Map each tile index to the exact tile covering its horizontal mirror."""

    if image_width <= 1 or len(tile_boxes) == 0:
        raise ValueError("tile layout and image_width must be valid")
    index_by_box = {tuple(box): index for index, box in enumerate(tile_boxes)}
    if len(index_by_box) != len(tile_boxes):
        raise ValueError("tile boxes must be unique")
    mapping: list[int] = []
    for x0, y0, x1, y1 in tile_boxes:
        mirrored = (image_width - x1, y0, image_width - x0, y1)
        if mirrored not in index_by_box:
            raise ValueError("tile layout is not horizontally flip-closed")
        mapping.append(index_by_box[mirrored])
    result = np.asarray(mapping, dtype=np.int32)
    if not np.array_equal(result[result], np.arange(len(tile_boxes))):
        raise RuntimeError("horizontal flip tile mapping is not an involution")
    return result


def weighted_local_descriptors(
    per_tile_descriptors: np.ndarray,
    tile_weights: np.ndarray,
) -> np.ndarray:
    """Fuse aligned per-tile candidate descriptors using GT-free weights."""

    descriptors = np.asarray(per_tile_descriptors, dtype=np.float32)
    weights = np.asarray(tile_weights, dtype=np.float32)
    if descriptors.ndim != 3:
        raise ValueError("per_tile_descriptors must have shape [N,T,D]")
    if weights.shape != descriptors.shape[:2]:
        raise ValueError("tile_weights must align with candidates and tiles")
    if not np.isfinite(descriptors).all() or not np.isfinite(weights).all():
        raise ValueError("descriptors and tile weights must be finite")
    if np.any(weights < 0) or not np.allclose(
        weights.sum(axis=1),
        1.0,
        atol=1.0e-6,
    ):
        raise ValueError("tile weights must be nonnegative and sum to one")
    return np.einsum("ntd,nt->nd", descriptors, weights).astype(np.float32)


__all__ = [
    "candidate_tile_mass_retention",
    "horizontal_flip_tile_mapping",
    "maximum_retention_tile_weights",
    "overlapping_corner_tiles",
    "weighted_local_descriptors",
]
