from __future__ import annotations

"""Prediction-only geodesic refinement for weak localization maps.

The module intentionally contains no dataset or annotation access.  It turns a
frozen continuous localization map into foreground/background seeds, computes
multi-source shortest-path distances on an image/feature graph, and fuses the
result with the source map.  Ambiguous pixels remain continuous; thresholding
and consumer training are separate protocol decisions.
"""

from dataclasses import dataclass
import heapq
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class GeodesicSeedExpansion:
    probability: np.ndarray
    foreground_seeds: np.ndarray
    background_seeds: np.ndarray
    foreground_distance: np.ndarray
    background_distance: np.ndarray
    diagnostics: dict[str, float | int]


def _as_probability(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError(f"{name} must be bounded in [0, 1]")
    return array


def exact_rank_seed_masks(
    probability: np.ndarray,
    valid_mask: np.ndarray,
    *,
    foreground_fraction: float = 0.01,
    background_fraction: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine deterministic, disjoint foreground/background rank seeds.

    Stable row-major ordering resolves equal-valued pixels.  Fractions are
    defined over the valid radiograph content only.
    """

    values = _as_probability(probability, "probability")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError("valid_mask and probability shapes differ")
    if not 0.0 < float(foreground_fraction) < 1.0:
        raise ValueError("foreground_fraction must lie in (0, 1)")
    if not 0.0 < float(background_fraction) < 1.0:
        raise ValueError("background_fraction must lie in (0, 1)")
    if float(foreground_fraction) + float(background_fraction) >= 1.0:
        raise ValueError("foreground and background seed fractions must be disjoint")

    flat_valid = np.flatnonzero(valid.reshape(-1))
    if flat_valid.size < 3:
        raise ValueError("At least three valid pixels are required")
    flat_values = values.reshape(-1)[flat_valid]
    order = np.argsort(flat_values, kind="mergesort")
    foreground_count = max(
        1, int(round(float(flat_valid.size) * float(foreground_fraction)))
    )
    background_count = max(
        1, int(round(float(flat_valid.size) * float(background_fraction)))
    )
    if foreground_count + background_count >= flat_valid.size:
        raise ValueError("Seed counts leave no ambiguous valid pixel")

    background_indices = flat_valid[order[:background_count]]
    foreground_indices = flat_valid[order[-foreground_count:]]
    foreground = np.zeros(values.size, dtype=bool)
    background = np.zeros(values.size, dtype=bool)
    foreground[foreground_indices] = True
    background[background_indices] = True
    foreground = foreground.reshape(values.shape)
    background = background.reshape(values.shape)
    if np.logical_and(foreground, background).any():
        raise RuntimeError("Foreground and background rank seeds overlap")
    return foreground, background


def _robust_edge_scale(features: np.ndarray, valid: np.ndarray) -> float:
    """Return a deterministic within-image scale for one feature branch."""

    differences: list[np.ndarray] = []
    right_valid = valid[:, 1:] & valid[:, :-1]
    if right_valid.any():
        right = np.linalg.norm(features[:, :, 1:] - features[:, :, :-1], axis=0)
        differences.append(right[right_valid])
    down_valid = valid[1:, :] & valid[:-1, :]
    if down_valid.any():
        down = np.linalg.norm(features[:, 1:, :] - features[:, :-1, :], axis=0)
        differences.append(down[down_valid])
    if not differences:
        return 1.0
    values = np.concatenate(differences)
    positive = values[values > 1e-8]
    if positive.size == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-6)


def prepare_geodesic_features(
    grayscale: np.ndarray,
    structural_features: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Normalize grayscale and frozen structural features with equal branches.

    Each branch is divided by its own median local edge magnitude and then by
    ``sqrt(2)``.  This makes the two modalities contribute equally without a
    fitted scalar weight.
    """

    image = np.asarray(grayscale, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("grayscale must be a finite two-dimensional array")
    structure = np.asarray(structural_features, dtype=np.float32)
    if structure.ndim != 3 or structure.shape[1:] != image.shape:
        raise ValueError("structural_features must have shape [C, H, W]")
    if not np.isfinite(structure).all():
        raise ValueError("structural_features must contain only finite values")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != image.shape or not valid.any():
        raise ValueError("valid_mask must align with and include grayscale pixels")

    image_mean = float(image[valid].mean())
    image_std = max(float(image[valid].std()), 1e-6)
    image_branch = np.clip((image - image_mean) / image_std, -5.0, 5.0)[None]

    norms = np.linalg.norm(structure, axis=0, keepdims=True)
    structure_branch = structure / np.maximum(norms, 1e-6)
    image_branch /= _robust_edge_scale(image_branch, valid)
    structure_branch /= _robust_edge_scale(structure_branch, valid)
    scale = np.float32(1.0 / sqrt(2.0))
    combined = np.concatenate(
        [image_branch * scale, structure_branch * scale], axis=0
    ).astype(np.float32)
    combined[:, ~valid] = 0.0
    return combined


def _edge_costs(
    features: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute deterministic undirected 8-neighbour graph costs."""

    _, height, width = features.shape
    spatial_floor = np.float32(1.0 / max(height, width))

    horizontal = np.linalg.norm(
        features[:, :, 1:] - features[:, :, :-1], axis=0
    ).astype(np.float32)
    horizontal += spatial_floor
    horizontal[~(valid[:, 1:] & valid[:, :-1])] = np.inf

    vertical = np.linalg.norm(
        features[:, 1:, :] - features[:, :-1, :], axis=0
    ).astype(np.float32)
    vertical += spatial_floor
    vertical[~(valid[1:, :] & valid[:-1, :])] = np.inf

    diagonal_right = np.linalg.norm(
        features[:, 1:, 1:] - features[:, :-1, :-1], axis=0
    ).astype(np.float32)
    diagonal_right = (diagonal_right + spatial_floor) * np.float32(sqrt(2.0))
    diagonal_right[~(valid[1:, 1:] & valid[:-1, :-1])] = np.inf

    diagonal_left = np.linalg.norm(
        features[:, 1:, :-1] - features[:, :-1, 1:], axis=0
    ).astype(np.float32)
    diagonal_left = (diagonal_left + spatial_floor) * np.float32(sqrt(2.0))
    diagonal_left[~(valid[1:, :-1] & valid[:-1, 1:])] = np.inf
    return horizontal, vertical, diagonal_right, diagonal_left


def multi_source_geodesic_distance(
    features: np.ndarray,
    seed_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute exact shortest-path distance on the local feature graph."""

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 3:
        raise ValueError("features must have shape [C, H, W]")
    valid = np.asarray(valid_mask, dtype=bool)
    seeds = np.asarray(seed_mask, dtype=bool)
    if valid.shape != feature_array.shape[1:] or seeds.shape != valid.shape:
        raise ValueError("features, seed_mask, and valid_mask shapes differ")
    if np.logical_and(seeds, ~valid).any():
        raise ValueError("seed_mask contains an invalid pixel")
    if not seeds.any():
        raise ValueError("seed_mask must contain at least one valid seed")

    height, width = valid.shape
    horizontal, vertical, diagonal_right, diagonal_left = _edge_costs(
        feature_array, valid
    )
    distance = np.full((height, width), np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for row, column in np.argwhere(seeds):
        distance[row, column] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(column)))

    while queue:
        current, row, column = heapq.heappop(queue)
        if current != distance[row, column]:
            continue
        neighbours: list[tuple[int, int, float]] = []
        if column + 1 < width:
            neighbours.append((row, column + 1, float(horizontal[row, column])))
        if column > 0:
            neighbours.append((row, column - 1, float(horizontal[row, column - 1])))
        if row + 1 < height:
            neighbours.append((row + 1, column, float(vertical[row, column])))
        if row > 0:
            neighbours.append((row - 1, column, float(vertical[row - 1, column])))
        if row + 1 < height and column + 1 < width:
            neighbours.append(
                (row + 1, column + 1, float(diagonal_right[row, column]))
            )
        if row > 0 and column > 0:
            neighbours.append(
                (row - 1, column - 1, float(diagonal_right[row - 1, column - 1]))
            )
        if row + 1 < height and column > 0:
            neighbours.append(
                (row + 1, column - 1, float(diagonal_left[row, column - 1]))
            )
        if row > 0 and column + 1 < width:
            neighbours.append(
                (row - 1, column + 1, float(diagonal_left[row - 1, column]))
            )
        for next_row, next_column, cost in neighbours:
            if not np.isfinite(cost):
                continue
            candidate = current + cost
            if candidate < distance[next_row, next_column]:
                distance[next_row, next_column] = candidate
                heapq.heappush(queue, (candidate, next_row, next_column))

    if not np.isfinite(distance[valid]).all():
        raise ValueError("valid_mask contains a graph component without any seed")
    distance[~valid] = np.inf
    return distance.astype(np.float32)


def exponential_geodesic_fusion(
    source_probability: np.ndarray,
    foreground_distance: np.ndarray,
    background_distance: np.ndarray,
    valid_mask: np.ndarray,
    *,
    ratio: float = 1.0,
) -> np.ndarray:
    """Fuse source probability with foreground/background geodesic cues.

    The exponential cue and confidence blend follow UM-CAM's GSE construction,
    while source foreground/background seeds are explicitly preserved.
    """

    source = _as_probability(source_probability, "source_probability")
    foreground = np.asarray(foreground_distance, dtype=np.float32)
    background = np.asarray(background_distance, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if foreground.shape != source.shape or background.shape != source.shape:
        raise ValueError("Distance maps and source_probability shapes differ")
    if valid.shape != source.shape or not valid.any():
        raise ValueError("valid_mask must align with source_probability")
    if not np.isfinite(foreground[valid]).all() or not np.isfinite(
        background[valid]
    ).all():
        raise ValueError("Distance maps must be finite within valid_mask")
    if float(ratio) <= 0.0:
        raise ValueError("ratio must be positive")

    probability_foreground = np.exp(-float(ratio) * foreground)
    probability_background = 1.0 - np.exp(-float(ratio) * background)
    geodesic = np.minimum(probability_background, 0.5)
    geodesic = np.where(probability_foreground > 0.5, probability_foreground, geodesic)
    conflict = (probability_background < 0.5) & (probability_foreground > 0.5)
    geodesic = np.where(
        conflict & (background < foreground), probability_background, geodesic
    )
    geodesic = np.where(
        conflict & (foreground < background), probability_foreground, geodesic
    )

    nearest = np.minimum(foreground, background)
    confidence = np.maximum(
        (np.exp(-float(ratio) * nearest) - 0.5) / 0.5,
        0.0,
    )
    fused = confidence * geodesic + (1.0 - confidence) * source
    valid_values = fused[valid]
    minimum = float(valid_values.min())
    maximum = float(valid_values.max())
    if maximum - minimum > 1e-8:
        fused = (fused - minimum) / (maximum - minimum)
    else:
        fused = source.copy()
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    fused[~valid] = 0.0
    return fused


def geodesic_seed_expansion(
    source_probability: np.ndarray,
    grayscale: np.ndarray,
    structural_features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    foreground_fraction: float = 0.01,
    background_fraction: float = 0.50,
    ratio: float = 1.0,
) -> GeodesicSeedExpansion:
    """Run deterministic image/feature geodesic seed expansion."""

    source = _as_probability(source_probability, "source_probability")
    valid = np.asarray(valid_mask, dtype=bool)
    foreground_seeds, background_seeds = exact_rank_seed_masks(
        source,
        valid,
        foreground_fraction=foreground_fraction,
        background_fraction=background_fraction,
    )
    features = prepare_geodesic_features(grayscale, structural_features, valid)
    foreground_distance = multi_source_geodesic_distance(
        features, foreground_seeds, valid
    )
    background_distance = multi_source_geodesic_distance(
        features, background_seeds, valid
    )
    refined = exponential_geodesic_fusion(
        source,
        foreground_distance,
        background_distance,
        valid,
        ratio=ratio,
    )
    refined[foreground_seeds] = 1.0
    refined[background_seeds] = 0.0
    return GeodesicSeedExpansion(
        probability=refined,
        foreground_seeds=foreground_seeds,
        background_seeds=background_seeds,
        foreground_distance=foreground_distance,
        background_distance=background_distance,
        diagnostics={
            "valid_pixels": int(valid.sum()),
            "foreground_seed_pixels": int(foreground_seeds.sum()),
            "background_seed_pixels": int(background_seeds.sum()),
            "ambiguous_pixels": int(
                np.logical_and(valid, ~(foreground_seeds | background_seeds)).sum()
            ),
            "foreground_fraction": float(foreground_fraction),
            "background_fraction": float(background_fraction),
            "ratio": float(ratio),
            "refined_min": float(refined[valid].min()),
            "refined_max": float(refined[valid].max()),
            "refined_mean": float(refined[valid].mean()),
        },
    )
