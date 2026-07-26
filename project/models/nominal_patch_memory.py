from __future__ import annotations

"""Image-only nominal patch-memory primitives.

These functions contain no dataset or GT access. A caller may build the bank
only from clean-train normal radiographs selected by image-level labels.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def l2_normalize_rows(values: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Features must be finite with a final embedding dimension")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("Feature bank contains a zero-norm embedding")
    return values / norms


def make_seeded_random_projection(
    *,
    input_dim: int,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Create a frozen Gaussian Johnson-Lindenstrauss projection.

    A direct seeded matrix avoids a dataset-fitted PCA step and therefore
    cannot leak validation distribution into the nominal representation.
    """
    if input_dim <= 0 or output_dim <= 0 or output_dim > input_dim:
        raise ValueError("Projection dimensions require 0 < output_dim <= input_dim")
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    projection /= np.sqrt(float(output_dim))
    if not np.isfinite(projection).all():
        raise RuntimeError("Random projection contains non-finite values")
    return projection


def projection_sha256(projection: np.ndarray) -> str:
    projection = np.asarray(projection, dtype=np.float32)
    if projection.ndim != 2 or not np.isfinite(projection).all():
        raise ValueError("Projection must be a finite matrix")
    payload = {
        "dtype": "float32",
        "shape": list(projection.shape),
        "bytes_sha256": hashlib.sha256(
            projection.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def project_features(
    values: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    projection = np.asarray(projection, dtype=np.float32)
    if values.ndim < 2 or projection.ndim != 2:
        raise ValueError("Features/projection must have embedding dimensions")
    if values.shape[-1] != projection.shape[0]:
        raise ValueError("Feature and projection dimensions are incompatible")
    if not np.isfinite(values).all() or not np.isfinite(projection).all():
        raise ValueError("Features/projection must be finite")
    projected = values @ projection
    return l2_normalize_rows(projected)


def projected_bank_size_bytes(
    *,
    images: int,
    grid_height: int,
    grid_width: int,
    output_dim: int,
    bytes_per_value: int = 2,
) -> int:
    dimensions = (images, grid_height, grid_width, output_dim, bytes_per_value)
    if any(value <= 0 for value in dimensions):
        raise ValueError("Projected bank dimensions must be positive")
    return int(np.prod(dimensions, dtype=np.int64))


def retrieve_normal_context(
    query_global: np.ndarray,
    normal_global_bank: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visually nearest normal-image indices using cosine similarity."""
    query = np.asarray(query_global, dtype=np.float32)
    bank = np.asarray(normal_global_bank, dtype=np.float32)
    if query.ndim != 1 or bank.ndim != 2 or bank.shape[1] != query.shape[0]:
        raise ValueError("Global query/bank feature shapes are incompatible")
    if not 1 <= top_k <= bank.shape[0]:
        raise ValueError("top_k lies outside the normal-image bank")
    query = l2_normalize_rows(query[None])[0]
    bank = l2_normalize_rows(bank)
    similarities = bank @ query
    # Stable sorting makes ties deterministic and favors the lower bank index.
    indices = np.argsort(-similarities, kind="stable")[:top_k]
    return indices.astype(np.int64), similarities[indices].astype(np.float32)


def flatten_context_patch_bank(
    normal_patch_bank: np.ndarray,
    context_indices: np.ndarray,
) -> np.ndarray:
    """Select image-conditioned normal patches and flatten their spatial grid."""
    bank = np.asarray(normal_patch_bank, dtype=np.float32)
    indices = np.asarray(context_indices, dtype=np.int64)
    if bank.ndim != 4:
        raise ValueError("normal_patch_bank must have shape [images,height,width,dim]")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("context_indices must be a non-empty vector")
    if int(indices.min()) < 0 or int(indices.max()) >= bank.shape[0]:
        raise ValueError("context index lies outside normal_patch_bank")
    selected = bank[indices]
    return selected.reshape(-1, bank.shape[-1])


def cosine_anomaly_scores(
    query_patches: np.ndarray,
    normal_memory_patches: np.ndarray,
    *,
    query_chunk_size: int = 256,
    memory_chunk_size: int = 8192,
) -> np.ndarray:
    """Score each query patch by one minus its maximum normal cosine match."""
    query = np.asarray(query_patches, dtype=np.float32)
    memory = np.asarray(normal_memory_patches, dtype=np.float32)
    if query.ndim != 2 or memory.ndim != 2 or query.shape[1] != memory.shape[1]:
        raise ValueError("Query and memory patch feature shapes are incompatible")
    if query.shape[0] == 0 or memory.shape[0] == 0:
        raise ValueError("Query and memory patch banks must be non-empty")
    if query_chunk_size <= 0 or memory_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")
    query = l2_normalize_rows(query)
    memory = l2_normalize_rows(memory)
    best = np.full(query.shape[0], -np.inf, dtype=np.float32)
    for query_start in range(0, query.shape[0], query_chunk_size):
        query_end = min(query_start + query_chunk_size, query.shape[0])
        local_best = np.full(query_end - query_start, -np.inf, dtype=np.float32)
        for memory_start in range(0, memory.shape[0], memory_chunk_size):
            memory_end = min(memory_start + memory_chunk_size, memory.shape[0])
            similarities = query[query_start:query_end] @ memory[memory_start:memory_end].T
            local_best = np.maximum(local_best, similarities.max(axis=1))
        best[query_start:query_end] = local_best
    return np.clip(1.0 - best, 0.0, 2.0).astype(np.float32)


def spatial_cosine_anomaly_scores(
    query_grid: np.ndarray,
    normal_context_grids: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Compare each patch only with nearby coordinates in retrieved normals.

    Restricting matches spatially prevents a query patch from matching an
    unrelated but visually similar patch elsewhere in the radiograph.
    """
    query = np.asarray(query_grid, dtype=np.float32)
    context = np.asarray(normal_context_grids, dtype=np.float32)
    if query.ndim != 3 or context.ndim != 4:
        raise ValueError("Expected query [H,W,D] and context [N,H,W,D]")
    if context.shape[1:] != query.shape:
        raise ValueError("Query and normal context grids must align")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    query = l2_normalize_rows(query)
    context = l2_normalize_rows(context)
    height, width, _dim = query.shape
    scores = np.empty((height, width), dtype=np.float32)
    for row in range(height):
        row_start = max(0, row - radius)
        row_end = min(height, row + radius + 1)
        for column in range(width):
            column_start = max(0, column - radius)
            column_end = min(width, column + radius + 1)
            local_memory = context[
                :, row_start:row_end, column_start:column_end, :
            ].reshape(-1, context.shape[-1])
            best_similarity = float((local_memory @ query[row, column]).max())
            scores[row, column] = np.clip(1.0 - best_similarity, 0.0, 2.0)
    return scores


def blend_anomaly_scores(
    unrestricted: np.ndarray,
    spatial: np.ndarray,
    *,
    spatial_weight: float,
) -> np.ndarray:
    unrestricted = np.asarray(unrestricted, dtype=np.float32)
    spatial = np.asarray(spatial, dtype=np.float32)
    if unrestricted.shape != spatial.shape or unrestricted.ndim != 2:
        raise ValueError("Anomaly score maps must be aligned 2D arrays")
    if not 0.0 <= spatial_weight <= 1.0:
        raise ValueError("spatial_weight must lie in [0,1]")
    if not np.isfinite(unrestricted).all() or not np.isfinite(spatial).all():
        raise ValueError("Anomaly score maps must be finite")
    return (
        (1.0 - spatial_weight) * unrestricted + spatial_weight * spatial
    ).astype(np.float32)


@dataclass(frozen=True)
class FrozenNormalCalibration:
    """Empirical calibration fitted only on clean-train normal patch scores."""

    sorted_normal_scores: np.ndarray

    @classmethod
    def fit(cls, normal_scores: np.ndarray) -> "FrozenNormalCalibration":
        scores = np.asarray(normal_scores, dtype=np.float32).reshape(-1)
        if scores.size < 2 or not np.isfinite(scores).all():
            raise ValueError("Normal calibration requires at least two finite scores")
        scores = np.sort(scores, kind="stable")
        scores.setflags(write=False)
        return cls(sorted_normal_scores=scores)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map raw distances to their empirical normal-bank percentile.

        Unlike per-image min-max normalization, this does not force every
        normal validation image to contain a high-scoring patch.
        """
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Values to calibrate must be finite")
        ranks = np.searchsorted(
            self.sorted_normal_scores, values, side="right"
        ).astype(np.float32)
        return ranks / float(self.sorted_normal_scores.size)

    def metadata(self) -> dict[str, float | int]:
        return {
            "normal_patch_scores": int(self.sorted_normal_scores.size),
            "minimum": float(self.sorted_normal_scores[0]),
            "median": float(np.median(self.sorted_normal_scores)),
            "maximum": float(self.sorted_normal_scores[-1]),
        }
