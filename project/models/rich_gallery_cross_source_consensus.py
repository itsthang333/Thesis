from __future__ import annotations

"""Annotation-free cross-source agreement for a frozen proposal gallery."""

import numpy as np

from models.rich_gallery_g2_objective import average_percentile_rank, stable_select


VARIANTS = (
    "g1_upstream_baseline",
    "cross_source_consensus_only",
    "g1_upstream_consensus_equal",
    "g1_upstream_consensus_product",
)


def cross_source_max_iou(
    masks: np.ndarray,
    source_ids: np.ndarray,
    *,
    block_size: int = 16,
) -> np.ndarray:
    """Return each mask's exact maximum IoU with a different-source mask.

    Masks are bit-packed and processed in bounded blocks, so the computation
    remains exact at the native candidate grid without materializing a dense
    ``N x N x H x W`` tensor.
    """

    array = np.asarray(masks, dtype=bool)
    sources = np.asarray(source_ids)
    if array.ndim != 3 or sources.shape != (len(array),):
        raise ValueError("masks/source_ids must have shapes [N,H,W]/[N]")
    if not len(array) or block_size < 1:
        raise ValueError("candidate set and block size must be positive")
    if len(np.unique(sources)) < 2:
        raise ValueError("cross-source consensus requires at least two sources")

    flat = array.reshape(len(array), -1)
    packed = np.packbits(flat, axis=1)
    areas = flat.sum(axis=1, dtype=np.int64)
    lookup = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
    result = np.zeros(len(array), dtype=np.float64)
    for source in np.unique(sources):
        left_indices = np.flatnonzero(sources == source)
        right_indices = np.flatnonzero(sources != source)
        for left_start in range(0, len(left_indices), block_size):
            left = left_indices[left_start : left_start + block_size]
            best = np.zeros(len(left), dtype=np.float64)
            for right_start in range(0, len(right_indices), block_size):
                right = right_indices[right_start : right_start + block_size]
                intersections = lookup[
                    np.bitwise_and(packed[left, None, :], packed[None, right, :])
                ].sum(axis=2, dtype=np.int64)
                unions = areas[left, None] + areas[None, right] - intersections
                iou = np.divide(
                    intersections,
                    unions,
                    out=np.zeros_like(intersections, dtype=np.float64),
                    where=unions > 0,
                )
                best = np.maximum(best, iou.max(axis=1))
            result[left] = best
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("invalid cross-source consensus score")
    return result


def consensus_selector_scores(
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
    consensus_iou: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the four predeclared score vectors without fitted weights."""

    g1 = np.asarray(g1_logits, dtype=np.float64)
    upstream = np.asarray(upstream_scores, dtype=np.float64)
    consensus = np.asarray(consensus_iou, dtype=np.float64)
    if g1.ndim != 1 or g1.shape != upstream.shape or g1.shape != consensus.shape:
        raise ValueError("selector inputs must be aligned one-dimensional vectors")
    if not len(g1) or not all(np.isfinite(values).all() for values in (g1, upstream, consensus)):
        raise ValueError("selector inputs must be finite and nonempty")
    g1_rank = average_percentile_rank(g1)
    upstream_rank = average_percentile_rank(upstream)
    consensus_rank = average_percentile_rank(consensus)
    baseline = 0.5 * (g1_rank + upstream_rank)
    return {
        "g1_upstream_baseline": baseline,
        "cross_source_consensus_only": consensus,
        "g1_upstream_consensus_equal": (g1_rank + upstream_rank + consensus_rank) / 3.0,
        "g1_upstream_consensus_product": np.sqrt(baseline * consensus_rank),
    }


def freeze_consensus_choices(
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
    consensus_iou: np.ndarray,
) -> dict[str, int]:
    scores = consensus_selector_scores(g1_logits, upstream_scores, consensus_iou)
    return {
        name: stable_select(values, np.asarray(g1_logits, dtype=np.float64))
        for name, values in scores.items()
    }


__all__ = [
    "VARIANTS",
    "consensus_selector_scores",
    "cross_source_max_iou",
    "freeze_consensus_choices",
]
