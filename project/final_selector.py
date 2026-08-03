from __future__ import annotations

"""Final candidate selector used by the thesis pipeline.

The selector intentionally has no access to images, masks, annotations, or
lesion area.  It combines the frozen G1 candidate logits and the frozen
upstream candidate scores with equal percentile-rank weight.
"""

import numpy as np


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return ascending average-tie percentile ranks in ``[0, 1]``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be one finite nonempty vector")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / max(len(array) - 1, 1)


def fixed_rank_fusion(g1_logits: np.ndarray, upstream_scores: np.ndarray) -> np.ndarray:
    """Fuse G1 and upstream evidence with the frozen 0.5/0.5 rule."""

    g1 = np.asarray(g1_logits, dtype=np.float64)
    upstream = np.asarray(upstream_scores, dtype=np.float64)
    if g1.shape != upstream.shape:
        raise ValueError("G1 logits and upstream scores must align")
    return 0.5 * (average_percentile_rank(g1) + average_percentile_rank(upstream))


def stable_select(fused_scores: np.ndarray, g1_logits: np.ndarray) -> int:
    """Select by fused score, then G1 logit, then lower frozen local index."""

    fused = np.asarray(fused_scores, dtype=np.float64)
    g1 = np.asarray(g1_logits, dtype=np.float64)
    if fused.ndim != 1 or fused.shape != g1.shape or not len(fused):
        raise ValueError("selector arrays must be aligned nonempty vectors")
    if not np.isfinite(fused).all() or not np.isfinite(g1).all():
        raise ValueError("selector arrays must be finite")
    return int(max(range(len(fused)), key=lambda index: (fused[index], g1[index], -index)))


def select_candidate(g1_logits: np.ndarray, upstream_scores: np.ndarray) -> tuple[int, np.ndarray]:
    fused = fixed_rank_fusion(g1_logits, upstream_scores)
    return stable_select(fused, g1_logits), fused


__all__ = [
    "average_percentile_rank",
    "fixed_rank_fusion",
    "select_candidate",
    "stable_select",
]
