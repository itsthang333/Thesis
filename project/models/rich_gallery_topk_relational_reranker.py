from __future__ import annotations

"""Tumor-conditioned, bounded relational reranking for a frozen gallery.

The candidate neighborhood is defined only by the immutable G1/upstream
baseline.  Spatial agreement can refine candidates inside that neighborhood;
it cannot introduce a globally frequent anatomy proposal from the gallery
tail.  No annotation, lesion-size, image-coordinate or fitted parameter enters
the rule.
"""

import numpy as np

from models.rich_gallery_g2_objective import average_percentile_rank, stable_select


VARIANTS = (
    "g1_upstream_baseline",
    "top10_cross_source_relational_product",
)
TOP_K = 10


def baseline_rank_score(
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
) -> np.ndarray:
    g1 = np.asarray(g1_logits, dtype=np.float64)
    upstream = np.asarray(upstream_scores, dtype=np.float64)
    if g1.ndim != 1 or upstream.shape != g1.shape or not len(g1):
        raise ValueError("G1/upstream scores must be aligned nonempty vectors")
    if not np.isfinite(g1).all() or not np.isfinite(upstream).all():
        raise ValueError("G1/upstream scores must be finite")
    return 0.5 * (average_percentile_rank(g1) + average_percentile_rank(upstream))


def stable_topk_indices(
    primary: np.ndarray,
    tie_break: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    tie_break = np.asarray(tie_break, dtype=np.float64)
    if primary.ndim != 1 or tie_break.shape != primary.shape or k < 1:
        raise ValueError("top-k inputs are invalid")
    indices = np.arange(len(primary), dtype=np.int64)
    order = np.lexsort((indices, -tie_break, -primary))
    return order[: min(k, len(order))]


def topk_cross_source_support(
    masks: np.ndarray,
    source_ids: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline-weighted cross-source support inside fixed top-k.

    For candidate ``c``, support is the mean across other source identities of
    ``max_d IoU(c,d) * baseline(d)``.  Source multiplicity therefore cannot
    increase support.  Non-top-k candidates receive zero support.
    """

    array = np.asarray(masks, dtype=bool)
    sources = np.asarray(source_ids)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    if array.ndim != 3 or sources.shape != (len(array),) or baseline.shape != (len(array),):
        raise ValueError("mask/source/baseline arrays are not aligned")
    if not len(array) or top_k < 1 or not np.isfinite(baseline).all():
        raise ValueError("relational support inputs are invalid")

    top = stable_topk_indices(baseline, baseline, k=top_k)
    support = np.zeros(len(array), dtype=np.float64)
    if len(np.unique(sources[top])) < 2:
        return support, top

    flat = array[top].reshape(len(top), -1)
    areas = flat.sum(axis=1, dtype=np.int64)
    intersections = flat.astype(np.int32) @ flat.astype(np.int32).T
    unions = areas[:, None] + areas[None, :] - intersections
    pair_iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    top_sources = sources[top]
    top_baseline = baseline[top]
    for row, candidate_index in enumerate(top):
        by_source: list[float] = []
        for other_source in np.unique(top_sources[top_sources != top_sources[row]]):
            members = top_sources == other_source
            by_source.append(float(np.max(pair_iou[row, members] * top_baseline[members])))
        support[candidate_index] = float(np.mean(by_source)) if by_source else 0.0
    if not np.isfinite(support).all() or np.any((support < 0.0) | (support > 1.0)):
        raise RuntimeError("top-k relational support is invalid")
    return support, top


def relational_scores_and_choices(
    masks: np.ndarray,
    source_ids: np.ndarray,
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
    *,
    top_k: int = TOP_K,
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, np.ndarray]:
    baseline = baseline_rank_score(g1_logits, upstream_scores)
    support, top = topk_cross_source_support(
        masks,
        source_ids,
        baseline,
        top_k=top_k,
    )
    # ``stable_select`` deliberately rejects non-finite arrays.  Baseline and
    # relational scores lie in [0,1], so -1 is an exact finite exclusion
    # sentinel for candidates outside the immutable top-k neighborhood.
    relational = np.full(len(baseline), -1.0, dtype=np.float64)
    if float(support[top].max(initial=0.0)) <= 0.0:
        relational = baseline.copy()
    else:
        relational[top] = np.sqrt(baseline[top] * support[top])
    scores = {
        "g1_upstream_baseline": baseline,
        "top10_cross_source_relational_product": relational,
    }
    tie = np.asarray(g1_logits, dtype=np.float64)
    choices = {name: stable_select(values, tie) for name, values in scores.items()}
    return scores, choices, support, top


__all__ = [
    "TOP_K",
    "VARIANTS",
    "baseline_rank_score",
    "relational_scores_and_choices",
    "stable_topk_indices",
    "topk_cross_source_support",
]
