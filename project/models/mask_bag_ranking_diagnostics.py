from __future__ import annotations

"""Post-freeze diagnostics for a proposal selector.

This module is evaluation-only.  It accepts already frozen candidate scores and
candidate qualities; it does not load annotations, train a model, or choose a
configuration.  Training code must not import it.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_TOP_K = (1, 3, 5, 10)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, with smaller values ranked first."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman(scores: np.ndarray, qualities: np.ndarray) -> float | None:
    if len(scores) < 2:
        return None
    score_ranks = _average_ranks(scores)
    quality_ranks = _average_ranks(qualities)
    if np.ptp(score_ranks) == 0.0 or np.ptp(quality_ranks) == 0.0:
        return None
    return float(np.corrcoef(score_ranks, quality_ranks)[0, 1])


def candidate_ranking_diagnostics(
    candidate_scores: Sequence[float] | np.ndarray,
    candidate_quality: Sequence[float] | np.ndarray,
    *,
    valid: Sequence[bool] | np.ndarray | None = None,
    top_k: Iterable[int] = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Measure how well frozen scores rank frozen candidate quality.

    Candidate quality may only be derived from validation annotations after the
    score manifest has been frozen.  Oracle reach is tie-aware: it uses the
    highest-ranked candidate attaining the best quality.
    """

    scores = np.asarray(candidate_scores, dtype=np.float64)
    quality = np.asarray(candidate_quality, dtype=np.float64)
    if scores.ndim != 1 or quality.ndim != 1 or scores.shape != quality.shape:
        raise ValueError("candidate_scores and candidate_quality must be matching 1D arrays")
    valid_mask = (
        np.ones(len(scores), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid_mask.shape != scores.shape:
        raise ValueError("valid must match candidate_scores")
    if not np.isfinite(scores[valid_mask]).all() or not np.isfinite(
        quality[valid_mask]
    ).all():
        raise ValueError("valid candidate scores and qualities must be finite")
    requested_top_k = tuple(sorted({int(value) for value in top_k}))
    if not requested_top_k or requested_top_k[0] <= 0:
        raise ValueError("top_k must contain positive integers")

    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return {
            "candidate_count": 0,
            "selected_candidate_index": None,
            "oracle_candidate_index": None,
            "oracle_best_rank": None,
            "selected_quality": 0.0,
            "oracle_quality": 0.0,
            "selected_to_oracle_regret": 0.0,
            "score_quality_spearman": None,
            "top_k_oracle_reach": {str(k): False for k in requested_top_k},
            "top_k_best_quality": {str(k): 0.0 for k in requested_top_k},
            "top_k_regret": {str(k): 0.0 for k in requested_top_k},
        }

    # Lexsort's last key is primary. Candidate index is the deterministic
    # secondary key for equal frozen scores.
    order = valid_indices[
        np.lexsort((valid_indices, -scores[valid_indices]))
    ]
    selected_index = int(order[0])
    oracle_quality = float(np.max(quality[valid_indices]))
    oracle_ties = set(
        int(index)
        for index in valid_indices[quality[valid_indices] == oracle_quality]
    )
    oracle_position = next(
        position for position, index in enumerate(order) if int(index) in oracle_ties
    )
    oracle_index = int(order[oracle_position])
    selected_quality = float(quality[selected_index])

    top_k_best_quality: dict[str, float] = {}
    top_k_regret: dict[str, float] = {}
    top_k_oracle_reach: dict[str, bool] = {}
    for k in requested_top_k:
        prefix = order[: min(k, len(order))]
        best = float(np.max(quality[prefix]))
        top_k_best_quality[str(k)] = best
        top_k_regret[str(k)] = oracle_quality - best
        top_k_oracle_reach[str(k)] = any(int(index) in oracle_ties for index in prefix)

    return {
        "candidate_count": int(len(valid_indices)),
        "selected_candidate_index": selected_index,
        "oracle_candidate_index": oracle_index,
        "oracle_best_rank": int(oracle_position + 1),
        "selected_quality": selected_quality,
        "oracle_quality": oracle_quality,
        "selected_to_oracle_regret": oracle_quality - selected_quality,
        "score_quality_spearman": _spearman(
            scores[valid_indices], quality[valid_indices]
        ),
        "top_k_oracle_reach": top_k_oracle_reach,
        "top_k_best_quality": top_k_best_quality,
        "top_k_regret": top_k_regret,
    }


def summarize_ranking_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    subgroup_names: Sequence[str] = ("overall", "small", "medium", "large"),
    top_k: Iterable[int] = DEFAULT_TOP_K,
) -> dict[str, dict[str, Any]]:
    """Aggregate frozen per-image ranking rows without dropping misses."""

    requested_top_k = tuple(sorted({int(value) for value in top_k}))
    result: dict[str, dict[str, Any]] = {}
    for subgroup in subgroup_names:
        selected = [
            row
            for row in rows
            if subgroup == "overall" or row.get("size_group") == subgroup
        ]
        if not selected:
            raise ValueError(f"subgroup has no rows: {subgroup}")
        regrets = np.asarray(
            [float(row["selected_to_oracle_regret"]) for row in selected],
            dtype=np.float64,
        )
        correlations = [
            float(row["score_quality_spearman"])
            for row in selected
            if row.get("score_quality_spearman") is not None
        ]
        result[subgroup] = {
            "n": len(selected),
            "mean_selected_to_oracle_regret": float(regrets.mean()),
            "median_selected_to_oracle_regret": float(np.median(regrets)),
            "mean_score_quality_spearman": (
                float(np.mean(correlations)) if correlations else None
            ),
            "undefined_spearman_images": len(selected) - len(correlations),
            "top_k_oracle_reach_rate": {
                str(k): float(
                    np.mean(
                        [
                            bool(row["top_k_oracle_reach"][str(k)])
                            for row in selected
                        ]
                    )
                )
                for k in requested_top_k
            },
            "mean_top_k_regret": {
                str(k): float(
                    np.mean(
                        [float(row["top_k_regret"][str(k)]) for row in selected]
                    )
                )
                for k in requested_top_k
            },
            "complete_misses": int(
                sum(bool(row["selected_complete_miss"]) for row in selected)
            ),
            "recovered_baseline_misses": int(
                sum(
                    bool(row["baseline_complete_miss"])
                    and not bool(row["selected_complete_miss"])
                    for row in selected
                )
            ),
            "lost_baseline_hits": int(
                sum(
                    not bool(row["baseline_complete_miss"])
                    and bool(row["selected_complete_miss"])
                    for row in selected
                )
            ),
        }
    return result
