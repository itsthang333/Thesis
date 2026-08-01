from __future__ import annotations

import numpy as np

from project.analyze_rich_gallery_g1_two_score_identifiability import (
    alpha_grid_selected_quality,
    pareto_frontier,
)


def test_pareto_frontier_rejects_jointly_dominated_candidate() -> None:
    g1 = np.asarray([1.0, 0.5, 0.8, 0.7])
    upstream = np.asarray([0.2, 0.9, 0.7, 0.6])
    assert pareto_frontier(g1, upstream).tolist() == [True, True, True, False]


def test_alpha_grid_contains_equal_rank_baseline_and_endpoints() -> None:
    g1 = np.asarray([2.0, 1.0, 0.0])
    upstream = np.asarray([0.0, 1.0, 2.0])
    quality = np.asarray([0.2, 0.8, 0.3])
    alphas = np.asarray([0.0, 0.5, 1.0])
    selected, selected_quality = alpha_grid_selected_quality(
        g1, upstream, quality, alphas
    )
    assert selected.tolist() == [2, 0, 0]
    assert np.allclose(selected_quality, [0.3, 0.2, 0.2])
