from __future__ import annotations

import numpy as np

from models.rich_gallery_topk_relational_reranker import (
    baseline_rank_score,
    relational_scores_and_choices,
    stable_topk_indices,
    topk_cross_source_support,
)


def test_stable_topk_uses_primary_then_tie_break_then_index() -> None:
    result = stable_topk_indices(
        np.asarray([0.5, 0.5, 0.4, 0.5]),
        np.asarray([0.0, 2.0, 9.0, 2.0]),
        k=3,
    )
    assert result.tolist() == [1, 3, 0]


def test_cross_source_support_ignores_same_source_multiplicity() -> None:
    masks = np.zeros((4, 8, 8), dtype=bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 1:5, 1:5] = True
    masks[2, 1:5, 1:5] = True
    masks[3, 5:7, 5:7] = True
    baseline = np.asarray([1.0, 0.8, 0.6, 0.7])
    support, top = topk_cross_source_support(
        masks,
        np.asarray([0, 0, 0, 1]),
        baseline,
        top_k=4,
    )
    assert set(top.tolist()) == {0, 1, 2, 3}
    assert support[0] == 0.0
    assert support[3] == 0.0


def test_relational_product_selects_high_baseline_supported_candidate() -> None:
    masks = np.zeros((4, 8, 8), dtype=bool)
    masks[0, 0:3, 0:3] = True
    masks[1, 4:7, 4:7] = True
    masks[2, 4:7, 4:7] = True
    masks[3, 1:3, 5:7] = True
    sources = np.asarray([0, 0, 1, 1])
    g1 = np.asarray([4.0, 3.0, 2.0, 1.0])
    upstream = np.asarray([4.0, 3.0, 2.0, 1.0])
    baseline = baseline_rank_score(g1, upstream)
    assert int(np.argmax(baseline)) == 0
    _scores, choices, support, _top = relational_scores_and_choices(
        masks,
        sources,
        g1,
        upstream,
        top_k=4,
    )
    assert support[1] > 0.0 and support[2] > 0.0
    assert choices["g1_upstream_baseline"] == 0
    assert choices["top10_cross_source_relational_product"] == 1
    assert np.isfinite(_scores["top10_cross_source_relational_product"]).all()


def test_single_source_topk_falls_back_to_baseline() -> None:
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[:, 1:4, 1:4] = True
    g1 = np.asarray([0.0, 2.0, 1.0])
    upstream = np.asarray([0.0, 2.0, 1.0])
    scores, choices, support, _top = relational_scores_and_choices(
        masks,
        np.asarray([0, 0, 0]),
        g1,
        upstream,
        top_k=3,
    )
    assert np.array_equal(support, np.zeros(3))
    assert np.array_equal(
        scores["g1_upstream_baseline"],
        scores["top10_cross_source_relational_product"],
    )
    assert choices["top10_cross_source_relational_product"] == choices["g1_upstream_baseline"]
