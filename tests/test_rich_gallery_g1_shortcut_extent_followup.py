from __future__ import annotations

import numpy as np

from project.evaluate_rich_gallery_g1_shortcut_extent_followup import (
    _average_percentile_rank,
    _choose,
)


def test_average_percentile_rank_uses_average_ties() -> None:
    result = _average_percentile_rank(np.asarray([3.0, 1.0, 3.0, 2.0]))
    assert np.allclose(result, [2.5 / 3.0, 0.0, 2.5 / 3.0, 1.0 / 3.0])


def test_choose_uses_g1_then_lower_index_for_ties() -> None:
    scores = np.asarray([1.0, 1.0, 1.0, 2.0])
    g1 = np.asarray([0.2, 0.7, 0.7, 9.0])
    eligible = np.asarray([True, True, True, False])
    assert _choose(scores, g1, eligible) == 1


def test_choose_rejects_empty_eligible_set() -> None:
    with np.testing.assert_raises(ValueError):
        _choose(np.ones(2), np.ones(2), np.zeros(2, dtype=bool))

