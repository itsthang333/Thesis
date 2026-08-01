from __future__ import annotations

import numpy as np

from models.rich_gallery_cross_source_consensus import (
    consensus_selector_scores,
    cross_source_max_iou,
    freeze_consensus_choices,
)


def test_cross_source_max_iou_is_exact_and_ignores_same_source() -> None:
    masks = np.zeros((4, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :2, :2] = True  # identical but same source as candidate 0
    masks[2, :2, 1:3] = True  # different source, IoU 2/6
    masks[3, 2:, 2:] = True
    scores = cross_source_max_iou(masks, np.asarray([0, 0, 1, 2]), block_size=1)
    assert np.isclose(scores[0], 2.0 / 6.0)
    assert np.isclose(scores[1], 2.0 / 6.0)
    assert np.isclose(scores[2], 2.0 / 6.0)
    assert scores[3] == 0.0


def test_consensus_rules_are_fixed_and_can_change_the_baseline_choice() -> None:
    g1 = np.asarray([3.0, 2.0, 1.0])
    upstream = np.asarray([1.0, 2.0, 3.0])
    consensus = np.asarray([0.0, 0.9, 0.1])
    scores = consensus_selector_scores(g1, upstream, consensus)
    assert set(scores) == {
        "g1_upstream_baseline",
        "cross_source_consensus_only",
        "g1_upstream_consensus_equal",
        "g1_upstream_consensus_product",
    }
    choices = freeze_consensus_choices(g1, upstream, consensus)
    assert choices["cross_source_consensus_only"] == 1
    assert choices["g1_upstream_consensus_product"] == 1
    assert choices["g1_upstream_baseline"] == 0  # stable raw-logit tie break
