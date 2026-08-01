from __future__ import annotations

import numpy as np

from project.analyze_rich_gallery_oracle_feature_gap import (
    _candidate_features,
    _delta_summary,
    _distribution,
    _percentile_rank,
)


def test_percentile_rank_preserves_average_ties() -> None:
    ranks = _percentile_rank(np.asarray([3.0, 1.0, 1.0, 2.0]))
    assert np.allclose(ranks, np.asarray([1.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]))


def test_candidate_features_measure_border_bone_and_prompt_contrast() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    prompt = np.zeros((20, 20), dtype=np.float32)
    prompt[mask] = 0.8
    bone = np.zeros((20, 20), dtype=bool)
    bone[5:10, 5:15] = True
    result = _candidate_features(mask, prompt, bone)
    assert np.isclose(result["area"], 0.25)
    assert result["border_fraction"] == 0.0
    assert np.isclose(result["bone_inside_fraction"], 0.5)
    assert np.isclose(result["bbox_fill"], 1.0)
    assert result["prompt_ring_contrast"] > 0.79
    assert result["components"] == 1.0


def test_delta_summary_and_distribution_are_paired_and_normalized() -> None:
    rows = [
        {"delta_area": -2.0, "source": "a"},
        {"delta_area": -1.0, "source": "a"},
        {"delta_area": 3.0, "source": "b"},
        {"delta_area": 0.0, "source": "b"},
    ]
    summary = _delta_summary(rows, "area")
    assert np.isclose(summary["median"], -0.5)
    assert np.isclose(summary["oracle_lower_fraction"], 0.5)
    assert np.isclose(summary["oracle_higher_fraction"], 0.25)
    assert np.isclose(summary["equal_fraction"], 0.25)
    assert _distribution(rows, "source") == {"a": 0.5, "b": 0.5}
