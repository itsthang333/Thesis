from __future__ import annotations

import numpy as np

from project.analyze_rich_gallery_bas_b2_collapse import rank_correlation, regime_summary


def test_rank_correlation_detects_area_proxy() -> None:
    area = np.asarray([0.01, 0.02, 0.04, 0.08])
    assert np.isclose(rank_correlation(area**0.5, area), 1.0)
    assert np.isclose(rank_correlation(area[::-1], area), -1.0)


def test_regime_summary_preserves_paired_dice_and_misses() -> None:
    rows = [
        {
            "baseline_dice": 0.2,
            "primary_dice": 0.1,
            "primary_delta": -0.1,
            "baseline_miss": 0,
            "primary_miss": 1,
            "primary_area_gt": 10.0,
            "score_dominance_gap": 0.2,
        },
        {
            "baseline_dice": 0.0,
            "primary_dice": 0.1,
            "primary_delta": 0.1,
            "baseline_miss": 1,
            "primary_miss": 0,
            "primary_area_gt": 2.0,
            "score_dominance_gap": 0.0,
        },
    ]
    result = regime_summary(rows)
    assert result["n"] == 2
    assert np.isclose(result["baseline_dice"], 0.1)
    assert np.isclose(result["primary_dice"], 0.1)
    assert result["baseline_misses"] == result["primary_misses"] == 1
    assert result["primary_area_gt_median"] == 6.0
