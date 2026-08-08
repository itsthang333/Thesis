from __future__ import annotations

import math

import numpy as np

from analyze_x4_risk_coverage import (
    average_ranks,
    binary_auroc,
    risk_coverage_rows,
    spearman_correlation,
)


def test_average_ranks_handles_exact_ties():
    np.testing.assert_allclose(average_ranks(np.asarray([2.0, 1.0, 2.0])), [2.5, 1.0, 2.5])


def test_binary_auroc_perfect_and_reversed():
    labels = np.asarray([0, 0, 1, 1])
    assert binary_auroc(labels, np.asarray([0.0, 0.1, 0.9, 1.0])) == 1.0
    assert binary_auroc(labels, np.asarray([1.0, 0.9, 0.1, 0.0])) == 0.0


def test_binary_auroc_ties_and_undefined():
    assert binary_auroc(np.asarray([0, 1]), np.asarray([0.5, 0.5])) == 0.5
    assert math.isnan(binary_auroc(np.asarray([1, 1]), np.asarray([0.1, 0.2])))


def test_spearman_uses_tie_correct_ranks():
    assert spearman_correlation(np.asarray([1.0, 2.0, 3.0]), np.asarray([4.0, 5.0, 6.0])) == 1.0
    assert spearman_correlation(np.asarray([1.0, 2.0, 3.0]), np.asarray([6.0, 5.0, 4.0])) == -1.0


def test_risk_coverage_keeps_highest_confidence_and_uses_ceil():
    rows = [
        {"image_id": "b", "confidence": 0.9, "dice": 0.0, "complete_miss": 1},
        {"image_id": "a", "confidence": 0.9, "dice": 1.0, "complete_miss": 0},
        {"image_id": "c", "confidence": 0.8, "dice": 0.5, "complete_miss": 0},
        {"image_id": "d", "confidence": 0.7, "dice": 0.5, "complete_miss": 0},
        {"image_id": "e", "confidence": 0.6, "dice": 0.5, "complete_miss": 0},
    ]
    result = risk_coverage_rows(rows)
    by_coverage = {row["coverage"]: row for row in result}
    assert by_coverage[0.8]["retained_images"] == 4
    assert by_coverage[0.4]["retained_images"] == 2
    assert by_coverage[0.4]["mean_dice"] == 0.5
