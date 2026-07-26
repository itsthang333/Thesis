from __future__ import annotations

import unittest

import numpy as np

from tools.audit_affinity_guided_proposal_selector import (
    dice,
    paired_report,
    percentile,
    subgroup,
)


class AffinityGuidedSelectorAuditTests(unittest.TestCase):
    def test_dice_includes_complete_miss(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        target[0, 0] = True
        prediction = np.zeros_like(target)
        self.assertEqual(dice(prediction, target), 0.0)

    def test_empty_normal_dice_is_one(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(dice(empty, empty), 1.0)

    def test_subgroup_boundaries_are_frozen(self) -> None:
        self.assertEqual(subgroup(0.009999), "small")
        self.assertEqual(subgroup(0.01), "medium")
        self.assertEqual(subgroup(0.049999), "medium")
        self.assertEqual(subgroup(0.05), "large")

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([0.0, 1.0], 0.5), 0.5)

    def test_paired_report_is_deterministic_and_group_resampled(self) -> None:
        rows = [
            {"group_id": "a", "baseline": 0.1, "candidate": 0.2},
            {"group_id": "a", "baseline": 0.3, "candidate": 0.4},
            {"group_id": "b", "baseline": 0.2, "candidate": 0.5},
        ]
        first = paired_report(
            rows,
            baseline_key="baseline",
            candidate_key="candidate",
            iterations=1000,
            seed=42,
        )
        second = paired_report(
            rows,
            baseline_key="baseline",
            candidate_key="candidate",
            iterations=1000,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["images"], 3)
        self.assertEqual(first["groups"], 2)
        self.assertAlmostEqual(first["mean_delta"], (0.1 + 0.1 + 0.3) / 3)


if __name__ == "__main__":
    unittest.main()
