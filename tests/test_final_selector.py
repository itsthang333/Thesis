from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from final_selector import average_percentile_rank, fixed_rank_fusion, select_candidate
from evaluation.frozen_test_guard import verify_frozen_test_config


class FinalSelectorTests(unittest.TestCase):
    def test_average_ties_are_deterministic(self) -> None:
        actual = average_percentile_rank(np.asarray([4.0, 1.0, 4.0, 2.0]))
        np.testing.assert_allclose(actual, [5 / 6, 0.0, 5 / 6, 1 / 3])

    def test_fixed_rule_is_equal_rank_fusion(self) -> None:
        g1 = np.asarray([1.0, 3.0, 2.0])
        upstream = np.asarray([3.0, 1.0, 2.0])
        np.testing.assert_allclose(fixed_rank_fusion(g1, upstream), [0.5, 0.5, 0.5])

    def test_tie_breaks_by_g1_then_lower_index(self) -> None:
        selected, _ = select_candidate(
            np.asarray([2.0, 3.0, 3.0]),
            np.asarray([3.0, 2.0, 2.0]),
        )
        self.assertEqual(selected, 1)

    def test_nonfinite_scores_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            select_candidate(np.asarray([1.0, np.nan]), np.asarray([0.0, 1.0]))

    def test_test_access_requires_final_lock_but_validation_does_not(self) -> None:
        self.assertIsNone(verify_frozen_test_config(None, split="val"))
        with self.assertRaisesRegex(ValueError, "frozen-config is required"):
            verify_frozen_test_config(None, split="test")


if __name__ == "__main__":
    unittest.main()
