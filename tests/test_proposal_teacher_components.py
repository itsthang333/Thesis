from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pseudo.tumor_morphology import build_probability_components


class ProposalTeacherComponentTests(unittest.TestCase):
    def test_thresholds_filters_and_ranks_components_by_confidence(self) -> None:
        probability = np.zeros((32, 32), dtype=np.float32)
        probability[2:8, 2:8] = 0.91
        probability[18:25, 19:26] = 0.88
        probability[29, 29] = 0.99
        original = probability.copy()

        support, components = build_probability_components(
            probability,
            threshold=0.85,
            min_component_area=4,
            max_components=2,
            points_per_component=3,
            bbox_padding_ratio=0.0,
            negative_points_per_component=2,
        )

        self.assertTrue(np.array_equal(probability, original))
        self.assertEqual(len(components), 2)
        self.assertGreater(components[0].score, components[1].score)
        self.assertEqual(int(support.sum()), 36 + 49)
        x0, y0, x1, y1 = components[0].bbox
        self.assertLessEqual(x0, 2)
        self.assertLessEqual(y0, 2)
        self.assertGreaterEqual(x1, 7)
        self.assertGreaterEqual(y1, 7)
        for component in components:
            self.assertGreater(len(component.positive_points), 0)
            self.assertLessEqual(len(component.positive_points), 3)
            for row, col in component.positive_points:
                self.assertEqual(int(component.mask[row, col]), 1)

    def test_component_budget_is_deterministic(self) -> None:
        probability = np.zeros((24, 24), dtype=np.float32)
        probability[1:5, 1:5] = 0.90
        probability[8:12, 8:12] = 0.92
        probability[16:20, 16:20] = 0.94
        _, components = build_probability_components(
            probability,
            threshold=0.85,
            min_component_area=4,
            max_components=2,
        )
        self.assertEqual([round(component.score, 2) for component in components], [0.94, 0.92])
        self.assertEqual([component.component_id for component in components], [0, 1])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_probability_components(np.zeros((2, 2, 2)), threshold=0.5)
        invalid = np.zeros((4, 4), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_probability_components(invalid, threshold=0.5)
        for threshold in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                build_probability_components(np.zeros((4, 4)), threshold=threshold)
        with self.assertRaises(ValueError):
            build_probability_components(
                np.zeros((4, 4)),
                threshold=0.5,
                min_component_area=0,
            )


if __name__ == "__main__":
    unittest.main()
