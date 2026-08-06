from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.segmentation_metrics import (
    paired_group_bootstrap_deltas,
    segmentation_metrics,
    summarize_segmentation_rows,
)


class G4SegmentationMetricTests(unittest.TestCase):
    def test_empty_prediction_and_nonempty_zero_overlap_are_distinct(self) -> None:
        target = np.zeros((8, 8), dtype=np.uint8)
        target[1:3, 1:3] = 1
        empty = segmentation_metrics(np.zeros_like(target), target)
        displaced = np.zeros_like(target)
        displaced[5:7, 5:7] = 1
        zero_overlap = segmentation_metrics(displaced, target)
        self.assertTrue(empty["empty_prediction"])
        self.assertTrue(empty["zero_overlap"])
        self.assertFalse(zero_overlap["empty_prediction"])
        self.assertTrue(zero_overlap["zero_overlap"])

    def test_shifted_identical_rectangles_have_symmetric_surface_metrics(self) -> None:
        target = np.zeros((12, 12), dtype=np.uint8)
        target[3:8, 3:8] = 1
        shifted = np.zeros_like(target)
        shifted[3:8, 4:9] = 1
        metrics = segmentation_metrics(shifted, target)
        self.assertAlmostEqual(float(metrics["hd95_px"]), 1.0, places=7)
        self.assertGreater(float(metrics["assd_px"]), 0.0)
        self.assertLessEqual(float(metrics["assd_px"]), 1.0)

    def test_unequal_surfaces_match_monai_symmetric_convention(self) -> None:
        target = np.zeros((12, 12), dtype=np.uint8)
        target[1:5, 1:9] = 1
        prediction = np.zeros_like(target)
        prediction[2:10, 2:4] = 1
        metrics = segmentation_metrics(prediction, target)
        # Independently recorded from MONAI 1.5.1
        # compute_average_surface_distance(..., symmetric=True).
        self.assertAlmostEqual(float(metrics["assd_px"]), 2.26246953, places=6)
        self.assertAlmostEqual(float(metrics["hd95_px"]), 5.00495100, places=6)

    def test_summary_reports_macro_micro_and_extent(self) -> None:
        first = np.zeros((4, 4), dtype=np.uint8)
        first[:2, :2] = 1
        second = np.zeros((4, 4), dtype=np.uint8)
        second[:, :] = 1
        rows = []
        for image_id, pred, target in (
            ("a", first, first),
            ("b", np.zeros_like(second), second),
        ):
            rows.append({"image_id": image_id, "group_id": image_id, **segmentation_metrics(pred, target)})
        summary = summarize_segmentation_rows(rows)
        self.assertAlmostEqual(summary["mean_tumor_dice"], 0.5)
        self.assertAlmostEqual(summary["micro_dice"], 8.0 / 24.0)
        self.assertEqual(summary["tumor_empty_prediction_count"], 1)
        self.assertEqual(summary["tumor_zero_overlap_count"], 1)
        self.assertAlmostEqual(summary["median_tumor_relative_area_difference"], -0.5)

    def test_paired_group_bootstrap_is_deterministic_and_fail_closed(self) -> None:
        reference = [
            {"image_id": "a", "group_id": "g1", "dice": 0.1},
            {"image_id": "b", "group_id": "g2", "dice": 0.2},
        ]
        comparison = [
            {"image_id": "a", "group_id": "g1", "dice": 0.3},
            {"image_id": "b", "group_id": "g2", "dice": 0.4},
        ]
        first = paired_group_bootstrap_deltas(
            reference, comparison, metrics=("dice",), iterations=100, seed=7
        )
        second = paired_group_bootstrap_deltas(
            reference, comparison, metrics=("dice",), iterations=100, seed=7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["intervals"]["dice"]["point_delta"], 0.2)
        with self.assertRaisesRegex(ValueError, "cohorts differ"):
            paired_group_bootstrap_deltas(reference, comparison[:1], metrics=("dice",))


if __name__ == "__main__":
    unittest.main()
