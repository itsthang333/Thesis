from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "project" / "tools" / "audit_wsl_gt_pair.py"
    spec = importlib.util.spec_from_file_location("audit_wsl_gt_pair_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module()


class WslGtPairAuditTests(unittest.TestCase):
    def test_lesion_size_boundaries_are_frozen(self) -> None:
        self.assertEqual(AUDIT.lesion_size(0.0), "small_lt_1pct")
        self.assertEqual(AUDIT.lesion_size(0.009999), "small_lt_1pct")
        self.assertEqual(AUDIT.lesion_size(0.01), "medium_1_to_5pct")
        self.assertEqual(AUDIT.lesion_size(0.049999), "medium_1_to_5pct")
        self.assertEqual(AUDIT.lesion_size(0.05), "large_ge_5pct")

    def test_paired_bootstrap_uses_candidate_minus_reference_gap(self) -> None:
        reference = [
            {
                "image_name": "a.jpeg",
                "group_id": "g1",
                "gt_positive": "True",
                "gt_area_ratio": "0.005",
                "dice": "0.60",
            },
            {
                "image_name": "b.jpeg",
                "group_id": "g2",
                "gt_positive": "True",
                "gt_area_ratio": "0.02",
                "dice": "0.70",
            },
            {
                "image_name": "c.jpeg",
                "group_id": "g3",
                "gt_positive": "True",
                "gt_area_ratio": "0.08",
                "dice": "0.80",
            },
        ]
        candidate = [
            {**reference[0], "dice": "0.56"},
            {**reference[1], "dice": "0.64"},
            {**reference[2], "dice": "0.76"},
        ]
        result = AUDIT.paired_bootstrap(reference, candidate, iterations=100, seed=42)
        self.assertAlmostEqual(result["small_lt_1pct"]["absolute_gap"], 0.04)
        self.assertTrue(result["small_lt_1pct"]["criterion_abs_gap_le_0_05"])
        self.assertAlmostEqual(
            result["medium_1_to_5pct"]["signed_gap_candidate_minus_reference"],
            -0.06,
        )
        self.assertFalse(result["medium_1_to_5pct"]["criterion_abs_gap_le_0_05"])
        self.assertAlmostEqual(result["large_ge_5pct"]["absolute_gap"], 0.04)

    def test_frozen_gt_reference_lock_passes(self) -> None:
        lock = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "reference"
            / "gt_resnet18_unet_448_v1"
            / "reference_lock.json"
        )
        result = AUDIT.verify_reference_lock(lock)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["population"]["size_counts"]["small_lt_1pct"], 94)

    def test_paired_protocol_matches_reference_and_historical_audit(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "reference"
            / "gt_resnet18_unet_448_v1"
        )
        lock = json.loads((root / "reference_lock.json").read_text(encoding="utf-8"))
        protocol = json.loads(
            (root / "paired_protocol_v1.json").read_text(encoding="utf-8")
        )
        baseline = json.loads(
            (root / "baseline_wsl_pair_audit.json").read_text(encoding="utf-8")
        )
        self.assertFalse(protocol["consumer_invariants"]["test_evaluated"])
        self.assertEqual(
            protocol["consumer_invariants"]["architecture"],
            lock["consumer_training_contract"]["architecture"],
        )
        for subgroup, values in protocol["subgroup_contract"].items():
            reference = float(values["reference_mean_dice"])
            lower, upper = map(float, values["success_interval_inclusive"])
            self.assertAlmostEqual(lower, reference - 0.05)
            self.assertAlmostEqual(upper, reference + 0.05)
            self.assertAlmostEqual(
                protocol["historical_wsl_baseline"]["absolute_gap"][subgroup],
                baseline["paired_gap"][subgroup]["absolute_gap"],
            )
        self.assertFalse(baseline["primary_success"])


if __name__ == "__main__":
    unittest.main()
