from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_gt_reproduction_under_test",
    TOOLS / "audit_gt_reproduction.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class GtReproductionAuditTests(unittest.TestCase):
    def test_source_hash_normalization_only_removes_project_prefix(self) -> None:
        normalized = AUDIT.normalize_source_hashes(
            {
                "project/train_segmentation.py": "abc",
                "models/unet.py": "def",
            }
        )
        self.assertEqual(
            normalized,
            {
                "train_segmentation.py": "abc",
                "models/unet.py": "def",
            },
        )

    def test_source_hash_normalization_rejects_duplicate_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
            AUDIT.normalize_source_hashes(
                {
                    "project/models/unet.py": "abc",
                    "models/unet.py": "def",
                }
            )

    def test_reference_training_log_reselects_epoch_20(self) -> None:
        path = (
            ROOT
            / "artifacts"
            / "best_pipeline"
            / "fs_resnet18_pw10_full_448_e20"
            / "training"
            / "training_log.csv"
        )
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = AUDIT.select_best_checkpoint(rows)
        self.assertEqual(selected["best_epoch"], 20)
        self.assertEqual(selected["last_completed_epoch"], 30)
        self.assertAlmostEqual(
            selected["best_val_positive_dice_at_0_5"],
            0.49017143767812976,
        )

    def test_noncontiguous_training_log_fails_closed(self) -> None:
        rows = [
            {
                "epoch": "1",
                "val_positive_dice": "0.2",
                "val_empty_specificity": "0.5",
            },
            {
                "epoch": "3",
                "val_positive_dice": "0.3",
                "val_empty_specificity": "0.6",
            },
        ]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            AUDIT.select_best_checkpoint(rows)

    def test_tie_breaker_prefers_specificity_within_tolerance(self) -> None:
        rows = [
            {
                "epoch": "1",
                "val_positive_dice": "0.50000",
                "val_empty_specificity": "0.6",
            },
            {
                "epoch": "2",
                "val_positive_dice": "0.50005",
                "val_empty_specificity": "0.7",
            },
        ]
        selected = AUDIT.select_best_checkpoint(rows)
        self.assertEqual(selected["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
