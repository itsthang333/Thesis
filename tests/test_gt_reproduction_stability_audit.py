from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_gt_reproduction_stability_under_test",
    TOOLS / "audit_gt_reproduction_stability.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class GtReproductionStabilityAuditTests(unittest.TestCase):
    def test_pearson_identical_vectors_is_one(self) -> None:
        self.assertTrue(
            math.isclose(
                AUDIT._pearson([0.1, 0.4, 0.9], [0.1, 0.4, 0.9]),
                1.0,
                abs_tol=1e-12,
            )
        )

    def test_identical_per_image_file_has_zero_repeat_delta(self) -> None:
        path = (
            ROOT
            / "artifacts"
            / "kaggle"
            / "gt_reference_independent_reproduction_v2"
            / "btxrd_gt_reference_independent_reproduction_v2"
            / "evaluation"
            / "selected_per_image.csv"
        )
        result = AUDIT.per_image_stability(path, path)
        for subgroup in ("overall", *AUDIT.SIZE_ORDER):
            self.assertEqual(
                result[subgroup]["mean_absolute_per_image_delta"],
                0.0,
            )
            self.assertTrue(
                math.isclose(
                    result[subgroup]["pearson_per_image_dice"],
                    1.0,
                    abs_tol=1e-12,
                )
            )

    def test_reference_and_v2_training_diverge_at_epoch_one(self) -> None:
        reference = (
            ROOT
            / "artifacts"
            / "best_pipeline"
            / "fs_resnet18_pw10_full_448_e20"
            / "training"
            / "training_log.csv"
        )
        v2 = (
            ROOT
            / "artifacts"
            / "kaggle"
            / "gt_reference_independent_reproduction_v2"
            / "btxrd_gt_reference_independent_reproduction_v2"
            / "fs_resnet18_pw10_full_448_seed42"
            / "training_log.csv"
        )
        result = AUDIT.compare_training_logs(reference, v2)
        self.assertEqual(result["exact_equal_prefix_epochs"], 0)
        self.assertEqual(result["first_numeric_divergence_epoch"], 1)

    def test_reference_reproducibility_evidence_is_explicitly_limited(self) -> None:
        evidence_root = (
            ROOT
            / "artifacts"
            / "reference"
            / "gt_resnet18_unet_448_v1"
        )
        determinism = json.loads(
            (evidence_root / "reproducibility_static_audit.json").read_text(
                encoding="utf-8"
            )
        )
        pretrained = json.loads(
            (evidence_root / "pretrained_weight_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(determinism["status"], "PASS_WITH_LIMITATIONS")
        self.assertFalse(
            determinism["limitations"][
                "bitwise_checkpoint_reproduction_guaranteed"
            ]
        )
        self.assertEqual(
            pretrained["sha256"],
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        )
        self.assertEqual(
            pretrained["status"],
            "PASS_WITH_PROVENANCE_LIMITATION",
        )


if __name__ == "__main__":
    unittest.main()
