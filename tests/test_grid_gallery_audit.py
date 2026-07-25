from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_grid_gallery_under_test",
    TOOLS / "audit_grid_gallery.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class GridGalleryAuditTests(unittest.TestCase):
    def test_frozen_full_grid_result_is_independently_rejected(self) -> None:
        candidate = (
            ROOT
            / "artifacts"
            / "kaggle"
            / "pro2sam_grid_full_v1"
            / "btxrd_pro2sam_grid_gallery_full_v1"
        )
        result = AUDIT.audit_grid_gallery(
            candidate,
            ROOT
            / "artifacts"
            / "best_pipeline"
            / "fs_resnet18_pw10_full_448_e20"
            / "data"
            / "split_manifest.csv",
            ROOT
            / "artifacts"
            / "kaggle"
            / "wsss_binary_cam_sam_tta_flip_v1"
            / "btxrd_binary_cam_sam_tta_flip"
            / "ground_truth"
            / "evaluation"
            / "per_image.csv",
            ROOT
            / "artifacts"
            / "kaggle"
            / "s2c_cpm_gate_c_v1"
            / "btxrd_s2c_cpm_gate_c_v1"
            / "ground_truth"
            / "evaluation"
            / "per_image.csv",
            ROOT
            / "artifacts"
            / "kaggle"
            / "s2c_cpm_gate_c_v1"
            / "btxrd_s2c_cpm_gate_c_v1"
            / "ground_truth"
            / "pseudo_masks"
            / "prompt_quality.csv",
            iterations=100,
            seed=42,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["decision"], "REJECT")
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["population"]["size_counts"], {
            "small_lt_1pct": 94,
            "medium_1_to_5pct": 72,
            "large_ge_5pct": 18,
        })
        self.assertEqual(
            result["protocol"]["max_direct_cam_prompt_metric_abs_delta_vs_cpm"],
            0.0,
        )
        self.assertAlmostEqual(
            result["overall_candidate_mechanism"]["oracle_best_single_dice"],
            0.11938725160133586,
        )
        self.assertFalse(result["promotion_rule_recomputed"])


if __name__ == "__main__":
    unittest.main()
