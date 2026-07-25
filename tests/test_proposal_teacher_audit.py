from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_proposal_teacher_under_test",
    TOOLS / "audit_proposal_teacher.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class ProposalTeacherAuditTests(unittest.TestCase):
    def test_identical_candidate_is_not_promoted(self) -> None:
        path = (
            ROOT
            / "artifacts"
            / "kaggle"
            / "wsss_binary_cam_sam_tta_flip_v1"
            / "btxrd_binary_cam_sam_tta_flip"
            / "ground_truth"
            / "evaluation"
            / "per_image.csv"
        )
        rows = AUDIT.read_csv(path)
        paired, decision = AUDIT.recompute_decision(
            rows,
            rows,
            iterations=100,
            seed=42,
        )
        self.assertEqual(decision, "REJECT")
        self.assertEqual(
            paired["overall"]["signed_gap_candidate_minus_reference"],
            0.0,
        )
        self.assertEqual(
            paired["small_lt_1pct"]["paired_group_bootstrap_ci95_low"],
            0.0,
        )

    def test_frozen_metadata_expresses_proposal_only_contract(self) -> None:
        frozen = AUDIT.FROZEN_GENERATION_METADATA
        self.assertEqual(frozen["selection_method"], "coverage_mass_sam")
        self.assertEqual(frozen["support_clip_kernel"], 5)
        self.assertEqual(frozen["proposal_teacher_threshold"], 0.85)
        self.assertEqual(
            frozen["proposal_teacher_semantics"],
            "proposal_components_only; CAM scoring and support clipping unchanged",
        )


if __name__ == "__main__":
    unittest.main()
