from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from summarize_g4_e2_candidate_decomposition import EXPECTED_ARMS, summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateDecompositionTest(unittest.TestCase):
    def test_complete_factorial_is_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, arm in enumerate(EXPECTED_ARMS):
                arm_dir = root / arm
                arm_dir.mkdir()
                selected_rows = []
                cam_rows = []
                for image_index in range(371):
                    tumor = image_index < 184
                    group = (
                        "small_lt_1pct" if image_index < 94 else
                        "medium_1_to_5pct" if image_index < 166 else
                        "large_ge_5pct" if tumor else "normal"
                    )
                    common = {
                        "image_id": f"img{image_index:04d}.png",
                        "group_id": f"g{image_index // 2}",
                        "gt_positive": str(tumor),
                        "native_size_group": group,
                    }
                    selected_rows.append({
                        **common,
                        "dice": 0.2 + index / 1000 if tumor else 1.0,
                        "candidate_oracle_dice": 0.5 if tumor else 0.0,
                    })
                    cam_rows.append({**common, "dice": 0.1 if tumor else 1.0})
                for name, rows in (("per_image.csv", selected_rows), ("cam_only_per_image.csv", cam_rows)):
                    path = arm_dir / name
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                summary = {
                    "candidate_analysis_enabled": True,
                    "images": 371,
                    "tumor_images": 184,
                    "validation_annotations_opened": 184,
                    "test_images_read": 0,
                    "test_evaluated": False,
                }
                (arm_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                audit = {
                    "pass": True,
                    "per_image_sha256": sha256(arm_dir / "per_image.csv"),
                    "cam_only_per_image_sha256": sha256(arm_dir / "cam_only_per_image.csv"),
                }
                (arm_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")

            result = summarize([root])
            self.assertEqual(result["best_selected_arm"], EXPECTED_ARMS[-1])
            self.assertAlmostEqual(result["arms"][EXPECTED_ARMS[0]]["proposal_oracle_dice"], 0.5)
            self.assertAlmostEqual(result["arms"][EXPECTED_ARMS[0]]["selector_regret"], 0.3)
            self.assertFalse(result["test_evaluated"])

    def test_missing_arm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize([Path(tmp)])


if __name__ == "__main__":
    unittest.main()
