from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "models" / "mask_bag_score_evidence.py"
SPEC = importlib.util.spec_from_file_location("mask_bag_score_evidence", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScoreEvidenceTests(unittest.TestCase):
    def _write_valid(self, root: Path) -> tuple[dict[str, object], dict[str, object]]:
        score_path = root / "scores" / "image_a.npz"
        saved = MODULE.save_candidate_score_evidence(
            score_path,
            candidate_indices=[1, 4, 8],
            candidate_logits=[0.2, 0.9, 0.1],
        )
        row = {
            "image_id": "image_a.jpeg",
            "group_id": "group_a",
            "tumor": "1",
            "candidate_payload_sha256": "a" * 64,
            **saved,
            "score_path": "scores/image_a.npz",
        }
        summary = MODULE.write_candidate_score_manifest(root, [row])
        expected = {
            "image_a.jpeg": {
                "group_id": "group_a",
                "tumor": "1",
                "candidate_payload_sha256": "a" * 64,
                "candidate_count": 3,
            }
        }
        return summary, expected

    def test_round_trip_binds_all_scores_to_candidate_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, expected = self._write_valid(root)
            rows = MODULE.validate_candidate_score_manifest(
                root,
                expected_manifest_sha256=summary["manifest_sha256"],
                expected_images=expected,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["selected_candidate_index"], "4")
            self.assertEqual(rows[0]["candidate_count"], "3")

    def test_tied_logits_select_first_gallery_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved = MODULE.save_candidate_score_evidence(
                Path(directory) / "scores.npz",
                candidate_indices=[2, 5],
                candidate_logits=[0.7, 0.7],
            )
            self.assertEqual(saved["selected_candidate_index"], 2)

    def test_non_gallery_order_and_nonfinite_logits_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            with self.assertRaises(ValueError):
                MODULE.save_candidate_score_evidence(
                    path, candidate_indices=[2, 1], candidate_logits=[0.1, 0.2]
                )
            with self.assertRaises(ValueError):
                MODULE.save_candidate_score_evidence(
                    path, candidate_indices=[1, 2], candidate_logits=[0.1, np.nan]
                )

    def test_payload_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, expected = self._write_valid(root)
            with (root / "scores" / "image_a.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                MODULE.validate_candidate_score_manifest(
                    root,
                    expected_manifest_sha256=summary["manifest_sha256"],
                    expected_images=expected,
                )

    def test_source_has_no_annotation_or_training_dependency(self) -> None:
        source = SOURCE.read_text(encoding="utf-8").lower()
        self.assertNotIn("datasets.factory", source)
        self.assertNotIn("build_segmentation_dataset", source)
        self.assertNotIn("mask_tensor", source)
        self.assertNotIn("annotation_name", source)
        self.assertNotIn("torch", source)
        self.assertNotIn("optimizer", source)
