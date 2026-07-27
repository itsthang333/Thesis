from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pseudo.candidate_diagnostics import (
    save_candidate_diagnostics,
    validate_candidate_diagnostics_manifest,
    write_candidate_diagnostics_manifest,
)


def _save(root: Path, name: str, candidate_count: int) -> dict[str, object]:
    shape = (8, 8)
    masks = np.zeros((candidate_count, *shape), dtype=np.uint8)
    if candidate_count:
        masks[0, 2:4, 2:4] = 1
    row = save_candidate_diagnostics(
        root / "candidate_diagnostics" / f"{Path(name).stem}.npz",
        sam_masks=masks,
        refined_mask=np.zeros(shape, dtype=np.uint8),
        final_mask=np.zeros(shape, dtype=np.uint8),
        bone_support=None,
        prompt_map=np.zeros(shape, dtype=np.float32),
        positive_points=[],
        negative_points=[],
        boxes=[],
        sam_scores=np.zeros(candidate_count, dtype=np.float32),
        selection_scores=np.zeros(candidate_count, dtype=np.float32),
        classifier_causal_scores=None,
        component_ids=None,
        prompt_modes=["point"] * candidate_count,
    )
    return {
        "image_name": name,
        "tumor_type": "tumor",
        "generation_status": "ok" if candidate_count else "empty_by_image_gate",
        **row,
    }


class CandidateDiagnosticsTests(unittest.TestCase):
    def test_prediction_first_manifest_includes_empty_complete_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_save(root, "a.png", 1), _save(root, "b.png", 0)]
            summary = write_candidate_diagnostics_manifest(
                root,
                rows,
                expected_image_names=["a.png", "b.png"],
                split="val",
                image_size=8,
                pseudo_manifest_sha256="a" * 64,
                selection_method="coverage_mass_sam",
                support_clip_kernel=5,
                cam_percentile=85.0,
            )
            indexed, audited = validate_candidate_diagnostics_manifest(
                root,
                expected_image_names=["a.png", "b.png"],
                split="val",
                expected_pseudo_manifest_sha256="a" * 64,
                expected_manifest_sha256=str(summary["manifest_sha256"]),
            )
            self.assertEqual(set(indexed), {"a", "b"})
            self.assertIs(audited["ground_truth_loaded_during_generation"], False)
            self.assertEqual(int(indexed["b"]["candidate_count"]), 0)

    def test_candidate_manifest_rejects_missing_tumor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "complete tumor cohort"):
                write_candidate_diagnostics_manifest(
                    root,
                    [_save(root, "a.png", 1)],
                    expected_image_names=["a.png", "b.png"],
                    split="val",
                    image_size=8,
                    pseudo_manifest_sha256="a" * 64,
                    selection_method="coverage_mass_sam",
                    support_clip_kernel=5,
                    cam_percentile=85.0,
                )

    def test_candidate_manifest_rejects_npz_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_save(root, "a.png", 1)]
            summary = write_candidate_diagnostics_manifest(
                root,
                rows,
                expected_image_names=["a.png"],
                split="val",
                image_size=8,
                pseudo_manifest_sha256="a" * 64,
                selection_method="coverage_mass_sam",
                support_clip_kernel=5,
                cam_percentile=85.0,
            )
            with (root / "candidate_diagnostics" / "a.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "file/hash mismatch"):
                validate_candidate_diagnostics_manifest(
                    root,
                    expected_image_names=["a.png"],
                    split="val",
                    expected_manifest_sha256=str(summary["manifest_sha256"]),
                )

    def test_complete_image_cohort_is_explicit_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_save(root, "positive.png", 2), _save(root, "normal.png", 1)]
            summary = write_candidate_diagnostics_manifest(
                root,
                rows,
                expected_image_names=["positive.png", "normal.png"],
                split="train",
                image_size=8,
                pseudo_manifest_sha256="b" * 64,
                selection_method="mask_bag_mil_input",
                support_clip_kernel=-1,
                cam_percentile=90.0,
                cohort="all",
            )
            self.assertEqual(summary["cohort"], "all")
            self.assertEqual(summary["expected_images"], 2)
            self.assertIsNone(summary["expected_tumor_images"])
            _rows, audit = validate_candidate_diagnostics_manifest(
                root,
                expected_image_names=["positive.png", "normal.png"],
                split="train",
                expected_manifest_sha256=str(summary["manifest_sha256"]),
            )
            self.assertEqual(audit["cohort"], "all")

    def test_generator_has_no_segmentation_gt_loader(self) -> None:
        source = (PROJECT / "generate_pseudo_masks.py").read_text(encoding="utf-8")
        self.assertNotIn("build_segmentation_dataset", source)
        self.assertIn("--save-candidate-diagnostics", source)
        self.assertIn("--candidate-diagnostics-cohort", source)
        self.assertIn("--force-normal-candidate-gallery", source)

    def test_evaluator_verifies_frozen_artifacts_before_gt_dataset(self) -> None:
        source = (PROJECT / "evaluate_saved_candidate_diagnostics.py").read_text(
            encoding="utf-8"
        )
        gt_loader = source.index("segmentation_dataset = build_segmentation_dataset(")
        self.assertLess(source.index("validate_pseudo_mask_manifest("), gt_loader)
        self.assertLess(
            source.index("validate_candidate_diagnostics_manifest("), gt_loader
        )
        self.assertIn('parser.add_argument("--split", choices=["val"]', source)


if __name__ == "__main__":
    unittest.main()
