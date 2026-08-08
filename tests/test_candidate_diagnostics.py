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
    def test_exact_prompt_provenance_round_trips_as_schema_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = np.zeros((3, 8, 8), dtype=np.uint8)
            masks[:, 2:4, 2:4] = 1
            path = root / "candidate_diagnostics" / "exact.npz"
            saved = save_candidate_diagnostics(
                path,
                sam_masks=masks,
                refined_mask=np.zeros((8, 8), dtype=np.uint8),
                final_mask=np.zeros((8, 8), dtype=np.uint8),
                bone_support=None,
                prompt_map=np.zeros((8, 8), dtype=np.float32),
                positive_points=[],
                negative_points=[],
                boxes=[],
                sam_scores=np.asarray([0.7, 0.8, 0.9], dtype=np.float32),
                selection_scores=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                classifier_causal_scores=None,
                component_ids=np.asarray([4, 4, 4]),
                prompt_modes=["point"] * 3,
                proposal_source_ids=["layercam"] * 3,
                cam_levels=[90.0] * 3,
                prompt_ids=["layercam|p90|c4|point"] * 3,
                multimask_indices=[0, 1, 2],
            )
            row = {
                "image_name": "exact.jpeg",
                **saved,
            }
            summary = write_candidate_diagnostics_manifest(
                root,
                [row],
                expected_image_names=["exact.jpeg"],
                split="val",
                image_size=8,
                pseudo_manifest_sha256="pseudo-lock",
                selection_method="test",
                support_clip_kernel=0,
                cam_percentile=90.0,
                cohort="all",
            )
            indexed, validated = validate_candidate_diagnostics_manifest(
                root,
                expected_image_names=["exact.jpeg"],
                split="val",
                expected_pseudo_manifest_sha256="pseudo-lock",
                expected_manifest_sha256=summary["manifest_sha256"],
            )
            self.assertEqual(set(indexed), {"exact"})
            self.assertEqual(validated["manifest_rows"], 1)
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(int(payload["schema_version"][0]), 3)
                self.assertEqual(len(set(payload["prompt_ids"].astype(str))), 1)
                self.assertEqual(payload["multimask_indices"].tolist(), [0, 1, 2])
                self.assertEqual(payload["cam_levels"].tolist(), [90.0] * 3)

    def test_partial_exact_prompt_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "one complete set"):
                save_candidate_diagnostics(
                    Path(directory) / "bad.npz",
                    sam_masks=np.zeros((1, 8, 8), dtype=np.uint8),
                    refined_mask=np.zeros((8, 8), dtype=np.uint8),
                    final_mask=np.zeros((8, 8), dtype=np.uint8),
                    bone_support=None,
                    prompt_map=np.zeros((8, 8), dtype=np.float32),
                    positive_points=[], negative_points=[], boxes=[],
                    sam_scores=[0.0], selection_scores=[0.0],
                    classifier_causal_scores=None, component_ids=[0],
                    prompt_modes=["point"], proposal_source_ids=["layercam"],
                    prompt_ids=["incomplete"],
                )

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
        source = (PROJECT / "evaluate_final_rich_gallery.py").read_text(
            encoding="utf-8"
        )
        annotation_boundary = source.index("# Annotation boundary")
        annotation_decode = source.index("_decode_labelme_polygon_mask(", annotation_boundary)
        self.assertLess(source.index("verify_frozen_test_config("), annotation_boundary)
        self.assertLess(source.index("freeze_path = args.selection_root"), annotation_boundary)
        self.assertLess(
            source.index("candidate_choices_frozen_before_spatial_gt"), annotation_boundary
        )
        self.assertLess(annotation_boundary, annotation_decode)
        self.assertIn('parser.add_argument("--split", choices=("val", "test")', source)


if __name__ == "__main__":
    unittest.main()
