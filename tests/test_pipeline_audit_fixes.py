from __future__ import annotations

import sys
import tempfile
import types
import unittest
import importlib.util
import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mask_selection = load_module("mask_selection_under_test", PROJECT_ROOT / "pseudo" / "mask_selection.py")
tumor_morphology = load_module("tumor_morphology_under_test", PROJECT_ROOT / "pseudo" / "tumor_morphology.py")
extract_prompts = load_module("extract_prompts_under_test", PROJECT_ROOT / "pseudo" / "extract_prompts.py")
mask_manifest = load_module("mask_manifest_under_test", PROJECT_ROOT / "pseudo" / "manifest.py")
segmentation_metrics = load_module(
    "segmentation_metrics_under_test", PROJECT_ROOT / "evaluation" / "segmentation_metrics.py"
)
classification_metrics = load_module(
    "classification_metrics_under_test", PROJECT_ROOT / "evaluation" / "classification_metrics.py"
)
frozen_test_guard = load_module(
    "frozen_test_guard_under_test", PROJECT_ROOT / "evaluation" / "frozen_test_guard.py"
)
sam_refine = load_module("sam_refine_under_test", PROJECT_ROOT / "pseudo" / "sam_refine.py")
select_and_fuse_masks = mask_selection.select_and_fuse_masks
build_class_conditioned_components = tumor_morphology.build_class_conditioned_components


def load_btxrd_module():
    torch_stub = types.ModuleType("torch")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_data_stub = types.ModuleType("torch.utils.data")
    torch_data_stub.Dataset = object
    torch_utils_stub.data = torch_data_stub
    torch_stub.utils = torch_utils_stub
    datasets_stub = types.ModuleType("datasets")
    common_stub = types.ModuleType("datasets.common")
    common_stub.IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
    common_stub.apply_clahe = lambda image: image
    common_stub.make_classification_transform = lambda *args, **kwargs: None
    common_stub.make_segmentation_image_transform = lambda *args, **kwargs: None
    common_stub.make_segmentation_mask_transform = lambda *args, **kwargs: None
    stubs = {
        "torch": torch_stub,
        "torch.utils": torch_utils_stub,
        "torch.utils.data": torch_data_stub,
        "datasets": datasets_stub,
        "datasets.common": common_stub,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        return load_module("btxrd_under_test", PROJECT_ROOT / "datasets" / "btxrd.py")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


btxrd = load_btxrd_module()


class MaskSelectionTests(unittest.TestCase):
    def test_rejects_all_candidates_below_threshold_by_default(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[1, 2:, 2:] = 1
        cam = np.full((4, 4), 0.1, dtype=np.float32)

        result = select_and_fuse_masks(
            masks,
            cam,
            mask_score_threshold=0.4,
            selection_method="mean",
        )

        self.assertFalse(result.any())

    def test_keep_best_policy_is_explicit_debug_opt_in(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[1, 2:, 2:] = 1
        cam = np.full((4, 4), 0.1, dtype=np.float32)

        result = select_and_fuse_masks(
            masks,
            cam,
            mask_score_threshold=0.4,
            selection_method="mean",
            low_score_policy="keep-best",
        )

        self.assertTrue(np.array_equal(result, masks[1]))

    def test_coverage_mass_sam_applies_support_clipping(self) -> None:
        masks = np.ones((1, 5, 5), dtype=np.uint8)
        cam = np.ones((5, 5), dtype=np.float32)
        support = np.zeros((5, 5), dtype=np.uint8)
        support[2, 2] = 1

        result = select_and_fuse_masks(
            masks,
            cam,
            mask_score_threshold=0.0,
            selection_method="coverage_mass_sam",
            bone_support=support,
            sam_scores=np.array([0.9], dtype=np.float32),
            support_clip_kernel=1,
        )

        self.assertTrue(np.array_equal(result, support))

    def test_support_clipping_fails_closed_when_candidate_is_off_support(self) -> None:
        masks = np.zeros((1, 5, 5), dtype=np.uint8)
        masks[0, 0, 0] = 1
        cam = np.zeros((5, 5), dtype=np.float32)
        cam[0, 0] = 1.0
        support = np.zeros((5, 5), dtype=np.uint8)
        support[4, 4] = 1

        result = select_and_fuse_masks(
            masks,
            cam,
            mask_score_threshold=0.0,
            selection_method="coverage_mass_sam",
            bone_support=support,
            sam_scores=np.array([0.9], dtype=np.float32),
            support_clip_kernel=1,
        )

        self.assertFalse(result.any())

    def test_selection_details_distinguish_threshold_candidates_and_components(self) -> None:
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[1, :2, :2] = 1
        masks[2, 2:, 2:] = 1
        cam = masks.any(axis=0).astype(np.float32)
        _, details = select_and_fuse_masks(
            masks,
            cam,
            mask_score_threshold=0.0,
            component_ids=np.array([0, 0, 1]),
            best_per_component=True,
            component_topk=2,
            return_details=True,
        )
        self.assertEqual(details["above_threshold_candidates"], 3)
        self.assertEqual(details["selected_candidates"], 2)
        self.assertEqual(details["selected_components"], 2)


class SAMPromptTests(unittest.TestCase):
    def test_pure_box_ignores_negative_points_and_passes_no_point_arrays(self) -> None:
        class FakePredictor:
            def set_image(self, image):
                self.shape = image.shape[:2]

            def predict(self, *, point_coords, point_labels, box, multimask_output):
                self.point_coords = point_coords
                self.point_labels = point_labels
                self.box = box
                return (
                    np.zeros((1, *self.shape), dtype=bool),
                    np.array([0.9], dtype=np.float32),
                    None,
                )

        wrapper = sam_refine.SAMPredictor.__new__(sam_refine.SAMPredictor)
        wrapper._predictor = FakePredictor()
        component = types.SimpleNamespace(
            component_id=7,
            bbox=(1, 1, 5, 5),
            mask=np.ones((8, 8), dtype=np.uint8),
            positive_points=((3, 3),),
            negative_points=((1, 1), (5, 5)),
        )
        wrapper.predict_from_components(
            np.zeros((8, 8, 3), dtype=np.uint8),
            [component],
            prompt_mode="box",
            negative_points_per_component=2,
        )
        self.assertIsNone(wrapper._predictor.point_coords)
        self.assertIsNone(wrapper._predictor.point_labels)
        self.assertIsNotNone(wrapper._predictor.box)
        self.assertEqual(wrapper.last_prompt_stats["unique_negative_prompt_points"], 0)


class TumorMorphologyTests(unittest.TestCase):
    def test_negative_points_are_outside_positive_component(self) -> None:
        component = np.zeros((12, 12), dtype=np.uint8)
        component[4:8, 4:8] = 1
        cam = np.random.default_rng(0).random((12, 12), dtype=np.float32)
        cam[component.astype(bool)] = 1.0
        points = tumor_morphology._select_negative_points(
            component, cam, positive_points=((5, 5),), max_points=4
        )
        self.assertTrue(points)
        self.assertTrue(all(component[row, col] == 0 for row, col in points))

    def test_constant_zero_cam_produces_empty_support(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        cam = np.zeros((16, 16), dtype=np.float32)

        likelihood, support, components = build_class_conditioned_components(
            image,
            [cam],
            [1.0],
            min_component_area=1,
        )

        self.assertFalse(likelihood.any())
        self.assertFalse(support.any())
        self.assertEqual(components, [])

    def test_flat_cam_does_not_create_corner_prompt(self) -> None:
        points = extract_prompts.extract_point_prompts(
            np.zeros((16, 16), dtype=np.float32), min_component_area=1
        )
        self.assertEqual(points, [])

    def test_legacy_tumor_guidance_also_fails_closed_on_flat_cam(self) -> None:
        likelihood, support = tumor_morphology.build_tumor_guidance(
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16), dtype=np.float32),
            min_component_area=1,
        )
        self.assertFalse(likelihood.any())
        self.assertFalse(support.any())

    def test_non_finite_cam_does_not_create_prompt(self) -> None:
        cam = np.zeros((16, 16), dtype=np.float32)
        cam[4, 4] = np.nan
        self.assertEqual(extract_prompts.extract_point_prompts(cam, min_component_area=1), [])


class ReportingMetricTests(unittest.TestCase):
    def test_classifier_auc_metrics_are_tie_invariant(self) -> None:
        target = np.array([1, 0, 1, 0], dtype=np.uint8)
        score = np.array([0.8, 0.8, 0.2, 0.2], dtype=np.float64)
        permutation = np.array([1, 0, 3, 2])

        self.assertAlmostEqual(classification_metrics.binary_auroc(target, score), 0.5)
        self.assertAlmostEqual(classification_metrics.binary_average_precision(target, score), 0.5)
        self.assertAlmostEqual(
            classification_metrics.binary_average_precision(target, score),
            classification_metrics.binary_average_precision(target[permutation], score[permutation]),
        )

    def test_binary_metric_math_on_small_mask(self) -> None:
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        metrics = segmentation_metrics.segmentation_metrics(pred, target)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["pixel_specificity"], 0.5)

    def test_normal_cases_are_not_mixed_into_main_tumor_dice(self) -> None:
        tumor = segmentation_metrics.segmentation_metrics(
            np.zeros((2, 2), dtype=np.uint8), np.array([[1, 0], [0, 0]], dtype=np.uint8)
        )
        normal = segmentation_metrics.segmentation_metrics(
            np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)
        )
        summary = segmentation_metrics.summarize_segmentation_rows([tumor, normal])
        self.assertEqual(summary["mean_tumor_dice"], 0.0)
        self.assertEqual(summary["normal_empty_prediction_rate"], 1.0)

    def test_boundary_means_are_explicitly_conditional_and_misses_are_counted(self) -> None:
        if segmentation_metrics.ndimage is None:
            self.skipTest("SciPy is not installed in the lightweight audit environment")
        hit = segmentation_metrics.segmentation_metrics(
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
        )
        miss = segmentation_metrics.segmentation_metrics(
            np.zeros((2, 2), dtype=np.uint8),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
        )
        summary = segmentation_metrics.summarize_segmentation_rows([hit, miss])
        self.assertIn("conditional", summary["boundary_metric_definition"])
        self.assertEqual(summary["boundary_metric_eligible_tumor_images"], 1)
        self.assertEqual(summary["boundary_metric_complete_misses"], 1)
        self.assertNotIn("mean_tumor_hd95_px", summary)

    def test_one_to_one_lesion_matching_does_not_double_credit_merged_prediction(self) -> None:
        if segmentation_metrics.ndimage is None:
            self.skipTest("SciPy is not installed in the lightweight audit environment")
        target = np.zeros((7, 7), dtype=np.uint8)
        target[1:3, 1:3] = 1
        target[1:3, 4:6] = 1
        prediction = np.zeros_like(target)
        prediction[1:3, 1:6] = 1
        metrics = segmentation_metrics.segmentation_metrics(prediction, target)
        self.assertEqual(metrics["detected_lesions_any_overlap"], 2)
        self.assertEqual(metrics["lesion_tp_one_to_one_iou10"], 1)
        self.assertEqual(metrics["lesion_tp_one_to_one_iou25"], 1)
        self.assertEqual(metrics["lesion_tp_one_to_one_iou50"], 0)
        summary = segmentation_metrics.summarize_segmentation_rows([metrics])
        self.assertIn("0.10, 0.25, 0.50", summary["lesion_matching_definition"])

    def test_classifier_group_bootstrap_reports_required_intervals(self) -> None:
        rows = [
            {"group_id": "a", "true_class": 0, "predicted_class": 0, "true_tumor": 0,
             "predicted_tumor": 0, "tumor_probability": 0.1},
            {"group_id": "b", "true_class": 1, "predicted_class": 1, "true_tumor": 1,
             "predicted_tumor": 1, "tumor_probability": 0.9},
            {"group_id": "c", "true_class": 1, "predicted_class": 0, "true_tumor": 1,
             "predicted_tumor": 0, "tumor_probability": 0.4},
        ]
        result = classification_metrics.classifier_group_bootstrap_confidence_intervals(
            rows, num_classes=2, iterations=20, seed=3
        )
        self.assertEqual(
            set(result["intervals"]),
            {"macro_f1", "tumor_gate_auroc", "tumor_gate_auprc",
             "tumor_gate_sensitivity", "tumor_gate_specificity"},
        )
        self.assertIn("not verified patient", result["group_provenance"])


class FrozenTestGuardTests(unittest.TestCase):
    def test_test_split_requires_frozen_config_before_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen-config"):
            frozen_test_guard.verify_frozen_test_config(None, split="test")

    def test_validation_split_does_not_require_frozen_config(self) -> None:
        self.assertIsNone(frozen_test_guard.verify_frozen_test_config(None, split="val"))

    def test_v4_guard_is_portable_and_locks_stage_threshold_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "split_manifest.csv"
            checkpoint = root / "best_unet.pt"
            manifest.write_bytes(b"locked split\n")
            checkpoint.write_bytes(b"locked checkpoint\n")
            source_path = PROJECT_ROOT / "evaluation" / "frozen_test_guard.py"
            document = {
                "schema_version": 4,
                "status": "final",
                "source": {
                    "files": [{
                        "path": "project/evaluation/frozen_test_guard.py",
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "bytes": source_path.stat().st_size,
                    }],
                },
                "split_manifest": {
                    "path_hint": manifest.name,
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "bytes": manifest.stat().st_size,
                },
                "unet_checkpoint": {
                    "path_hint": checkpoint.name,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "bytes": checkpoint.stat().st_size,
                },
                "evaluation": {
                    "threshold": 0.85,
                    "image_size": 448,
                    "threshold_selection_partition": "val",
                    "threshold_sweep_forbidden": True,
                },
                "allowed_test_stages": ["official_wsss_segmenter"],
            }
            document["freeze_sha256"] = hashlib.sha256(
                frozen_test_guard._canonical_bytes(document)
            ).hexdigest()
            config_path = root / "frozen.json"
            config_path.write_text(json.dumps(document), encoding="utf-8")

            verified = frozen_test_guard.verify_frozen_test_config(
                config_path,
                split="test",
                split_manifest=manifest,
                requested_checkpoint=checkpoint,
                checkpoint_any_of=("unet_checkpoint",),
                requested_threshold=0.85,
                requested_image_size=448,
                requested_stage="official_wsss_segmenter",
            )
            self.assertEqual(verified["schema_version"], 4)
            with self.assertRaisesRegex(ValueError, "threshold"):
                frozen_test_guard.verify_frozen_test_config(
                    config_path,
                    split="test",
                    split_manifest=manifest,
                    requested_checkpoint=checkpoint,
                    checkpoint_any_of=("unet_checkpoint",),
                    requested_threshold=0.5,
                    requested_image_size=448,
                    requested_stage="official_wsss_segmenter",
                )
            with self.assertRaisesRegex(ValueError, "permits only"):
                frozen_test_guard.verify_frozen_test_config(
                    config_path,
                    split="test",
                    requested_stage="pseudo_masks",
                )


class PseudoManifestTests(unittest.TestCase):
    def test_manifest_detects_mask_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "masks").mkdir()
            (output / "run_metadata.json").write_text('{"protocol":"ground_truth"}\n', encoding="utf-8")
            Image.new("L", (8, 8), 0).save(output / "masks" / "sample.png")
            mask_manifest.write_pseudo_mask_manifest(
                output,
                [{
                    "image_name": "sample.jpeg", "true_tumor": 0, "status": "empty_by_image_gate",
                    "above_threshold_candidates": 0, "selected_candidates": 0,
                    "selected_components": 0, "sam_prompt_calls": 0, "unique_prompt_points": 0,
                }],
                expected_image_names=["sample.jpeg"],
                split="train",
                image_size=8,
                run_metadata_sha256=mask_manifest.sha256_file(output / "run_metadata.json"),
            )
            info = mask_manifest.validate_pseudo_mask_manifest(
                output / "masks", [{"image_id": "sample.jpeg", "tumor": 0}], split="train", image_size=8
            )
            self.assertTrue(info["complete"])
            Image.new("L", (8, 8), 255).save(output / "masks" / "sample.png")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                mask_manifest.validate_pseudo_mask_manifest(
                    output / "masks", [{"image_id": "sample.jpeg", "tumor": 0}], split="train", image_size=8
                )

    def test_non_finite_cam_produces_empty_support(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        cam = np.zeros((16, 16), dtype=np.float32)
        cam[0, 0] = np.nan

        likelihood, support, components = build_class_conditioned_components(
            image,
            [cam],
            [1.0],
            min_component_area=1,
        )

        self.assertTrue(np.isfinite(likelihood).all())
        self.assertFalse(support.any())
        self.assertEqual(components, [])


class BTXRDDatasetValidationTests(unittest.TestCase):
    @staticmethod
    def normal_row(image_id: str = "normal.png") -> dict[str, str]:
        row = {
            "image_id": image_id,
            "tumor": "0",
            "benign": "0",
            "malignant": "0",
        }
        row.update({column: "0" for column in btxrd.TUMOR_TYPE_COLUMNS})
        return row

    @staticmethod
    def write_dataset(root: Path, row: dict[str, str]) -> None:
        import csv

        (root / "images").mkdir()
        (root / "Annotations").mkdir()
        Image.new("RGB", (8, 8), "black").save(root / "images" / row["image_id"])
        with (root / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_invalid_flag_is_not_silently_converted_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.normal_row()
            row["tumor"] = "yes"
            self.write_dataset(root, row)

            with self.assertRaisesRegex(ValueError, "must contain 0 or 1"):
                btxrd.load_btxrd_records(root)

    def test_missing_pseudo_mask_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.normal_row()
            self.write_dataset(root, row)
            masks = root / "masks"
            masks.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "pseudo-masks are missing"):
                btxrd.BTXRDSegmentationDataset(root, split="train", pred_mask_dir=masks)

    def test_training_resizes_verified_pseudo_masks_from_source_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.normal_row()
            self.write_dataset(root, row)
            pseudo = root / "pseudo"
            (pseudo / "masks").mkdir(parents=True)
            (pseudo / "run_metadata.json").write_text(
                '{"protocol":"ground_truth"}\n', encoding="utf-8"
            )
            Image.new("L", (8, 8), 0).save(pseudo / "masks" / "normal.png")
            mask_manifest.write_pseudo_mask_manifest(
                pseudo,
                [{
                    "image_name": "normal.png", "true_tumor": 0,
                    "status": "empty_by_image_gate", "above_threshold_candidates": 0,
                    "selected_candidates": 0, "selected_components": 0,
                    "sam_prompt_calls": 0, "unique_prompt_points": 0,
                }],
                expected_image_names=["normal.png"],
                split="train",
                image_size=8,
                run_metadata_sha256=mask_manifest.sha256_file(
                    pseudo / "run_metadata.json"
                ),
            )

            pseudo_package = types.ModuleType("pseudo")
            pseudo_package.manifest = mask_manifest
            with mock.patch.dict(
                sys.modules,
                {"pseudo": pseudo_package, "pseudo.manifest": mask_manifest},
            ):
                dataset = btxrd.BTXRDSegmentationDataset(
                    root,
                    split="train",
                    image_size=16,
                    pred_mask_dir=pseudo / "masks",
                )
            self.assertEqual(dataset.pseudo_manifest_info["source_image_size"], 8)
            self.assertEqual(dataset.pseudo_manifest_info["consumer_image_size"], 16)
            self.assertIs(dataset.pseudo_manifest_info["resized_for_consumer"], True)
            resized_mask = dataset._build_mask(dataset.samples[0], (16, 16))
            self.assertEqual(resized_mask.size, (16, 16))

    @staticmethod
    def write_split_manifest(root: Path, path: Path, rows: list[dict[str, str]]) -> None:
        import csv

        fields = [
            "image_id", "group_id", "split", "eligible", "tumor", "benign", "malignant",
            "tumor_type", "image_sha256", "dataset_table", "dataset_table_sha256",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_manifest_assignments_are_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.normal_row("normal.png")
            self.write_dataset(root, row)
            Image.new("RGB", (8, 8), "white").save(root / "images" / "normal2.png")
            manifest = root / "split_manifest.csv"
            self.write_split_manifest(
                root,
                manifest,
                [
                    {"image_id": "normal.png", "group_id": "g1", "split": "train", "eligible": "1",
                     "tumor": "0", "benign": "0", "malignant": "0", "tumor_type": "0",
                     "image_sha256": "", "dataset_table": "dataset.csv", "dataset_table_sha256": ""},
                    {"image_id": "normal2.png", "group_id": "g2", "split": "val", "eligible": "1",
                     "tumor": "0", "benign": "0", "malignant": "0", "tumor_type": "0",
                     "image_sha256": "", "dataset_table": "dataset.csv", "dataset_table_sha256": ""},
                ],
            )
            records = btxrd.load_btxrd_records(root, split_manifest=manifest)
            self.assertEqual([x["image_id"] for x in btxrd.split_btxrd_records(records, "train")], ["normal.png"])
            self.assertEqual([x["image_id"] for x in btxrd.split_btxrd_records(records, "val")], ["normal2.png"])

    def test_manifest_rejects_group_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.normal_row("normal.png")
            self.write_dataset(root, row)
            Image.new("RGB", (8, 8), "white").save(root / "images" / "normal2.png")
            manifest = root / "split_manifest.csv"
            rows = [
                {"image_id": "normal.png", "group_id": "same", "split": "train", "eligible": "1",
                 "tumor": "0", "benign": "0", "malignant": "0", "tumor_type": "0",
                 "image_sha256": "", "dataset_table": "dataset.csv", "dataset_table_sha256": ""},
                {"image_id": "normal2.png", "group_id": "same", "split": "test", "eligible": "1",
                 "tumor": "0", "benign": "0", "malignant": "0", "tumor_type": "0",
                 "image_sha256": "", "dataset_table": "dataset.csv", "dataset_table_sha256": ""},
            ]
            self.write_split_manifest(root, manifest, rows)
            with self.assertRaisesRegex(ValueError, "group overlap"):
                btxrd.load_btxrd_records(root, split_manifest=manifest)

    def test_manifest_rejects_changed_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.normal_row("normal.png")
            self.write_dataset(root, row)
            manifest = root / "split_manifest.csv"
            self.write_split_manifest(
                root,
                manifest,
                [{"image_id": "normal.png", "group_id": "g1", "split": "train", "eligible": "1",
                  "tumor": "0", "benign": "0", "malignant": "0", "tumor_type": "0",
                  "image_sha256": "not-the-real-hash", "dataset_table": "dataset.csv",
                  "dataset_table_sha256": ""}],
            )
            with self.assertRaisesRegex(ValueError, "image hash does not match"):
                btxrd.load_btxrd_records(root, split_manifest=manifest)

if __name__ == "__main__":
    unittest.main()
