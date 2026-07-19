from __future__ import annotations

import sys
import tempfile
import types
import unittest
import importlib.util
from pathlib import Path

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


class TumorMorphologyTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
