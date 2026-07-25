from __future__ import annotations

import unittest
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_sam_predictor():
    path = Path(__file__).resolve().parents[1] / "project" / "pseudo" / "sam_refine.py"
    spec = importlib.util.spec_from_file_location("sam_refine_grid_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SAMPredictor


SAMPredictor = _load_sam_predictor()


class _FakeAutomaticMaskGenerator:
    last_kwargs: dict | None = None

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        type(self).last_kwargs = kwargs

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        h, w = image_rgb.shape[:2]
        first = np.zeros((h, w), dtype=bool)
        second = np.zeros((h, w), dtype=bool)
        first[1:3, 1:3] = True
        second[2:4, 2:4] = True
        return [
            {"segmentation": first, "predicted_iou": 0.91},
            {"segmentation": second, "predicted_iou": 0.83},
        ]


class _FakePredictor:
    model = object()


class Pro2SAMGridGalleryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = SAMPredictor.__new__(SAMPredictor)
        self.predictor._predictor = _FakePredictor()
        self.predictor._automatic_mask_generator_cls = _FakeAutomaticMaskGenerator
        self.predictor.last_prompt_stats = {}

    def test_dense_grid_uses_official_generator_and_preserves_candidates(self) -> None:
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        masks, scores = self.predictor.predict_grid_gallery(
            image,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            box_nms_thresh=0.7,
        )

        self.assertEqual(masks.shape, (2, 8, 10))
        self.assertEqual(masks.dtype, np.bool_)
        np.testing.assert_allclose(scores, [0.91, 0.83], rtol=0, atol=1e-6)
        self.assertEqual(
            _FakeAutomaticMaskGenerator.last_kwargs,
            {
                "points_per_side": 32,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "box_nms_thresh": 0.7,
                "crop_n_layers": 0,
                "min_mask_region_area": 0,
                "output_mode": "binary_mask",
            },
        )
        self.assertEqual(self.predictor.last_prompt_stats["sam_prompt_calls"], 1024)
        self.assertEqual(self.predictor.last_prompt_stats["unique_prompt_points"], 1024)
        self.assertEqual(self.predictor.last_prompt_stats["box_prompt_calls"], 0)

    def test_invalid_grid_configuration_fails_closed(self) -> None:
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.predictor.predict_grid_gallery(image, points_per_side=0)
        with self.assertRaises(ValueError):
            self.predictor.predict_grid_gallery(image, points_per_batch=0)
        with self.assertRaises(ValueError):
            self.predictor.predict_grid_gallery(image, pred_iou_thresh=1.1)
        with self.assertRaises(ValueError):
            self.predictor.predict_grid_gallery(image, stability_score_thresh=-0.1)
        with self.assertRaises(ValueError):
            self.predictor.predict_grid_gallery(image, box_nms_thresh=2.0)


if __name__ == "__main__":
    unittest.main()
