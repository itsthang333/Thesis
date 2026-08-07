from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from freeze_g4_source_correct_upstream import resize_float_map_bilinear


class SourceCorrectUpstreamFreezeTests(unittest.TestCase):
    def test_float_resize_is_finite_bounded_and_repeatable(self) -> None:
        source = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        first = resize_float_map_bilinear(source, (5, 7))
        second = resize_float_map_bilinear(source, (5, 7))
        self.assertEqual(first.shape, (5, 7))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        np.testing.assert_array_equal(first, second)

    def test_float_resize_rejects_nonfinite_or_unbounded_maps(self) -> None:
        with self.assertRaises(ValueError):
            resize_float_map_bilinear(np.asarray([[np.nan]], dtype=np.float32), (2, 2))
        with self.assertRaises(ValueError):
            resize_float_map_bilinear(np.asarray([[1.1]], dtype=np.float32), (1, 1))

    def test_freezer_source_does_not_import_spatial_dataset_decoder(self) -> None:
        source = (PROJECT / "freeze_g4_source_correct_upstream.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_decode_labelme_polygon_mask", source)
        self.assertNotIn("Annotations", source)
        self.assertIn("allow_test=False", source)
        self.assertIn('"test_images_read": 0', source)


if __name__ == "__main__":
    unittest.main()
