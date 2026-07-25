from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from project.models.biomedclip_saliency import (
    TileSaliency,
    aggregate_full_and_tiles,
    pad_to_square,
    project_padded_square_map,
    robust_normalize,
    square_crop_boxes,
)


class BiomedClipSaliencyGeometryTests(unittest.TestCase):
    def test_square_crop_grid_is_deterministic_and_in_bounds(self) -> None:
        boxes = square_crop_boxes(100, 80, crop_fraction=0.5, positions_per_axis=3)
        self.assertEqual(len(boxes), 9)
        self.assertEqual(boxes[0], (0, 0, 40, 40))
        self.assertEqual(boxes[4], (30, 20, 70, 60))
        self.assertEqual(boxes[-1], (60, 40, 100, 80))

    def test_padding_and_projection_remove_only_padding(self) -> None:
        image = Image.new("RGB", (6, 4), color=(10, 10, 10))
        padded, content = pad_to_square(image)
        self.assertEqual(padded.size, (6, 6))
        self.assertEqual(content, (0, 1, 6, 5))
        square_map = np.zeros((6, 6), dtype=np.float32)
        square_map[1:5] = 2.0
        projected = project_padded_square_map(
            square_map,
            padded_side=6,
            content_box=content,
            output_height=4,
            output_width=6,
        )
        np.testing.assert_allclose(projected, 2.0)

    def test_robust_normalization_is_bounded(self) -> None:
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        normalized = robust_normalize(values)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        self.assertEqual(float(robust_normalize(np.ones((3, 3))).max()), 0.0)

    def test_aggregation_uses_only_top_three_scores(self) -> None:
        full = np.zeros((8, 8), dtype=np.float32)
        tiles = [
            TileSaliency((0, 0, 4, 4), 0.1, np.arange(16, dtype=np.float32).reshape(4, 4)),
            TileSaliency((4, 0, 8, 4), 0.4, np.arange(16, dtype=np.float32).reshape(4, 4)),
            TileSaliency((0, 4, 4, 8), 0.3, np.arange(16, dtype=np.float32).reshape(4, 4)),
            TileSaliency((4, 4, 8, 8), 0.2, np.arange(16, dtype=np.float32).reshape(4, 4)),
        ]
        fused, selected = aggregate_full_and_tiles(
            full, tiles, output_height=8, output_width=8, top_k_tiles=3
        )
        self.assertEqual([tile.contrast_score for tile in selected], [0.4, 0.3, 0.2])
        self.assertEqual(float(fused[:4, :4].max()), 0.0)
        self.assertGreater(float(fused[:4, 4:].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
