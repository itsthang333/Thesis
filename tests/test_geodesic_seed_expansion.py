from __future__ import annotations

import unittest

import numpy as np

from pseudo.geodesic_seed_expansion import (
    exact_rank_seed_masks,
    exponential_geodesic_fusion,
    geodesic_seed_expansion,
    multi_source_geodesic_distance,
    prepare_geodesic_features,
)


class GeodesicSeedExpansionTests(unittest.TestCase):
    def test_rank_seeds_are_exact_disjoint_and_deterministic(self) -> None:
        probability = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
        valid = np.ones_like(probability, dtype=bool)
        foreground, background = exact_rank_seed_masks(
            probability,
            valid,
            foreground_fraction=0.10,
            background_fraction=0.50,
        )
        self.assertEqual(int(foreground.sum()), 10)
        self.assertEqual(int(background.sum()), 50)
        self.assertFalse(np.logical_and(foreground, background).any())
        self.assertTrue(foreground.reshape(-1)[-10:].all())
        self.assertTrue(background.reshape(-1)[:50].all())

    def test_rank_ties_use_stable_row_major_order(self) -> None:
        probability = np.full((4, 4), 0.5, dtype=np.float32)
        valid = np.ones_like(probability, dtype=bool)
        foreground, background = exact_rank_seed_masks(
            probability,
            valid,
            foreground_fraction=0.25,
            background_fraction=0.25,
        )
        expected_background = np.zeros(16, dtype=bool)
        expected_background[:4] = True
        expected_foreground = np.zeros(16, dtype=bool)
        expected_foreground[-4:] = True
        np.testing.assert_array_equal(background.reshape(-1), expected_background)
        np.testing.assert_array_equal(foreground.reshape(-1), expected_foreground)

    def test_feature_preparation_is_finite_and_zeros_invalid_pixels(self) -> None:
        image = np.arange(25, dtype=np.float32).reshape(5, 5)
        structure = np.stack([image, image[::-1]], axis=0)
        valid = np.ones((5, 5), dtype=bool)
        valid[0, :] = False
        features = prepare_geodesic_features(image, structure, valid)
        self.assertEqual(features.shape, (3, 5, 5))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[:, ~valid] == 0.0))

    def test_geodesic_distance_respects_a_feature_boundary(self) -> None:
        features = np.zeros((1, 5, 5), dtype=np.float32)
        features[:, :, 3:] = 10.0
        valid = np.ones((5, 5), dtype=bool)
        seeds = np.zeros((5, 5), dtype=bool)
        seeds[2, 0] = True
        distance = multi_source_geodesic_distance(features, seeds, valid)
        self.assertLess(float(distance[2, 2]), float(distance[2, 3]))
        self.assertEqual(float(distance[2, 0]), 0.0)

    def test_fusion_is_bounded_and_zeros_invalid_pixels(self) -> None:
        source = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
        foreground = np.tile(
            np.arange(5, dtype=np.float32)[:, None], (1, 5)
        )
        background = foreground[::-1].copy()
        valid = np.ones((5, 5), dtype=bool)
        valid[:, 0] = False
        fused = exponential_geodesic_fusion(
            source, foreground, background, valid
        )
        self.assertTrue(np.isfinite(fused).all())
        self.assertGreaterEqual(float(fused.min()), 0.0)
        self.assertLessEqual(float(fused.max()), 1.0)
        self.assertTrue(np.all(fused[:, 0] == 0.0))

    def test_end_to_end_preserves_rank_seeds_and_ambiguity(self) -> None:
        probability = np.linspace(
            0.0, 1.0, 100, dtype=np.float32
        ).reshape(10, 10)
        image = np.zeros((10, 10), dtype=np.float32)
        image[:, 5:] = 1.0
        structure = np.stack([image, 1.0 - image], axis=0)
        valid = np.ones((10, 10), dtype=bool)
        result = geodesic_seed_expansion(
            probability,
            image,
            structure,
            valid,
            foreground_fraction=0.10,
            background_fraction=0.50,
        )
        self.assertTrue(np.all(result.probability[result.foreground_seeds] == 1.0))
        self.assertTrue(np.all(result.probability[result.background_seeds] == 0.0))
        self.assertEqual(result.diagnostics["foreground_seed_pixels"], 10)
        self.assertEqual(result.diagnostics["background_seed_pixels"], 50)
        self.assertEqual(result.diagnostics["ambiguous_pixels"], 40)
        self.assertTrue(np.isfinite(result.foreground_distance).all())
        self.assertTrue(np.isfinite(result.background_distance).all())


if __name__ == "__main__":
    unittest.main()
