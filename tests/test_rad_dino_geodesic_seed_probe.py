from __future__ import annotations

import unittest

import numpy as np

from models.mae_reconstruction import SquareProjection
from run_rad_dino_geodesic_seed_probe import project_square_features


class RadDinoGeodesicSeedProbeTests(unittest.TestCase):
    def test_multichannel_projection_preserves_identity_geometry(self) -> None:
        first = np.arange(16, dtype=np.float32).reshape(4, 4)
        second = first[::-1].copy()
        features = np.stack([first, second], axis=0)
        projection = SquareProjection(
            padded_side=4,
            content_box=(0, 0, 4, 4),
        )
        actual = project_square_features(
            features,
            projection,
            output_height=4,
            output_width=4,
        )
        np.testing.assert_allclose(actual, features, atol=1e-6, rtol=1e-6)

    def test_content_crop_keeps_channels_finite_and_oriented(self) -> None:
        horizontal = np.broadcast_to(
            np.arange(4, dtype=np.float32)[None, :], (4, 4)
        )
        vertical = horizontal.T.copy()
        actual = project_square_features(
            np.stack([horizontal, vertical], axis=0),
            SquareProjection(padded_side=8, content_box=(1, 2, 7, 6)),
            output_height=5,
            output_width=7,
        )
        self.assertEqual(actual.shape, (2, 5, 7))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(np.diff(actual[0], axis=1) >= 0.0))
        self.assertTrue(np.all(np.diff(actual[1], axis=0) >= 0.0))

    def test_projection_rejects_non_square_features(self) -> None:
        with self.assertRaises(ValueError):
            project_square_features(
                np.zeros((2, 3, 4), dtype=np.float32),
                SquareProjection(padded_side=4, content_box=(0, 0, 4, 4)),
                output_height=4,
                output_width=4,
            )


if __name__ == "__main__":
    unittest.main()
