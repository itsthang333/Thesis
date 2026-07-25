from __future__ import annotations

import unittest


class S2CCPMTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional local dependency
            self.skipTest(f"torch unavailable: {exc}")

    def test_model_emits_stride8_cam_and_fused_features(self) -> None:
        import torch

        from models.s2c_cpm import DenseNet121S2CCPMClassifier

        model = DenseNet121S2CCPMClassifier(
            pretrained=False,
            feature_channels=32,
        ).eval()
        with torch.no_grad():
            logits, features, cams = model(
                torch.randn(1, 3, 64, 64),
                return_spatial=True,
            )
        self.assertEqual(tuple(logits.shape), (1, 1))
        self.assertEqual(tuple(features.shape), (1, 32, 8, 8))
        self.assertEqual(tuple(cams.shape), (1, 1, 8, 8))
        self.assertEqual(model.feature_stride, 8)

    def test_cpm_loss_prefers_cam_aligned_with_sam_mask(self) -> None:
        import torch

        from models.s2c_cpm import cpm_cross_entropy_loss

        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2, :2] = 1
        aligned = torch.full((1, 1, 4, 4), -3.0)
        aligned[:, :, :2, :2] = 3.0
        inverted = -aligned
        labels = torch.ones(1, 1)
        self.assertLess(
            float(cpm_cross_entropy_loss(aligned, mask, labels)),
            float(cpm_cross_entropy_loss(inverted, mask, labels)),
        )

    def test_known_normal_forces_all_background(self) -> None:
        import torch

        from models.s2c_cpm import cpm_cross_entropy_loss

        accidental_mask = torch.ones(1, 1, 4, 4)
        labels = torch.zeros(1, 1)
        background_cam = torch.full((1, 1, 4, 4), -2.0)
        foreground_cam = torch.full((1, 1, 4, 4), 2.0)
        self.assertLess(
            float(cpm_cross_entropy_loss(background_cam, accidental_mask, labels)),
            float(cpm_cross_entropy_loss(foreground_cam, accidental_mask, labels)),
        )


if __name__ == "__main__":
    unittest.main()
