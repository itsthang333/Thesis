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

    def test_cpm_peak_extraction_keeps_global_and_separated_peaks(self) -> None:
        import torch

        from train_s2c_cpm_classifier import extract_cpm_peaks

        cam = torch.zeros(40, 40)
        cam[4, 5] = 1.0
        cam[5, 6] = 0.9
        cam[30, 31] = 0.8
        peaks = extract_cpm_peaks(
            cam,
            threshold=0.5,
            min_distance=10,
            max_peaks=8,
        )
        self.assertEqual(peaks.tolist(), [[5.0, 4.0], [31.0, 30.0]])

    def test_direct_cam_adapter_returns_normalized_multiscale_map(self) -> None:
        import torch
        from torch import nn

        from models.s2c_cpm import S2CCPMDirectCAM

        class FakeCPM(nn.Module):
            def forward(self, images, *, return_spatial=False):
                cam = images[:, :1]
                logits = cam.mean(dim=(2, 3))
                features = cam.repeat(1, 2, 1, 1)
                return (logits, features, cam) if return_spatial else logits

        adapter = S2CCPMDirectCAM(FakeCPM(), scales=(0.5, 1.0))
        image = torch.linspace(0, 1, 64).view(1, 1, 8, 8).repeat(1, 3, 1, 1)
        output = adapter.cam_for_class(image, 0)
        self.assertEqual(tuple(output.cam.shape), (1, 8, 8))
        self.assertGreaterEqual(float(output.cam.min()), 0.0)
        self.assertLessEqual(float(output.cam.max()), 1.0)
        self.assertGreater(float(output.cam.max()), 0.99)

    def test_multiscale_teacher_cam_restores_training_state(self) -> None:
        import torch
        from torch import nn

        from train_s2c_cpm_classifier import multiscale_teacher_cam

        class FakeCPM(nn.Module):
            def forward(self, images, *, return_spatial=False):
                cam = images[:, :1]
                logits = cam.mean(dim=(2, 3))
                features = cam
                return (logits, features, cam) if return_spatial else logits

        model = FakeCPM().train()
        output = multiscale_teacher_cam(model, torch.rand(1, 3, 16, 16))
        self.assertTrue(model.training)
        self.assertEqual(tuple(output.shape), (1, 1, 320, 320))


if __name__ == "__main__":
    unittest.main()
