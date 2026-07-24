from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised in Kaggle/CI with torch
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight audit environment")
class TorchMathTests(unittest.TestCase):
    def test_regularized_adversarial_climbing_is_bounded_and_aggregates(self) -> None:
        from models.layercam import (
            LayerCAMOutput,
            regularized_adversarial_climbing_layercam,
        )

        class TinyClassifier(torch.nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        class FakeDifferentiableLayerCAM:
            def __init__(self):
                self.model = TinyClassifier()
                self.seen = []

            def cam_for_class_differentiable(self, inputs, class_index):
                self.seen.append(inputs.detach().clone())
                logits = self.model(inputs)
                cam = torch.sigmoid(inputs[:, 0])
                native_cam = torch.sigmoid(inputs[:, 0, ::2, ::2])
                return LayerCAMOutput(logits=logits, cam=cam), native_cam

        image = torch.linspace(-2.0, 2.0, 16).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
        layercam = FakeDifferentiableLayerCAM()
        result = regularized_adversarial_climbing_layercam(
            layercam,
            image,
            0,
            iterations=2,
            step_size=0.08,
            ad_coeff=7.0,
            score_threshold=0.5,
        )

        self.assertEqual(len(layercam.seen), 3)
        self.assertTrue(torch.isfinite(result.cam).all())
        self.assertGreaterEqual(float(result.cam.min()), 0.0)
        self.assertLessEqual(float(result.cam.max()), 1.0)
        for observed in layercam.seen:
            self.assertGreaterEqual(float(observed.min()), float(image.min()))
            self.assertLessEqual(float(observed.max()), float(image.max()))

    def test_regularized_adversarial_climbing_rejects_invalid_coeff(self) -> None:
        from models.layercam import regularized_adversarial_climbing_layercam

        with self.assertRaisesRegex(ValueError, "non-negative"):
            regularized_adversarial_climbing_layercam(
                object(),
                torch.zeros(1, 3, 4, 4),
                0,
                iterations=1,
                ad_coeff=-1.0,
            )

    def test_classifier_budget_audit_runs_after_early_stopping(self) -> None:
        from train_classifier import classifier_epoch_budget_audit

        records = [
            {"epoch": 13, "val_f1": 0.38},
            {"epoch": 14, "val_f1": 0.4096},
            {"epoch": 15, "val_f1": 0.39},
            {"epoch": 16, "val_f1": 0.37},
            {"epoch": 17, "val_f1": 0.36},
            {"epoch": 18, "val_f1": 0.35},
            {"epoch": 19, "val_f1": 0.34},
            {"epoch": 20, "val_f1": 0.33},
            {"epoch": 21, "val_f1": 0.32},
        ]
        audit = classifier_epoch_budget_audit(
            records,
            requested_epochs=30,
            stopped_early=True,
            early_stop_patience=7,
        )

        self.assertEqual(audit["assessment"], "plateau_or_decline_observed")
        self.assertEqual(audit["best_epoch"], 14)
        self.assertAlmostEqual(audit["best_val_f1"], 0.4096)
        self.assertTrue(audit["valid_early_stop"])
        self.assertIn("early stopping fired", audit["assessment_basis"])

    def test_bce_dice_formula(self) -> None:
        from models.losses import bce_dice_loss, dice_loss_from_logits

        logits = torch.zeros((1, 1, 1, 2))
        targets = torch.tensor([[[[1.0, 0.0]]]])
        dice = dice_loss_from_logits(logits, targets)
        expected_dice = 1.0 - (2 * 0.5 + 1e-6) / (1.0 + 1.0 + 1e-6)
        self.assertAlmostEqual(float(dice), expected_dice, places=6)
        expected = 0.5 * math.log(2.0) + 0.5 * expected_dice
        self.assertAlmostEqual(float(bce_dice_loss(logits, targets)), expected, places=6)

    def test_puzzle_tile_merge_is_exact(self) -> None:
        from models.puzzle_cam import merge_2x2, tile_2x2

        image = torch.arange(16).reshape(1, 1, 4, 4).float()
        reconstructed = merge_2x2(tile_2x2(image).squeeze(1), batch_size=1)
        self.assertTrue(torch.equal(reconstructed, image.squeeze(1)))

    def test_flat_puzzle_cam_normalises_to_zero(self) -> None:
        from models.puzzle_cam import _normalize_cam

        self.assertFalse(_normalize_cam(torch.ones((2, 4, 4))).any())

    def test_unet_preserves_spatial_shape_and_backpropagates(self) -> None:
        from models.unet import UNet

        model = UNet(in_channels=3, out_channels=1, base_channels=8)
        inputs = torch.randn(2, 3, 32, 32)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (2, 1, 32, 32))
        output.mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_resnet18_unet_preserves_spatial_shape_and_backpropagates(self) -> None:
        from models.unet import ResNet18UNet

        model = ResNet18UNet(out_channels=1, pretrained=False)
        inputs = torch.randn(1, 3, 64, 64)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
        output.mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_postprocess_guidance_threshold_fails_closed(self) -> None:
        import numpy as np
        from pseudo.morphology import morphological_refinement

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        result = morphological_refinement(
            mask,
            min_size=1,
            guidance_map=np.zeros((8, 8), dtype=np.float32),
            guidance_threshold=0.4,
        )
        self.assertFalse(result.any())


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight audit environment")
class ResumeIntegrationTests(unittest.TestCase):
    def test_resume_config_is_path_portable_but_training_strict(self) -> None:
        from train_segmentation import _portable_resume_config

        local = {
            "data_root": r"D:\\thesis\\BTXRD\\BTXRD",
            "split_manifest": r"D:\\thesis\\artifacts\\split_manifest.csv",
            "image_size": 320,
            "batch_size": 8,
            "early_stop_patience": 0,
        }
        kaggle = dict(local)
        kaggle["data_root"] = "/kaggle/input/btxrd-raw/BTXRD"
        kaggle["split_manifest"] = "/kaggle/input/research-bundle/split_manifest.csv"
        kaggle["early_stop_patience"] = 8
        self.assertEqual(_portable_resume_config(local), _portable_resume_config(kaggle))

        kaggle["image_size"] = 448
        self.assertNotEqual(_portable_resume_config(local), _portable_resume_config(kaggle))

    def test_resume_matches_uninterrupted_next_step(self) -> None:
        from models.losses import bce_dice_loss
        from models.unet import UNet
        from train_segmentation import save_checkpoint

        torch.manual_seed(7)
        model = UNet(in_channels=3, out_channels=1, base_channels=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        first_x = torch.randn(1, 3, 32, 32)
        first_y = (torch.rand(1, 1, 32, 32) > 0.8).float()
        second_x = torch.randn(1, 3, 32, 32)
        second_y = (torch.rand(1, 1, 32, 32) > 0.8).float()

        optimizer.zero_grad()
        bce_dice_loss(model(first_x), first_y).backward()
        optimizer.step()
        saved_best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        run_args = argparse.Namespace(
            dataset="btxrd", ram_root=Path("BTXRD"), train_split="train", val_split="val",
            split_manifest=None, annotation_name="unused", image_size=32, batch_size=1,
            lr=1e-4, weight_decay=1e-4, seed=7, use_clahe=False, early_stop_patience=0,
            train_pred_mask_root=None, val_pred_mask_root=None, pos_weight_mode="none",
            pos_weight_value=None, output_dir=Path("unused"), resume_from=None, epochs=2,
            num_workers=0, multi_gpu=False,
            model_architecture="unet", no_pretrained_encoder=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "last.pt"
            save_checkpoint(
                checkpoint_path, model, optimizer, 1, 0.2, "btxrd", None, scaler,
                run_args, 0, saved_best, 1, 1,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("scaler_state_dict", checkpoint)
            self.assertIn("best_model_state_dict", checkpoint)
            self.assertIn("resolved_config_sha256", checkpoint)

            uninterrupted_model = copy.deepcopy(model)
            uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_model.parameters(), lr=1e-4)
            uninterrupted_optimizer.load_state_dict(optimizer.state_dict())
            uninterrupted_optimizer.zero_grad()
            bce_dice_loss(uninterrupted_model(second_x), second_y).backward()
            uninterrupted_optimizer.step()

            resumed_model = UNet(in_channels=3, out_channels=1, base_channels=64)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)
            resumed_model.load_state_dict(checkpoint["model_state_dict"])
            resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resumed_optimizer.zero_grad()
            bce_dice_loss(resumed_model(second_x), second_y).backward()
            resumed_optimizer.step()
            for expected, actual in zip(uninterrupted_model.parameters(), resumed_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
