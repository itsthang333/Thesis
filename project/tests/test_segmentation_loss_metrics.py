from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.losses import (
    binary_segmentation_metric_sums,
    finalize_binary_segmentation_metrics,
    grouped_pseudo_segmentation_loss,
    soft_boundary_weight_map,
)
from train_segmentation import run_epoch


class PseudoSegmentationMetricTests(unittest.TestCase):
    def _metrics(self, probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        status = torch.tensor([True, False])
        return finalize_binary_segmentation_metrics(
            binary_segmentation_metric_sums(probabilities, targets, status, threshold=0.5)
        )

    def test_perfect_tumor_and_empty_normal_score_one(self) -> None:
        targets = torch.zeros(2, 1, 4, 4)
        targets[0, 0, 1:3, 1:3] = 1
        metrics = self._metrics(targets.clone(), targets)
        self.assertAlmostEqual(metrics["tumor_dice"], 1.0, places=6)
        self.assertAlmostEqual(metrics["normal_specificity"], 1.0, places=6)
        self.assertAlmostEqual(metrics["hmean"], 1.0, places=6)

    def test_all_background_cannot_win_checkpoint_selection(self) -> None:
        targets = torch.zeros(2, 1, 4, 4)
        targets[0, 0, 1:3, 1:3] = 1
        metrics = self._metrics(torch.zeros_like(targets), targets)
        self.assertLess(metrics["tumor_dice"], 1e-5)
        self.assertEqual(metrics["normal_specificity"], 1.0)
        self.assertLess(metrics["hmean"], 1e-5)

    def test_all_foreground_cannot_win_checkpoint_selection(self) -> None:
        targets = torch.zeros(2, 1, 4, 4)
        targets[0, 0, 1:3, 1:3] = 1
        metrics = self._metrics(torch.ones_like(targets), targets)
        self.assertEqual(metrics["normal_specificity"], 0.0)
        self.assertEqual(metrics["hmean"], 0.0)


class PseudoSegmentationLossTests(unittest.TestCase):
    def test_blank_tumor_is_unknown_and_has_no_supervised_gradient(self) -> None:
        logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
        targets = torch.zeros_like(logits)
        status = torch.tensor([True, False])
        loss, diagnostics = grouped_pseudo_segmentation_loss(logits, targets, status)
        loss.backward()
        self.assertEqual(diagnostics["blank_tumor_images"], 1.0)
        self.assertTrue(torch.equal(logits.grad[0], torch.zeros_like(logits.grad[0])))
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.0)

    def test_soft_boundary_never_deletes_one_pixel_lesion(self) -> None:
        target = torch.zeros(1, 1, 5, 5)
        target[0, 0, 2, 2] = 1
        weights = soft_boundary_weight_map(target, radius=1, boundary_weight=0.25)
        self.assertEqual(float(weights[0, 0, 2, 2]), 0.25)
        self.assertGreater(float(weights.min()), 0.0)

    def test_loss_is_finite_and_backpropagates_for_mixed_batch(self) -> None:
        logits = torch.randn(3, 1, 8, 8, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, 0, 2:5, 3:6] = 1
        status = torch.tensor([True, False, True])  # third sample is an unknown blank tumor
        weights = soft_boundary_weight_map(targets, radius=1, boundary_weight=0.25)
        loss, _ = grouped_pseudo_segmentation_loss(
            logits, targets, status, pos_weight=8.0, pixel_weights=weights
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad[:2].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[2].abs().sum()), 0.0)


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.images = torch.randn(4, 3, 8, 8)
        self.masks = torch.zeros(4, 1, 8, 8)
        self.masks[0, 0, 2:5, 2:5] = 1
        self.masks[1, 0, 1:4, 3:6] = 1
        self.names = ["tumor-a.png", "tumor-b.png", "normal-a.png", "normal-b.png"]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        return self.images[index], self.masks[index], self.names[index]


class TrainerSmokeTests(unittest.TestCase):
    def test_pseudo_validation_threshold_sweep_runs(self) -> None:
        dataset = _TinyDataset()
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = nn.Conv2d(3, 1, kernel_size=1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        loss, metrics = run_epoch(
            model,
            loader,
            scaler,
            torch.device("cpu"),
            train=False,
            pos_weight=4.0,
            pseudo_supervision=True,
            group_explicit_metrics=True,
            tumor_status_by_name={name: name.startswith("tumor") for name in dataset.names},
            pseudo_boundary_soft_px=1,
            metric_thresholds=(0.3, 0.5, 0.7),
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertIn(metrics["threshold"], (0.3, 0.5, 0.7))
        self.assertIn("normal_specificity", metrics)
        self.assertIn("checkpoint_score", metrics)


if __name__ == "__main__":
    unittest.main()
