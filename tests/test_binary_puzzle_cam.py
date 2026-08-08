from __future__ import annotations

import torch
from torch import nn

from project.models.puzzle_cam import normalized_classic_cam, puzzle_cam_consistency_loss


class TinyCamClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(5, classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.features(images))


def test_binary_puzzle_cam_uses_one_logit_bce_and_backpropagates() -> None:
    torch.manual_seed(1)
    model = TinyCamClassifier(classes=1)
    images = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([[0.0], [1.0]])
    full, reconstructed, re_loss, puzzle_cls = puzzle_cam_consistency_loss(
        model, images, targets
    )
    assert full.shape == reconstructed.shape == (2, 16, 16)
    assert torch.isfinite(re_loss) and torch.isfinite(puzzle_cls)
    (puzzle_cls + 4.0 * re_loss).backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_multiclass_puzzle_cam_path_is_preserved() -> None:
    torch.manual_seed(2)
    model = TinyCamClassifier(classes=3)
    images = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([0, 2], dtype=torch.long)
    _, _, re_loss, puzzle_cls = puzzle_cam_consistency_loss(model, images, targets)
    assert torch.isfinite(re_loss) and torch.isfinite(puzzle_cls)


def test_binary_inference_cam_is_finite_and_normalized() -> None:
    torch.manual_seed(3)
    model = TinyCamClassifier(classes=1)
    images = torch.randn(2, 3, 16, 16)
    classes = torch.zeros(2, dtype=torch.long)
    cam = normalized_classic_cam(model, images, classes)
    assert cam.shape == (2, 16, 16)
    assert torch.isfinite(cam).all()
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
