from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from train_x4_matched_student import choose_threshold, threshold_metrics  # noqa: E402
from models.unet import build_segmentation_model  # noqa: E402
from x4_training_targets import inspect_mask  # noqa: E402


def test_threshold_rule_uses_positive_dice_then_empty_specificity_then_lower() -> None:
    metrics = {
        0.2: {"target_positive_dice": 0.5, "target_empty_specificity": 0.8},
        0.3: {"target_positive_dice": 0.5, "target_empty_specificity": 0.8},
        0.4: {"target_positive_dice": 0.5, "target_empty_specificity": 0.7},
    }
    threshold, _ = choose_threshold(metrics)
    assert threshold == 0.2


def test_threshold_metrics_separates_positive_and_empty_targets() -> None:
    probabilities = torch.tensor([[[[0.9, 0.1]]], [[[0.1, 0.1]]]])
    targets = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    result = threshold_metrics(probabilities, targets, thresholds=(0.5,))
    assert result[0.5]["positive_count"] == 1
    assert result[0.5]["positive_dice_sum"] == pytest.approx(1.0)
    assert result[0.5]["empty_count"] == 1
    assert result[0.5]["empty_correct"] == 1


def test_x4_mask_inspection_rejects_nonbinary(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    Image.fromarray(np.asarray([[0, 12], [0, 255]], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="not a binary"):
        inspect_mask(path)


def test_protocol_declares_no_outer_validation_selection() -> None:
    protocol = json.loads(
        (PROJECT.parent / "artifacts/final_pipeline/x4/x4_protocol.json").read_text()
    )
    assert protocol["inner_split"]["outer_validation_selection_forbidden"] is True
    assert protocol["student_seeds"] == [42, 43, 44]


def test_resnet18_unet_accepts_explicit_offline_encoder_state() -> None:
    reference = build_segmentation_model("resnet18_unet", pretrained=False)
    # Reconstruct the torchvision key space expected by the explicit loader.
    from torchvision.models import resnet18

    encoder = resnet18(weights=None)
    with torch.no_grad():
        encoder.conv1.weight.fill_(0.125)
    model = build_segmentation_model(
        "resnet18_unet",
        pretrained=False,
        encoder_state_dict=encoder.state_dict(),
    )
    assert torch.all(model.stem[0].weight == 0.125)
    assert not torch.all(reference.stem[0].weight == 0.125)
