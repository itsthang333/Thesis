from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def test_s2c_generator_is_stride4_and_has_no_global_bypass() -> None:
    from models.s2c import DenseNet121S2C

    model = DenseNet121S2C(
        fpn_channels=16,
        embedding_dim=8,
        pretrained=False,
        dropout=0.0,
        top_fraction=0.0025,
    ).eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 64, 64)))
    assert output["tumor_cam_logits"].shape == (1, 1, 16, 16)
    assert output["embedding"].shape == (1, 8, 16, 16)
    assert output["tumor_logit"].shape == (1,)
    assert model.classification_pool.mode == "top_percent"
    assert not hasattr(model, "classifier")


def test_s2c_cam_guided_proposal_uses_image_label_and_local_evidence() -> None:
    from models.s2c import select_cam_guided_proposals

    cam = torch.zeros((8, 8))
    cam[:2, :2] = 1.0
    proposals = torch.zeros((2, 8, 8), dtype=torch.bool)
    proposals[0, :2, :2] = True
    proposals[1, 4:, 4:] = True
    quality = torch.tensor([0.9, 0.9])
    selected, info = select_cam_guided_proposals(
        cam, proposals, quality, image_is_tumor=True,
        positive_threshold=0.5, min_positive_score=0.2,
        min_sam_quality=0.7, top_k=1,
    )
    assert info["selected_ids"] == [0]
    assert np.array_equal(selected.numpy(), proposals[0].numpy())

    normal, normal_info = select_cam_guided_proposals(
        cam, proposals, quality, image_is_tumor=False
    )
    assert not normal.any()
    assert normal_info["reason"] == "known_image_label_normal"


def test_x4_s2c_trainer_has_fixed_terminal_no_outer_validation_contract() -> None:
    script = PROJECT / "train_x4_s2c_generator.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--train-segment-cache" in completed.stdout
    assert "--val-segment-cache" not in completed.stdout
    assert "--early-stop-patience" not in completed.stdout


def test_x4_s2c_freezer_help_is_importable() -> None:
    script = PROJECT / "freeze_x4_s2c_masks.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-cache-manifest-sha256" in completed.stdout
    assert "--expected-checkpoint-sha256" in completed.stdout
