from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_x4_matched_student_output import audit_training_output  # noqa: E402
from frozen_io import sha256_file  # noqa: E402
from x4_contract import (  # noqa: E402
    CANONICAL_SPLIT_SHA256,
    RESNET18_IMAGENET1K_V1_SHA256,
    load_x4_protocol,
)


INNER_SHA = "a" * 64
TARGET_SHA = "b" * 64


def _checkpoint(epoch: int, threshold: float, protocol_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "x4_matched_student_checkpoint_v1",
        "arm": "cam",
        "seed": 42,
        "epoch": epoch,
        "model_state_dict": {"weight": torch.ones(2, 3)},
        "architecture": {"name": "resnet18_unet"},
        "model_architecture": "resnet18_unet",
        "pretrained_encoder": True,
        "encoder_weight_sha256": RESNET18_IMAGENET1K_V1_SHA256,
        "image_size": 448,
        "decision_threshold": threshold,
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "inner_split_sha256": INNER_SHA,
        "x4_protocol_sha256": protocol_sha,
        "target_freeze_sha256": TARGET_SHA,
        "supervision_mode": "image_label_only_pseudo_mask",
        "outer_validation_checkpoint_selection": False,
        "test_evaluated": False,
    }


def _bundle(root: Path) -> None:
    protocol, protocol_sha = load_x4_protocol(REPO)
    root.mkdir()
    history_path = root / "training_history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "inner_holdout_loss",
        "selected_threshold",
        "inner_target_positive_dice",
        "inner_target_empty_specificity",
        "amp_skipped_steps",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, 31):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "inner_holdout_loss": 0.1 if epoch == 7 else 1.0 + epoch / 100.0,
                    "selected_threshold": 0.5,
                    "inner_target_positive_dice": 0.4,
                    "inner_target_empty_specificity": 0.8,
                    "amp_skipped_steps": 0,
                }
            )
    best_path = root / "best_student.pt"
    last_path = root / "last_student.pt"
    torch.save(_checkpoint(7, 0.5, protocol_sha), best_path)
    torch.save(_checkpoint(30, 0.5, protocol_sha), last_path)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "stage": "x4_matched_student_training_v1",
        "arm": "cam",
        "seed": 42,
        "train_images": 2516,
        "inner_holdout_images": 465,
        "outer_validation_images_used": 0,
        "best_epoch": 7,
        "best_threshold": 0.5,
        "epochs_completed": 30,
        "amp_skipped_steps": 0,
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "inner_split_sha256": INNER_SHA,
        "x4_protocol_sha256": protocol_sha,
        "target_freeze_sha256": TARGET_SHA,
        "target_freeze": {"stage": "x4_train_target_freeze_v1"},
        "architecture": {"name": "resnet18_unet"},
        "encoder_weight_sha256": RESNET18_IMAGENET1K_V1_SHA256,
        "scientific_config": protocol["matched_student"],
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "training_history_sha256": sha256_file(history_path),
        "cuda_devices": ["fixture"],
        "data_parallel": False,
        "spatial_ground_truth_training": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (root / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_x4_student_output_auditor_recomputes_selection(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _bundle(root)
    report = audit_training_output(
        root,
        repo_root=REPO,
        arm="cam",
        seed=42,
        expected_inner_split_sha256=INNER_SHA,
        expected_target_freeze_sha256=TARGET_SHA,
    )
    assert report["pass"] is True
    assert report["best_epoch"] == 7
    assert report["parameters"] == 6


def test_x4_student_output_auditor_rejects_history_tamper(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _bundle(root)
    history = root / "training_history.csv"
    history.write_text(history.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="history SHA-256"):
        audit_training_output(
            root,
            repo_root=REPO,
            arm="cam",
            seed=42,
            expected_inner_split_sha256=INNER_SHA,
            expected_target_freeze_sha256=TARGET_SHA,
        )
