from __future__ import annotations

"""Independently audit one completed matched X4 student training bundle."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from frozen_io import sha256_file
from x4_contract import (
    CANONICAL_SPLIT_SHA256,
    PSEUDO_STUDENT_ARMS,
    RESNET18_IMAGENET1K_V1_SHA256,
    STUDENT_ARMS,
    STUDENT_SEEDS,
    THRESHOLD_GRID,
    load_x4_protocol,
)


EPOCHS = 30
TRAIN_IMAGES = 2516
INNER_HOLDOUT_IMAGES = 465


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite X4 history value: {key}")
    return value


def _validate_checkpoint(
    checkpoint: object,
    *,
    arm: str,
    seed: int,
    expected_epoch: int,
    expected_threshold: float,
    expected_inner_split_sha256: str,
    expected_target_freeze_sha256: str | None,
    protocol_sha256: str,
) -> int:
    if not isinstance(checkpoint, dict):
        raise ValueError("X4 student checkpoint is not a mapping")
    expected_supervision = (
        "fully_supervised" if arm == "fully_supervised" else "image_label_only_pseudo_mask"
    )
    required = {
        "schema_version": 1,
        "stage": "x4_matched_student_checkpoint_v1",
        "arm": arm,
        "seed": seed,
        "epoch": expected_epoch,
        "model_architecture": "resnet18_unet",
        "pretrained_encoder": True,
        "encoder_weight_sha256": RESNET18_IMAGENET1K_V1_SHA256,
        "image_size": 448,
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "inner_split_sha256": expected_inner_split_sha256,
        "x4_protocol_sha256": protocol_sha256,
        "target_freeze_sha256": expected_target_freeze_sha256,
        "supervision_mode": expected_supervision,
        "outer_validation_checkpoint_selection": False,
        "test_evaluated": False,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"X4 checkpoint field differs: {key}")
    if not math.isclose(
        float(checkpoint.get("decision_threshold", float("nan"))),
        expected_threshold,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("X4 checkpoint threshold differs from history")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("X4 student state dict is empty")
    parameters = 0
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("X4 student state dict is not tensor-only")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite X4 student tensor: {key}")
        parameters += int(value.numel())
    if parameters <= 0:
        raise ValueError("X4 student state dict has no parameters")
    return parameters


def audit_training_output(
    output_root: Path,
    *,
    repo_root: Path,
    arm: str,
    seed: int,
    expected_inner_split_sha256: str,
    expected_target_freeze_sha256: str | None,
) -> dict[str, object]:
    if arm not in STUDENT_ARMS or seed not in STUDENT_SEEDS:
        raise ValueError("unknown X4 student arm or seed")
    if arm in PSEUDO_STUDENT_ARMS and expected_target_freeze_sha256 is None:
        raise ValueError("pseudo X4 student audit requires target freeze SHA-256")
    if arm == "fully_supervised" and expected_target_freeze_sha256 is not None:
        raise ValueError("fully supervised X4 audit cannot bind a pseudo target")

    protocol, protocol_sha = load_x4_protocol(repo_root)
    metadata_path = output_root / "training_metadata.json"
    history_path = output_root / "training_history.csv"
    best_path = output_root / "best_student.pt"
    last_path = output_root / "last_student.pt"
    for path in (metadata_path, history_path, best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "schema_version": 1,
        "status": "complete",
        "stage": "x4_matched_student_training_v1",
        "arm": arm,
        "seed": seed,
        "train_images": TRAIN_IMAGES,
        "inner_holdout_images": INNER_HOLDOUT_IMAGES,
        "outer_validation_images_used": 0,
        "epochs_completed": EPOCHS,
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "inner_split_sha256": expected_inner_split_sha256,
        "x4_protocol_sha256": protocol_sha,
        "target_freeze_sha256": expected_target_freeze_sha256,
        "encoder_weight_sha256": RESNET18_IMAGENET1K_V1_SHA256,
        "scientific_config": protocol["matched_student"],
        "spatial_ground_truth_training": arm == "fully_supervised",
        "test_images_read": 0,
        "test_evaluated": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"X4 training metadata differs: {key}")
    if metadata.get("training_history_sha256") != sha256_file(history_path):
        raise ValueError("X4 training history SHA-256 mismatch")
    if metadata.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise ValueError("X4 best checkpoint SHA-256 mismatch")
    if metadata.get("last_checkpoint_sha256") != sha256_file(last_path):
        raise ValueError("X4 last checkpoint SHA-256 mismatch")

    history = _read_csv(history_path)
    if len(history) != EPOCHS or [int(row["epoch"]) for row in history] != list(
        range(1, EPOCHS + 1)
    ):
        raise ValueError("X4 training history is incomplete")
    previous_skips = -1
    keys: list[tuple[float, float, float, float]] = []
    for row in history:
        loss = _finite(row, "inner_holdout_loss")
        _finite(row, "train_loss")
        dice = _finite(row, "inner_target_positive_dice")
        specificity = _finite(row, "inner_target_empty_specificity")
        threshold = _finite(row, "selected_threshold")
        if threshold not in THRESHOLD_GRID:
            raise ValueError("X4 selected threshold is outside the frozen grid")
        if not (0.0 <= dice <= 1.0 and 0.0 <= specificity <= 1.0):
            raise ValueError("X4 inner metric is outside [0, 1]")
        skips = int(row["amp_skipped_steps"])
        if skips < previous_skips:
            raise ValueError("X4 AMP skip count is not cumulative")
        previous_skips = skips
        keys.append((-loss, dice, specificity, -threshold))
    selected_index = max(range(EPOCHS), key=lambda index: keys[index])
    selected_row = history[selected_index]
    best_epoch = int(selected_row["epoch"])
    best_threshold = float(selected_row["selected_threshold"])
    if int(metadata.get("best_epoch", -1)) != best_epoch:
        raise ValueError("X4 best epoch was not selected by the frozen rule")
    if not math.isclose(
        float(metadata.get("best_threshold", float("nan"))),
        best_threshold,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("X4 best threshold differs from the selected history row")
    if int(metadata.get("amp_skipped_steps", -1)) != previous_skips:
        raise ValueError("X4 final AMP skip count differs")

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    best_parameters = _validate_checkpoint(
        best_checkpoint,
        arm=arm,
        seed=seed,
        expected_epoch=best_epoch,
        expected_threshold=best_threshold,
        expected_inner_split_sha256=expected_inner_split_sha256,
        expected_target_freeze_sha256=expected_target_freeze_sha256,
        protocol_sha256=protocol_sha,
    )
    last_threshold = float(history[-1]["selected_threshold"])
    last_parameters = _validate_checkpoint(
        last_checkpoint,
        arm=arm,
        seed=seed,
        expected_epoch=EPOCHS,
        expected_threshold=last_threshold,
        expected_inner_split_sha256=expected_inner_split_sha256,
        expected_target_freeze_sha256=expected_target_freeze_sha256,
        protocol_sha256=protocol_sha,
    )
    if best_parameters != last_parameters:
        raise ValueError("X4 best/last checkpoint parameter counts differ")

    return {
        "schema_version": 1,
        "pass": True,
        "stage": "x4_matched_student_training_audit_v1",
        "arm": arm,
        "seed": seed,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "amp_skipped_steps": previous_skips,
        "parameters": best_parameters,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "inner_split_sha256": expected_inner_split_sha256,
        "x4_protocol_sha256": protocol_sha,
        "target_freeze_sha256": expected_target_freeze_sha256,
        "training_metadata_sha256": sha256_file(metadata_path),
        "training_history_sha256": sha256_file(history_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "outer_validation_images_used": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--arm", choices=STUDENT_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=STUDENT_SEEDS, required=True)
    parser.add_argument("--expected-inner-split-sha256", required=True)
    parser.add_argument("--expected-target-freeze-sha256")
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_training_output(
        args.output_root,
        repo_root=args.repo_root,
        arm=args.arm,
        seed=args.seed,
        expected_inner_split_sha256=args.expected_inner_split_sha256,
        expected_target_freeze_sha256=args.expected_target_freeze_sha256,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
