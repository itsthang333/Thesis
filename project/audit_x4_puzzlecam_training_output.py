from __future__ import annotations

"""Audit the fixed-terminal clean binary PuzzleCAM X4 generator run."""

import argparse
import csv
import json
from pathlib import Path

import torch

from pseudo.manifest import sha256_file
from x4_contract import CANONICAL_SPLIT_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--expected-puzzle-sha256", required=True)
    parser.add_argument("--expected-imagenet-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.output_root / "x4_puzzlecam_training_receipt.json"
    metadata_path = args.output_root / "training_metadata.json"
    log_path = args.output_root / "training_log.csv"
    checkpoint_path = args.output_root / "last_classifier.pt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with log_path.open("r", newline="", encoding="utf-8-sig") as handle:
        log = list(csv.DictReader(handle))
    required_receipt = {
        "schema_version": 1,
        "stage": "x4_binary_puzzlecam_training_v1",
        "source_commit": args.expected_source_commit,
        "trainer_sha256": args.expected_trainer_sha256,
        "puzzle_cam_sha256": args.expected_puzzle_sha256,
        "imagenet_weight_sha256": args.expected_imagenet_sha256,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "terminal_epoch": 30,
        "puzzle_alpha_max": 4.0,
        "train_spatial_annotations_read": 0,
        "outer_validation_spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {k: {"actual": receipt.get(k), "expected": v}
                   for k, v in required_receipt.items() if receipt.get(k) != v}
    if differences:
        raise ValueError(f"PuzzleCAM receipt differs: {differences}")
    expected_metadata = {
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "target_columns": ["tumor"],
        "image_size": 320,
        "batch_size": 4,
        "epochs": 30,
        "seed": 42,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "puzzle_alpha_max": 4.0,
        "attention_alpha_max": 0.0,
        "normalization": "imagenet",
        "sam_segment_contrastive": None,
    }
    differences = {k: {"actual": metadata.get(k), "expected": v}
                   for k, v in expected_metadata.items() if metadata.get(k) != v}
    if differences:
        raise ValueError(f"PuzzleCAM training metadata differs: {differences}")
    if len(log) != 30 or [int(row["epoch"]) for row in log] != list(range(1, 31)):
        raise ValueError("PuzzleCAM training log is not the complete 30-epoch budget")
    puzzle = checkpoint.get("puzzle_cam")
    if (
        int(checkpoint.get("epoch", -1)) != 30
        or checkpoint.get("seed") != 42
        or checkpoint.get("split_manifest_sha256") != CANONICAL_SPLIT_SHA256
        or checkpoint.get("target_columns") != ["tumor"]
        or checkpoint.get("image_size") != 320
        or not isinstance(puzzle, dict)
        or puzzle.get("method") != "PuzzleCAM reconstruction consistency"
        or puzzle.get("task") != "binary"
        or puzzle.get("alpha_max") != 4.0
    ):
        raise ValueError("PuzzleCAM terminal checkpoint contract differs")
    checkpoint_sha = sha256_file(checkpoint_path)
    if receipt.get("terminal_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("PuzzleCAM terminal checkpoint SHA differs")
    result = {
        "schema_version": 1,
        "stage": "independent_x4_puzzlecam_training_audit_v1",
        "status": "pass",
        "source_commit": args.expected_source_commit,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "receipt_sha256": sha256_file(receipt_path),
        "training_metadata_sha256": sha256_file(metadata_path),
        "training_log_sha256": sha256_file(log_path),
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_epoch": 30,
        "outer_validation_spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "audit_sha256": sha256_file(args.audit_output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
