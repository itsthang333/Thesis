from __future__ import annotations

"""Independent contract audit for the fixed-terminal X4 S2C generator."""

import argparse
import csv
import json
from pathlib import Path

import torch

from pseudo.manifest import sha256_file
from x4_contract import CANONICAL_SPLIT_SHA256, load_x4_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-train-cache-summary-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 S2C canonical split SHA-256 mismatch")
    _protocol, protocol_sha = load_x4_protocol(args.repo_root)
    metadata_path = args.output_root / "training_metadata.json"
    log_path = args.output_root / "training_log.csv"
    checkpoint_path = args.output_root / "last_s2c.pt"
    manifest_path = args.output_root / "run_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with log_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    required_metadata = {
        "method": "x4_cached_sam_s2c_style_wsss",
        "ground_truth_spatial_supervision": False,
        "annotation_files_opened": False,
        "outer_validation_images_opened": False,
        "validation_gt_accessed": False,
        "student_trained": False,
        "test_evaluated": False,
        "selection_rule": "fixed terminal epoch; no outer-validation selection",
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "train_segment_summary_sha256": args.expected_train_cache_summary_sha256,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": protocol_sha,
    }
    differences = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in required_metadata.items()
        if metadata.get(key) != value
    }
    if differences:
        raise ValueError(f"X4 S2C training metadata differs: {differences}")
    epochs = int(metadata["epochs"])
    if len(rows) != epochs or [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError("X4 S2C training log is not the complete fixed budget")
    if checkpoint.get("method") != required_metadata["method"]:
        raise ValueError("X4 S2C checkpoint method differs")
    if checkpoint.get("checkpoint_role") != "fixed_epoch_snapshot" or int(checkpoint.get("epoch", -1)) != epochs:
        raise ValueError("X4 S2C checkpoint is not terminal")
    if checkpoint.get("ground_truth_spatial_supervision") is not False:
        raise ValueError("X4 S2C checkpoint does not certify no spatial GT")
    if checkpoint.get("training_metadata") != metadata:
        raise ValueError("X4 S2C checkpoint metadata differs from sidecar")
    if manifest.get("training_metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("X4 S2C metadata SHA differs")
    if manifest.get("training_log_sha256") != sha256_file(log_path):
        raise ValueError("X4 S2C log SHA differs")
    if manifest.get("last_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("X4 S2C checkpoint SHA differs")
    if int(manifest.get("epochs_completed", -1)) != epochs or manifest.get("outer_validation_selection") is not False:
        raise ValueError("X4 S2C run manifest violates terminal protocol")

    result = {
        "schema_version": 1,
        "stage": "independent_x4_s2c_training_output_audit_v1",
        "status": "pass",
        "source_commit": args.expected_source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "train_cache_summary_sha256": args.expected_train_cache_summary_sha256,
        "training_metadata_sha256": sha256_file(metadata_path),
        "training_log_sha256": sha256_file(log_path),
        "terminal_checkpoint_sha256": sha256_file(checkpoint_path),
        "epochs": epochs,
        "outer_validation_selection": False,
        "ground_truth_spatial_supervision": False,
        "student_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "audit_sha256": sha256_file(args.audit_output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
