from __future__ import annotations

"""Run a matched three-seed G4 E1 classifier arm without spatial GT."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("binary", "ten_class"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=cwd, env=os.environ.copy(), check=True)


def main() -> None:
    args = parse_args()
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("split manifest hash mismatch")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seeds must be a non-empty unique list")
    project = Path(__file__).resolve().parent
    source_root = project.parent
    target = "tumor" if args.arm == "binary" else "tumor_type"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in seeds:
        seed_root = args.output_dir / f"seed_{seed}"
        training_root = seed_root / "training"
        run([
            sys.executable,
            str(project / "train_classifier.py"),
            "--data-root", str(args.data_root),
            "--split-manifest", str(args.split_manifest),
            "--target-columns", target,
            "--image-size", "320",
            "--batch-size", "4",
            "--lr", "0.0001",
            "--weight-decay", "0.0001",
            "--epochs", str(args.epochs),
            "--early-stop-patience", str(args.early_stop_patience),
            "--checkpoint-selection-metric", "binary_f1",
            "--seed", str(seed),
            "--num-workers", str(args.num_workers),
            "--output-dir", str(training_root),
        ], source_root)
        checkpoint = training_root / "best_classifier.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        evaluation_root = seed_root / "label_safe_evaluation"
        run([
            sys.executable,
            str(project / "evaluate_g4_classifier_labels.py"),
            "--data-root", str(args.data_root),
            "--split-manifest", str(args.split_manifest),
            "--expected-split-sha256", args.expected_split_sha256,
            "--checkpoint", str(checkpoint),
            "--output-dir", str(evaluation_root),
            "--batch-size", "16",
            "--num-workers", str(args.num_workers),
        ], source_root)
        evaluation = json.loads((evaluation_root / "summary.json").read_text(encoding="utf-8"))
        records.append({
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": evaluation["checkpoint_epoch"],
            "binary_endpoint": evaluation["binary_endpoint"],
            "evaluation_summary_sha256": sha256_file(evaluation_root / "summary.json"),
        })
    metrics = ("f1", "auroc", "average_precision_auprc", "brier_score", "ece_15_equal_width")
    aggregate = {}
    for name in metrics:
        values = [float(row["binary_endpoint"][name]) for row in records]
        aggregate[name] = {
            "mean": sum(values) / len(values),
            "sample_std": (
                (sum((value - sum(values) / len(values)) ** 2 for value in values) / (len(values) - 1)) ** 0.5
                if len(values) > 1 else 0.0
            ),
            "values": values,
        }
    summary = {
        "stage": "g4_e1_label_granularity_training_v1",
        "arm": args.arm,
        "target_columns": [target],
        "seeds": seeds,
        "fixed_protocol": {
            "image_size": 320,
            "batch_size": 4,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "maximum_epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "checkpoint_selection": "validation binary_f1",
            "augmentation": False,
            "initialization": "ImageNet DenseNet121",
        },
        "split_manifest_sha256": args.expected_split_sha256,
        "runs": records,
        "aggregate": aggregate,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "arm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
