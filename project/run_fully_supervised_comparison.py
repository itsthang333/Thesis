from __future__ import annotations

"""Train the locked fully-supervised comparison and score validation only."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from frozen_io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--resnet18-weight",
        type=Path,
        required=True,
        help="Exact torchvision resnet18-f37072fd.pth file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def run(command: list[str], env: dict[str, str]) -> None:
    print("[fully-supervised] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.resnet18_weight) != RESNET18_SHA256:
        raise ValueError("ResNet18 ImageNet weight SHA-256 mismatch")
    if args.resnet18_weight.name != "resnet18-f37072fd.pth":
        raise ValueError("expected the canonical resnet18-f37072fd.pth filename")
    checkpoint_dir = args.resnet18_weight.resolve().parent
    if checkpoint_dir.name != "checkpoints" or checkpoint_dir.parent.name != "hub":
        raise ValueError("place ResNet18 weight under TORCH_HOME/hub/checkpoints/")
    torch_home = checkpoint_dir.parent.parent
    env = dict(os.environ)
    env["TORCH_HOME"] = str(torch_home)
    training_dir = args.output_dir / "training"
    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "train_segmentation.py"),
            "--pipeline-profile", "btxrd_best",
            "--supervision-mode", "fully_supervised_comparison",
            "--data-root", str(args.dataset_root),
            "--split-manifest", str(args.split_manifest),
            "--train-split", "train",
            "--val-split", "val",
            "--image-size", "448",
            "--model-architecture", "resnet18_unet",
            "--batch-size", "8",
            "--lr", "0.0001",
            "--weight-decay", "0.0001",
            "--epochs", "30",
            "--seed", "42",
            "--num-workers", str(args.num_workers),
            "--early-stop-patience", "10",
            "--pos-weight-mode", "manual",
            "--pos-weight-value", "10",
            "--multi-gpu",
            "--output-dir", str(training_dir),
        ],
        env,
    )
    checkpoint = training_dir / "best_unet.pt"
    evaluations: dict[str, dict[str, object]] = {}
    for threshold, name in ((0.50, "fixed_0.50"), (0.20, "locked_0.20")):
        evaluation_dir = args.output_dir / f"validation_{name}"
        summary_path = evaluation_dir / "summary.json"
        run(
            [
                sys.executable,
                str(PROJECT_ROOT / "evaluate_unet.py"),
                "--data-root", str(args.dataset_root),
                "--split-manifest", str(args.split_manifest),
                "--split", "val",
                "--checkpoint", str(checkpoint),
                "--image-size", "448",
                "--batch-size", "8",
                "--num-workers", str(args.num_workers),
                "--threshold", str(threshold),
                "--output-csv", str(evaluation_dir / "per_image.csv"),
                "--output-json", str(summary_path),
                "--bootstrap-iterations", "10000",
                "--bootstrap-seed", "42",
            ],
            env,
        )
        evaluations[name] = {
            "threshold": threshold,
            "summary_sha256": sha256_file(summary_path),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }
    manifest = {
        "status": "fully_supervised_comparison_complete",
        "comparison_only": True,
        "wsss_eligible": False,
        "train_split": "train",
        "selection_split": "val",
        "test_evaluated": False,
        "scientific_config": {
            "architecture": "ResNet18UNet",
            "image_size": 448,
            "batch_size": 8,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "maximum_epochs": 30,
            "early_stop_patience": 10,
            "pos_weight": 10.0,
            "seed": 42,
            "final_test_threshold": 0.20,
        },
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "resnet18_weight_sha256": RESNET18_SHA256,
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "training_metadata_sha256": sha256_file(training_dir / "training_metadata.json"),
        "evaluations": evaluations,
    }
    manifest_path = args.output_dir / "fully_supervised_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "run_manifest_sha256": sha256_file(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
