from __future__ import annotations

"""Reproduce the frozen 448px binary classifier using image labels only.

This bounded supply stage is intentionally separate from spatial validation.
It never imports a segmentation evaluator and never enumerates BTXRD
``Annotations`` or test images.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_label_counts(split_manifest: Path) -> dict[str, dict[str, int]]:
    counts = {
        "train": {"images": 0, "tumor": 0, "normal": 0},
        "val": {"images": 0, "tumor": 0, "normal": 0},
    }
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split", ""))
            if split not in counts or str(row.get("eligible", "")) != "1":
                continue
            label = int(row["tumor"])
            if label not in (0, 1):
                raise ValueError("Non-binary tumor image label")
            counts[split]["images"] += 1
            counts[split]["tumor"] += label
            counts[split]["normal"] += 1 - label
    expected = {
        "train": {"images": 2981, "tumor": 1488, "normal": 1493},
        "val": {"images": 371, "tumor": 184, "normal": 187},
    }
    if counts != expected:
        raise ValueError(f"Canonical train/validation counts differ: {counts}")
    return counts


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--densenet-weight", type=Path, required=True)
    parser.add_argument("--expected-densenet-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    project = source_root / "project"
    if not (project / "train_classifier.py").is_file():
        raise FileNotFoundError("Source snapshot does not contain train_classifier.py")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Classifier supply output must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("Split manifest SHA-256 mismatch")
    if sha256_file(args.densenet_weight) != args.expected_densenet_sha256:
        raise ValueError("DenseNet121 weight SHA-256 mismatch")
    counts = load_image_label_counts(args.split_manifest)

    torch_home = args.output_dir / "torch_home"
    checkpoint_cache = torch_home / "hub" / "checkpoints"
    checkpoint_cache.mkdir(parents=True)
    cached_weight = checkpoint_cache / "densenet121-a639ec97.pth"
    shutil.copy2(args.densenet_weight, cached_weight)
    if sha256_file(cached_weight) != args.expected_densenet_sha256:
        raise RuntimeError("Copied DenseNet121 weight differs")

    training = args.output_dir / "training"
    evaluation = args.output_dir / "evaluation"
    log = args.output_dir / "kernel.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": str(torch_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    started = datetime.now(timezone.utc)
    train_command = [
        sys.executable,
        str(project / "train_classifier.py"),
        "--pipeline-profile",
        "default",
        "--data-root",
        str(args.data_root),
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--split-manifest",
        str(args.split_manifest),
        "--target-columns",
        "tumor",
        "--image-size",
        "448",
        "--batch-size",
        "8",
        "--lr",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--epochs",
        "30",
        "--seed",
        "42",
        "--num-workers",
        "2",
        "--early-stop-patience",
        "7",
        "--output-dir",
        str(training),
    ]
    run(train_command, cwd=source_root, env=env, log=log)
    checkpoint = training / "best_classifier.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Training did not emit best_classifier.pt")
    evaluation.mkdir()
    evaluate_command = [
        sys.executable,
        str(project / "evaluate_classifier.py"),
        "--data-root",
        str(args.data_root),
        "--split",
        "val",
        "--split-manifest",
        str(args.split_manifest),
        "--checkpoint",
        str(checkpoint),
        "--image-size",
        "448",
        "--batch-size",
        "8",
        "--num-workers",
        "2",
        "--gate-rule",
        "probability",
        "--gate-threshold",
        "0.5",
        "--bootstrap-iterations",
        "2000",
        "--bootstrap-seed",
        "42",
        "--output-dir",
        str(evaluation),
    ]
    run(evaluate_command, cwd=source_root, env=env, log=log)
    summary_path = evaluation / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("images", -1)) != 371:
        raise RuntimeError("Classifier validation cohort differs")

    manifest = {
        "schema_version": 1,
        "stage": "classifier448_image_label_only_supply",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "densenet_weight_sha256": args.expected_densenet_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_log_sha256": sha256_file(training / "training_log.csv"),
        "evaluation_summary_sha256": sha256_file(summary_path),
        "counts": counts,
        "validation_gate": summary.get("gate_tumor_vs_normal"),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    manifest_path = args.output_dir / "classifier448_supply_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
