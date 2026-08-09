from __future__ import annotations

"""Offline Kaggle Stage-A/Stage-B runner for completed X4 YOLO bundles."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from frozen_io import sha256_file, validate_sha256
from run_x4_yolo_kaggle import locate_dataset_root


def exact_file(root: Path, name: str, expected_sha256: str) -> Path:
    expected = validate_sha256(expected_sha256, name=f"{name} SHA-256")
    matches = [
        path for path in root.rglob(name) if path.is_file() and sha256_file(path) == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise RuntimeError(f"Unsafe training archive member: {member.filename}")
        handle.extractall(destination)


def validate_training_receipt(
    receipt: dict[str, object],
    *,
    seed: int,
    split_sha256: str,
    archive_sha256: str,
) -> None:
    required = {
        "stage": "x4_yolov8s_seg_kaggle_wrapper_v1",
        "seed": seed,
        "split_sha256": split_sha256,
        "training_archive_sha256": archive_sha256,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    for key in ("best_checkpoint_sha256", "training_report_sha256"):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64:
            differences[key] = {"actual": value, "expected": "SHA-256"}
    if differences:
        raise RuntimeError(f"X4 YOLO training receipt differs: {differences}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--runtime-manifest-name", default="runtime_manifest.json")
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--split-manifest-name", default="canonical_split_manifest_85511.csv")
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--training-bundle-name", required=True)
    parser.add_argument("--expected-training-bundle-sha256", required=True)
    parser.add_argument("--training-receipt-name", required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--expected-freeze-runner-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_sha = validate_sha256(args.expected_split_sha256, name="split SHA-256")
    bundle_sha = validate_sha256(
        args.expected_training_bundle_sha256, name="training bundle SHA-256"
    )
    runtime_manifest = exact_file(
        args.input_root,
        args.runtime_manifest_name,
        args.expected_runtime_manifest_sha256,
    )
    split_manifest = exact_file(args.input_root, args.split_manifest_name, split_sha)
    bundle = exact_file(args.input_root, args.training_bundle_name, bundle_sha)
    receipt_path = exact_file(
        args.input_root,
        args.training_receipt_name,
        args.expected_training_receipt_sha256,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_training_receipt(
        receipt, seed=args.seed, split_sha256=split_sha, archive_sha256=bundle_sha
    )

    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    if runtime.get("schema_version") != 1 or runtime.get("ultralytics_version") != "8.4.0":
        raise RuntimeError("X4 YOLO runtime contract differs")
    runtime_root = runtime_manifest.parent
    wheels: list[Path] = []
    for record in runtime.get("files", []):
        path = runtime_root / str(record["name"])
        if (
            not path.is_file()
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != int(record["bytes"])
        ):
            raise RuntimeError(f"X4 YOLO runtime file differs: {path.name}")
        if path.suffix == ".whl":
            wheels.append(path)
    if not wheels:
        raise RuntimeError("X4 YOLO runtime contains no wheels")

    package_root = Path("/kaggle/working/x4_yolo_eval_packages")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(package_root),
            *[str(path) for path in wheels],
        ],
        check=True,
    )
    training_extract = Path("/kaggle/working/x4_yolo_training")
    safe_extract(bundle, training_extract)
    training_report = exact_file(
        training_extract, "training_report.json", str(receipt["training_report_sha256"])
    )
    training_root = training_report.parent
    checkpoint = training_root / "best.pt"
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != str(receipt["best_checkpoint_sha256"])
    ):
        raise RuntimeError("Top-level X4 YOLO best checkpoint differs")
    dataset_root = locate_dataset_root(args.input_root)

    freeze_runner = exact_file(
        args.input_root,
        "freeze_x4_yolo_predictions.py",
        args.expected_freeze_runner_sha256,
    )
    evaluator = exact_file(
        args.input_root,
        "evaluate_x4_yolo_predictions.py",
        args.expected_evaluator_sha256,
    )
    if freeze_runner.parent != evaluator.parent:
        raise RuntimeError("X4 YOLO freeze/evaluator source roots differ")
    source_root = freeze_runner.parent.parent
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(package_root), str(source_root / "project"), str(source_root))
            ),
            "PYTHONHASHSEED": str(args.seed),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPYCACHEPREFIX": "/kaggle/working/pycache",
            "YOLO_CONFIG_DIR": "/kaggle/working/ultralytics_config",
            "MPLCONFIGDIR": "/kaggle/working/matplotlib_config",
            "WANDB_DISABLED": "true",
            "MALLOC_ARENA_MAX": "2",
        }
    )
    prediction_root = Path(f"/kaggle/working/{args.output_prefix}_predictions")
    evaluation_root = Path(f"/kaggle/working/{args.output_prefix}_evaluation")
    freeze_command = [
        sys.executable,
        str(freeze_runner),
        "--dataset-root",
        str(dataset_root),
        "--split-manifest",
        str(split_manifest),
        "--expected-split-sha256",
        split_sha,
        "--training-root",
        str(training_root),
        "--expected-training-report-sha256",
        str(receipt["training_report_sha256"]),
        "--expected-checkpoint-sha256",
        str(receipt["best_checkpoint_sha256"]),
        "--output-dir",
        str(prediction_root),
        "--device",
        args.device,
        "--batch",
        str(args.batch),
    ]
    print(json.dumps({"freeze_command": freeze_command}), flush=True)
    subprocess.run(freeze_command, cwd=source_root, env=environment, check=True)
    freeze_path = prediction_root / "prediction_freeze.json"
    freeze_sha = sha256_file(freeze_path)
    evaluation_command = [
        sys.executable,
        str(evaluator),
        "--dataset-root",
        str(dataset_root),
        "--split-manifest",
        str(split_manifest),
        "--expected-split-sha256",
        split_sha,
        "--prediction-root",
        str(prediction_root),
        "--expected-prediction-freeze-sha256",
        freeze_sha,
        "--training-root",
        str(training_root),
        "--expected-training-report-sha256",
        str(receipt["training_report_sha256"]),
        "--output-dir",
        str(evaluation_root),
    ]
    print(json.dumps({"evaluation_command": evaluation_command}), flush=True)
    subprocess.run(evaluation_command, cwd=source_root, env=environment, check=True)

    output_archive = Path(f"/kaggle/working/{args.output_prefix}_evaluated_bundle.zip")
    staging = Path("/kaggle/working/x4_yolo_evaluated_staging")
    staging.mkdir()
    shutil.copytree(prediction_root, staging / "predictions")
    shutil.copytree(evaluation_root, staging / "evaluation")
    shutil.make_archive(str(output_archive.with_suffix("")), "zip", staging)
    evaluation_report = evaluation_root / "evaluation_report.json"
    terminal = {
        "schema_version": 1,
        "stage": "x4_yolo_kaggle_evaluation_v1",
        "seed": args.seed,
        "split_sha256": split_sha,
        "training_bundle_sha256": bundle_sha,
        "training_report_sha256": receipt["training_report_sha256"],
        "checkpoint_sha256": receipt["best_checkpoint_sha256"],
        "prediction_freeze_sha256": freeze_sha,
        "evaluation_report_sha256": sha256_file(evaluation_report),
        "evaluated_bundle_sha256": sha256_file(output_archive),
        "images": 371,
        "tumor_images": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    terminal_path = Path(f"/kaggle/working/{args.output_prefix}_evaluation_receipt.json")
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
