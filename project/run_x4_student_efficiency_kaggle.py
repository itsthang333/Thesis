from __future__ import annotations

"""Kaggle wrapper for the X4 same-GPU student inference benchmark.

The scientific prediction freezer is reused unchanged.  This wrapper only
resolves an audited checkpoint, runs the 371-image annotation-free freeze on a
T4, retains the X12 timing/memory/storage report, and deletes the redundant
mask archive after its manifest has been bound by hash.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(root: Path, name: str, expected: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact {name}, found {matches}")
    return matches[0]


def source_project(root: Path) -> Path:
    matches = sorted(
        path.parent
        for path in root.rglob("x4_contract.py")
        if path.is_file() and (path.parent / "models" / "unet.py").is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one complete X4 source project, found {matches}")
    return matches[0]


def dataset_root(root: Path) -> Path:
    matches = sorted(
        {
            path.parent
            for path in root.rglob("images")
            if path.is_dir() and (path.parent / "Annotations").is_dir()
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {matches}")
    return matches[0]


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    resolved_root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved_root and resolved_root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def validate_efficiency_freeze(
    freeze: dict[str, object], *, arm: str, seed: int, checkpoint_sha256: str
) -> dict[str, object]:
    required = {
        "stage": "x4_student_prediction_freeze_v1",
        "arm": arm,
        "seed": seed,
        "split": "val",
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "predictions_frozen_before_spatial_ground_truth": True,
        "spatial_ground_truth_used": False,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": freeze.get(key), "expected": value}
        for key, value in required.items()
        if freeze.get(key) != value
    }
    if differences:
        raise RuntimeError(f"efficiency freeze differs: {differences}")
    efficiency = freeze.get("x12_efficiency")
    if not isinstance(efficiency, dict):
        raise RuntimeError("X12 efficiency evidence is missing")
    if (
        efficiency.get("stage") != "matched_student_online_inference_and_freeze"
        or int(efficiency.get("timed_images", -1)) != 371
        or int(efficiency.get("warmup_iterations", -1)) != 3
        or efficiency.get("offline_pseudo_label_generation_included") is not False
    ):
        raise RuntimeError("X12 efficiency contract differs")
    devices = efficiency.get("device_memory")
    if not isinstance(devices, list) or not devices:
        raise RuntimeError("X12 CUDA memory evidence is missing")
    if not all("T4" in str(device.get("device_name", "")) for device in devices):
        raise RuntimeError(f"X12 benchmark is not on T4: {devices}")
    return efficiency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--freezer-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--training-archive-name")
    parser.add_argument("--training-archive-sha256")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(
        {
            "PYTHONHASHSEED": str(args.seed),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    freezer = exact_file(INPUT, "freeze_x4_student_predictions.py", args.freezer_sha256)
    split = exact_file(INPUT, "canonical_split_manifest_85511.csv", SPLIT_SHA256)
    project = source_project(INPUT)
    os.environ["PYTHONPATH"] = str(project)

    extracted: Path | None = None
    if args.checkpoint_name:
        checkpoint = exact_file(INPUT, args.checkpoint_name, args.checkpoint_sha256)
    else:
        if not args.training_archive_name or not args.training_archive_sha256:
            raise ValueError("checkpoint name or exact training archive is required")
        archive = exact_file(INPUT, args.training_archive_name, args.training_archive_sha256)
        extracted = WORKING / "training_extract"
        safe_extract(archive, extracted)
        candidates = [
            path for path in extracted.rglob("*.pt") if path.is_file() and sha256(path) == args.checkpoint_sha256
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one extracted checkpoint, found {candidates}")
        checkpoint = candidates[0]

    output = WORKING / f"x4_{args.arm}_student_seed{args.seed}_efficiency_predictions"
    command = [
        sys.executable,
        str(freezer),
        "--arm",
        args.arm,
        "--seed",
        str(args.seed),
        "--dataset-root",
        str(dataset_root(INPUT)),
        "--split-manifest",
        str(split),
        "--checkpoint",
        str(checkpoint),
        "--expected-checkpoint-sha256",
        args.checkpoint_sha256,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        "0",
        "--output-dir",
        str(output),
    ]
    print(json.dumps({"command": command, "pythonpath": str(project)}), flush=True)
    subprocess.run(command, cwd=freezer.parent.parent, env=os.environ.copy(), check=True)

    freeze_path = output / "prediction_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    efficiency = validate_efficiency_freeze(
        freeze, arm=args.arm, seed=args.seed, checkpoint_sha256=args.checkpoint_sha256
    )
    report = {
        "schema_version": 1,
        "stage": "x4_student_efficiency_benchmark_v1",
        "arm": args.arm,
        "seed": args.seed,
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": args.checkpoint_sha256,
        "prediction_freeze_sha256": sha256(freeze_path),
        "prediction_manifest_sha256": freeze["prediction_manifest_sha256"],
        "x4_protocol_sha256": freeze["x4_protocol_sha256"],
        "x12_efficiency": efficiency,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    report_path = WORKING / f"x4_{args.arm}_seed{args.seed}_efficiency.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(output)
    if extracted is not None:
        shutil.rmtree(extracted)
    print(json.dumps({**report, "report_sha256": sha256(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
