from __future__ import annotations

"""Offline Kaggle orchestrator for the X4 YOLOv8s-seg upper bound."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(root: Path, name: str, expected: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and sha256_file(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def locate_dataset_root(input_root: Path) -> Path:
    roots = sorted(
        {
            path.parent.resolve()
            for path in input_root.rglob("images")
            if path.is_dir() and (path.parent / "Annotations").is_dir()
        }
    )
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one BTXRD dataset root, found {roots}")
    return roots[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--split-manifest-name", default="canonical_split_manifest_85511.csv")
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pretrained-name", default="yolov8s-seg.pt")
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-prefix", default="x4_yolov8s_seg_seed42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if sha256_file(args.runtime_manifest) != args.expected_runtime_manifest_sha256:
        raise RuntimeError("X4 YOLO runtime manifest changed")
    runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    if runtime.get("schema_version") != 1 or runtime.get("ultralytics_version") != "8.4.0":
        raise RuntimeError("X4 YOLO runtime contract differs")
    runtime_root = args.runtime_manifest.parent
    wheels: list[Path] = []
    for record in runtime.get("files", []):
        path = runtime_root / str(record["name"])
        if sha256_file(path) != record["sha256"] or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"X4 YOLO runtime file differs: {path.name}")
        if path.suffix == ".whl":
            wheels.append(path)
    if not wheels:
        raise RuntimeError("X4 YOLO runtime contains no wheels")
    pretrained = exact_file(
        args.input_root, args.pretrained_name, args.expected_pretrained_sha256
    )
    split = exact_file(args.input_root, args.split_manifest_name, args.expected_split_sha256)
    dataset_root = locate_dataset_root(args.input_root)

    package_root = Path("/kaggle/working/x4_yolo_packages")
    package_root.mkdir(parents=True, exist_ok=False)
    install = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--target",
        str(package_root),
        *[str(path) for path in wheels],
    ]
    print(json.dumps({"offline_install": [Path(item).name if item.endswith(".whl") else item for item in install]}), flush=True)
    subprocess.run(install, check=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(package_root) + os.pathsep + environment.get("PYTHONPATH", ""),
            "PYTHONHASHSEED": str(args.seed),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPYCACHEPREFIX": "/kaggle/working/pycache",
            "YOLO_CONFIG_DIR": "/kaggle/working/ultralytics_config",
            "MPLCONFIGDIR": "/kaggle/working/matplotlib_config",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "MALLOC_ARENA_MAX": "2",
        }
    )
    project = Path(__file__).resolve().parent
    export_root = Path("/kaggle/working/x4_yolo_export")
    export_command = [
        sys.executable,
        str(project / "export_x4_yolo_dataset.py"),
        "--dataset-root",
        str(dataset_root),
        "--split-manifest",
        str(split),
        "--expected-split-sha256",
        args.expected_split_sha256,
        "--output-dir",
        str(export_root),
        "--image-mode",
        "symlink",
    ]
    print(json.dumps({"export_command": export_command}), flush=True)
    subprocess.run(export_command, cwd=project.parent, env=environment, check=True)
    export_report = export_root / "export_report.json"
    training_root = Path(f"/kaggle/working/{args.output_prefix}_training")
    train_command = [
        sys.executable,
        str(project / "train_x4_yolov8s_seg.py"),
        "--dataset-root",
        str(export_root),
        "--expected-export-report-sha256",
        sha256_file(export_report),
        "--pretrained-checkpoint",
        str(pretrained),
        "--expected-pretrained-sha256",
        args.expected_pretrained_sha256,
        "--output-dir",
        str(training_root),
        "--seed",
        str(args.seed),
        "--epochs",
        "300",
        "--imgsz",
        "600",
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
    ]
    print(json.dumps({"train_command": train_command}), flush=True)
    subprocess.run(train_command, cwd=project.parent, env=environment, check=True)
    report_path = training_root / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("stage") != "x4_yolov8s_seg_training_v1"
        or report.get("epochs_completed") != 300
        or report.get("imgsz") != 600
        or report.get("seed") != args.seed
        or report.get("test_images_read") != 0
        or report.get("test_evaluated") is not False
    ):
        raise RuntimeError("X4 YOLO terminal report violates the protocol")
    archive = Path(
        shutil.make_archive(
            f"/kaggle/working/{args.output_prefix}_training_bundle",
            "zip",
            root_dir="/kaggle/working",
            base_dir=training_root.name,
        )
    )
    receipt = {
        "schema_version": 1,
        "stage": "x4_yolov8s_seg_kaggle_wrapper_v1",
        "seed": args.seed,
        "ultralytics_version": "8.4.0",
        "split_sha256": args.expected_split_sha256,
        "pretrained_sha256": args.expected_pretrained_sha256,
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "training_report_sha256": sha256_file(report_path),
        "best_checkpoint_sha256": report["best_checkpoint_sha256"],
        "last_checkpoint_sha256": report["last_checkpoint_sha256"],
        "training_archive_sha256": sha256_file(archive),
        "native_ultralytics_validation": report["native_ultralytics_validation"],
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt_path = Path(f"/kaggle/working/{args.output_prefix}_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(export_root)
    shutil.rmtree(training_root)
    shutil.rmtree(package_root)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
