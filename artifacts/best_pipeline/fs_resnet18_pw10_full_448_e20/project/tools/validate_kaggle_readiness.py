from __future__ import annotations

"""Fail-fast preflight for a locked, Internet-off Kaggle pipeline run."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--unet-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_requirements(path: Path) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or " @ " in line:
            continue
        if "==" not in line:
            raise ValueError(f"Unpinned requirement in {path}: {line}")
        name, version = line.split("==", 1)
        pinned[name.lower()] = version
    return pinned


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        dirty_lines = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    except Exception as exc:
        commit, branch, dirty_lines = "unknown", "unknown", []
        failures.append(f"Cannot read git provenance: {exc}")
    details["git"] = {"branch": branch, "commit": commit, "dirty": bool(dirty_lines), "dirty_files": dirty_lines}
    if branch != "pipeline":
        failures.append(f"Expected branch 'pipeline', got {branch!r}")
    if args.expected_commit and commit != args.expected_commit:
        failures.append(f"Commit mismatch: expected {args.expected_commit}, got {commit}")
    if dirty_lines and not args.allow_dirty:
        failures.append("Working tree is dirty; a final Kaggle run must use a committed snapshot")

    requirements_path = ROOT / "project" / "requirements.txt"
    expected_versions = pinned_requirements(requirements_path)
    installed: dict[str, str | None] = {}
    for package, expected in expected_versions.items():
        distribution = "opencv-python-headless" if package == "opencv-python-headless" else package
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            # Kaggle can provide opencv-python rather than the headless wheel;
            # still report it as a mismatch instead of silently accepting ABI drift.
            actual = None
        installed[package] = actual
        if actual != expected:
            failures.append(f"Dependency {package}: expected {expected}, installed {actual}")
    try:
        sam_version = importlib.metadata.version("segment-anything")
    except importlib.metadata.PackageNotFoundError:
        sam_version = None
        failures.append("segment-anything is not installed from the pinned local/offline source")
    details["dependencies"] = {"expected": expected_versions, "installed": installed, "segment_anything": sam_version}

    dataset_root = args.dataset_root.resolve()
    required_dataset_paths = [dataset_root / "images", dataset_root / "Annotations"]
    if not any((dataset_root / name).is_file() for name in ("dataset.csv", "dataset.xlsx")):
        failures.append(f"Dataset metadata dataset.csv/dataset.xlsx is missing under {dataset_root}")
    for path in required_dataset_paths:
        if not path.is_dir():
            failures.append(f"Dataset directory is missing: {path}")
    if not args.split_manifest.is_file():
        failures.append(f"Split manifest is missing: {args.split_manifest}")
        split_hash = None
    else:
        split_hash = sha256_file(args.split_manifest)
    details["dataset"] = {"root": str(dataset_root), "split_manifest_sha256": split_hash}

    checkpoint_details: dict[str, object] = {}
    for label, path in (
        ("classifier", args.classifier_checkpoint),
        ("sam", args.sam_checkpoint),
        ("unet", args.unet_checkpoint),
    ):
        if path is None:
            checkpoint_details[label] = None
            continue
        if not path.is_file():
            failures.append(f"{label} checkpoint is missing: {path}")
            checkpoint_details[label] = {"path": str(path), "sha256": None}
        else:
            checkpoint_details[label] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    details["checkpoints"] = checkpoint_details

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.allow_existing_output:
        failures.append(f"Output root is non-empty and overwrite was not authorised: {output_root}")
    disk_anchor = output_root if output_root.exists() else output_root.parent
    disk_anchor.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(disk_anchor).free / (1024 ** 3)
    details["storage"] = {"output_root": str(output_root), "free_gb": free_gb, "minimum_free_gb": args.min_free_gb}
    if free_gb < args.min_free_gb:
        failures.append(f"Insufficient disk: {free_gb:.1f} GiB free, need {args.min_free_gb:.1f} GiB")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception as exc:
        cuda_available, gpu_names, torch_version, cuda_version = False, [], None, None
        failures.append(f"PyTorch/CUDA probe failed: {exc}")
    details["runtime"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "torch_cuda": cuda_version,
        "cuda_available": cuda_available,
        "gpus": gpu_names,
    }
    if not cuda_available:
        failures.append("CUDA GPU is not available; the full classifier+SAM+U-Net run is not feasible")

    report = {"ready": not failures, "failures": failures, "warnings": warnings, "details": details}
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
