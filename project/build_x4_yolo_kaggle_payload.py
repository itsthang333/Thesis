from __future__ import annotations

"""Build and verify the minimal private/offline Kaggle payload for X4 YOLO."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE_FILES = (
    "project/__init__.py",
    # ``export_x4_yolo_dataset.py`` is intentionally executed as a script on
    # Kaggle.  In that entrypoint ``datasets.common`` falls back to importing
    # this module as top-level ``config``.  Keep it in the minimal payload;
    # otherwise export fails before reading data or starting the GPU job.
    "project/config.py",
    "project/run_x4_yolo_kaggle.py",
    "project/export_x4_yolo_dataset.py",
    "project/train_x4_yolov8s_seg.py",
    "project/frozen_io.py",
    "project/requirements-yolo.txt",
    "project/datasets/__init__.py",
    "project/datasets/btxrd.py",
    "project/datasets/common.py",
    "artifacts/final_pipeline/x4/x4_protocol.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size or sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"Payload copy differs: {source}")


def payload_rows(root: Path) -> list[dict[str, str | int]]:
    excluded = {"dataset-metadata.json", "payload_manifest.csv", "payload_build_report.json"}
    rows: list[dict[str, str | int]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            raise RuntimeError(f"Compiled Python artifact entered payload: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_manifest(root: Path, rows: list[dict[str, str | int]]) -> Path:
    manifest = root / "payload_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def verify_payload(root: Path) -> dict[str, object]:
    manifest = root / "payload_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_paths: set[str] = set()
    for row in rows:
        relative = str(row["path"])
        if relative in expected_paths:
            raise RuntimeError(f"Duplicate payload path: {relative}")
        expected_paths.add(relative)
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Payload manifest mismatch: {relative}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"dataset-metadata.json", "payload_manifest.csv", "payload_build_report.json"}
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            f"Payload file set differs: missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    return {
        "manifest_rows": len(rows),
        "payload_manifest_sha256": sha256_file(manifest),
        "exact_match": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--title", default="BTXRD X4 YOLOv8s-seg offline inputs")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_payload(args.output_dir), indent=2, sort_keys=True))
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.repo_root is None or args.split_manifest is None or args.runtime_root is None:
        raise ValueError("Build mode requires repo root, split manifest and runtime root")
    if not args.dataset_id or "/" not in args.dataset_id:
        raise ValueError("Build mode requires an owner/slug dataset id")
    args.output_dir.mkdir(parents=True)
    source_root = args.output_dir / "x4_yolo_source"
    for relative in SOURCE_FILES:
        copy_exact(args.repo_root / relative, source_root / relative)
    split_destination = args.output_dir / args.split_manifest.name
    copy_exact(args.split_manifest, split_destination)

    runtime_manifest_source = args.runtime_root / "runtime_manifest.json"
    runtime = json.loads(runtime_manifest_source.read_text(encoding="utf-8"))
    if runtime.get("schema_version") != 1 or runtime.get("ultralytics_version") != "8.4.0":
        raise RuntimeError("X4 YOLO runtime manifest differs")
    runtime_destination = args.output_dir / "runtime"
    copy_exact(runtime_manifest_source, runtime_destination / runtime_manifest_source.name)
    for record in runtime.get("files", []):
        source = args.runtime_root / str(record["name"])
        if source.stat().st_size != int(record["bytes"]) or sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"Runtime dependency differs: {source.name}")
        copy_exact(source, runtime_destination / source.name)

    metadata = {
        "id": args.dataset_id,
        "title": args.title,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (args.output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = payload_rows(args.output_dir)
    manifest = write_manifest(args.output_dir, rows)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    verification = verify_payload(args.output_dir)
    report = {
        "schema_version": 1,
        "stage": "x4_yolo_kaggle_payload_v2",
        "source_commit": commit,
        "source_files": list(SOURCE_FILES),
        "split_sha256": sha256_file(split_destination),
        "runtime_manifest_sha256": sha256_file(runtime_destination / "runtime_manifest.json"),
        **verification,
        "test_images_read": 0,
        "test_annotations_read": 0,
        "test_evaluated": False,
    }
    report_path = args.output_dir / "payload_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "payload_build_report_sha256": sha256_file(report_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
