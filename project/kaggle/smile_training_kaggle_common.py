from __future__ import annotations

"""Minimal private/offline Kaggle launcher for a frozen SMILE arm."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SPLIT_SHA256 = "7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c"
REFERENCE_SHA256 = "c37561eec0fcffa67d99d1650720557260531b934d3bc87aec8fd780c9a34085"
DENSENET_SHA256 = "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395"
PROTOCOL_SHA256 = "b79aa0c42b694d6fe7986e74be062296e424100513a5ad188444b38a20a73af6"
# Filled after the source manifest is frozen; deliberately outside that manifest.
SOURCE_SHA256 = "d8b58ffaa932d8d0739f4ffee9929f39ecce1306b9d60e5e67cc0971eab70844"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_unique(name: str) -> Path:
    matches = sorted(path for path in INPUT.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}: {matches}")
    return matches[0]


def find_dataset_root() -> Path:
    roots = sorted(
        set(path.parent for path in INPUT.rglob("images") if path.is_dir() and path.parent.name == "BTXRD")
    )
    if len(roots) != 1:
        raise FileNotFoundError(f"Cannot resolve one BTXRD root: {roots}")
    return roots[0]


def verify_source() -> tuple[Path, dict[str, object]]:
    runner = find_unique("train_smile_rich_gallery.py")
    source_root = runner.parent.parent
    manifest_path = find_unique("SMILE_SOURCE_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("combined_sha256") != SOURCE_SHA256:
        raise ValueError("SMILE source manifest identity mismatch")
    for relative, expected in manifest.get("files", {}).items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"SMILE source mismatch: {relative}")
    return source_root, manifest


def run_arm(arm: str) -> None:
    if arm not in {"control", "full"}:
        raise ValueError("arm must be control or full")
    os.environ.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONPYCACHEPREFIX": "/kaggle/working/pycache",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    source_root, _ = verify_source()
    split = find_unique("split_manifest.csv")
    references = find_unique("REFERENCE_ASSIGNMENTS.csv")
    weights = find_unique("densenet121-a639ec97.pth")
    protocol = find_unique("smile_rich_gallery_v1.json")
    locks = (
        (split, SPLIT_SHA256),
        (references, REFERENCE_SHA256),
        (weights, DENSENET_SHA256),
        (protocol, PROTOCOL_SHA256),
    )
    for path, expected in locks:
        if sha256_file(path) != expected:
            raise ValueError(f"Input lock mismatch: {path.name}")
    data_root = find_dataset_root()
    output = WORKING / f"smile_{arm}_training"
    command = [
        sys.executable,
        str(source_root / "project" / "train_smile_rich_gallery.py"),
        "--arm", arm,
        "--dataset-root", str(data_root),
        "--split-manifest", str(split),
        "--split-sha256", SPLIT_SHA256,
        "--reference-manifest", str(references),
        "--reference-sha256", REFERENCE_SHA256,
        "--densenet-weights", str(weights),
        "--densenet-sha256", DENSENET_SHA256,
        "--protocol-sha256", PROTOCOL_SHA256,
        "--source-sha256", SOURCE_SHA256,
        "--output-dir", str(output),
        "--device", "cuda",
    ]
    print(json.dumps({"arm": arm, "command": command}), flush=True)
    subprocess.run(command, cwd=source_root, check=True, env=os.environ.copy())
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("arm") != arm
        or summary.get("global_step") != 2986
        or summary.get("terminal_epoch") != 1
        or summary.get("spatial_ground_truth_used") is not False
        or summary.get("test_images_read") != 0
        or summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("SMILE terminal output contract mismatch")
    print(json.dumps({"complete": True, "arm": arm, "summary": summary}, sort_keys=True), flush=True)
