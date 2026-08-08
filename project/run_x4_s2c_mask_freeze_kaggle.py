from __future__ import annotations

"""Kaggle entrypoint that freezes the audited X4 S2C train/val masks.

The generator checkpoint and both class-agnostic SAM proposal caches are
immutable inputs.  Train and validation are processed sequentially in one GPU
job so the checkpoint is loaded under the same software and hardware contract.
The two native-resolution mask trees are archived before kernel completion to
avoid exposing thousands of incidental files as Kaggle outputs.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SOURCE_COMMIT = "ffb29546111a4e99238a93585cd6c3edcae4bb5a"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CHECKPOINT_SHA256 = "b46ac9af9a37c8a206ec2de071500600e40bcb6ae7a268ea97a63a01802405f3"
TRAIN_CACHE_MANIFEST_SHA256 = "63029e41066af3b044082c6a9c689612073e92cf6595b21702de672059643c59"
VAL_CACHE_MANIFEST_SHA256 = "3298c74e75e2ff2019a60d0026651efd51dcb8814d6d94d3b0efe0fe60dd3d0c"
FREEZER_SHA256 = "2ac5609a963d0b9940094a43e49f39a646770c21bb983a54ab3290cf590821df"
MODEL_SHA256 = "dd3d3e085578bec038dcee4d0c0768b7e364cb877833fc30ef4d413afa69d070"
DATASET_SHA256 = "da6c23b19c01000a69b02ffa43acafe6c418ec350e7e8645c7caf454f6489b32"
FROZEN_IO_SHA256 = "423c5b9eef87a59d9f457bc4acc1f6795cff9c4b1c875f25d651b5aa88987a2d"
CONTRACT_SHA256 = "cc41be0f8ea5cd675fc3e4ce21b7b47a805446898bc55fb5bda5eeadde5f8bb5"
CONFIG_SHA256 = "3e41acc0729694c340b838ed1965411c9604ad2c2e2c2a88ae1725785e893a32"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(name: str, expected: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if path.is_file() and sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name} with SHA {expected}, found {matches}")
    return matches[0]


def dataset_root() -> Path:
    roots = sorted({
        path.parent
        for path in INPUT.rglob("images")
        if path.is_dir() and (path.parent / "Annotations").is_dir()
    })
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one BTXRD root, found {roots}")
    return roots[0]


def freeze_split(
    *,
    freezer: Path,
    data: Path,
    split_manifest: Path,
    split: str,
    cache: Path,
    cache_manifest_sha256: str,
    checkpoint: Path,
) -> dict[str, object]:
    output = WORKING / f"x4_s2c_masks_{split}"
    command = [
        sys.executable,
        str(freezer),
        "--repo-root", str(freezer.parents[1]),
        "--dataset-root", str(data),
        "--split-manifest", str(split_manifest),
        "--split", split,
        "--segment-cache", str(cache),
        "--expected-cache-manifest-sha256", cache_manifest_sha256,
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", CHECKPOINT_SHA256,
        "--source-commit", SOURCE_COMMIT,
        "--batch-size", "4",
        "--num-workers", "0",
        "--device", "cuda",
        "--output-dir", str(output),
    ]
    print(json.dumps({"split": split, "command": command}), flush=True)
    subprocess.run(command, cwd=freezer.parents[1], check=True)
    freeze = json.loads((output / "x4_s2c_mask_freeze.json").read_text(encoding="utf-8"))
    expected_images = 2981 if split == "train" else 371
    if int(freeze.get("images", -1)) != expected_images:
        raise RuntimeError(f"{split} freeze count differs")
    if freeze.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError(f"{split} freeze checkpoint differs")
    if freeze.get("segment_cache_manifest_sha256") != cache_manifest_sha256:
        raise RuntimeError(f"{split} freeze cache differs")
    archive = Path(shutil.make_archive(
        str(WORKING / f"x4_s2c_masks_{split}"),
        "zip",
        root_dir=WORKING,
        base_dir=output.name,
    ))
    receipt = {
        "split": split,
        "images": expected_images,
        "archive_sha256": sha256(archive),
        "freeze_sha256": sha256(output / "x4_s2c_mask_freeze.json"),
        "manifest_sha256": sha256(output / "x4_s2c_mask_manifest.csv"),
        "cache_manifest_sha256": cache_manifest_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    shutil.rmtree(output)
    return receipt


def main() -> None:
    started = time.perf_counter()
    os.environ.update({
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MALLOC_ARENA_MAX": "2",
    })
    freezer = exact_file("freeze_x4_s2c_masks.py", FREEZER_SHA256)
    project = freezer.parent
    locked = {
        project / "models" / "s2c.py": MODEL_SHA256,
        project / "datasets" / "s2c.py": DATASET_SHA256,
        project / "frozen_io.py": FROZEN_IO_SHA256,
        project / "x4_contract.py": CONTRACT_SHA256,
        project / "config.py": CONFIG_SHA256,
    }
    for path, expected in locked.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"locked source dependency differs: {path}")
    split_manifest = exact_file("canonical_split_manifest_85511.csv", SPLIT_SHA256)
    checkpoint = exact_file("last_s2c.pt", CHECKPOINT_SHA256)
    train_cache = exact_file("sam_segment_manifest.csv", TRAIN_CACHE_MANIFEST_SHA256).parent
    val_cache = exact_file("sam_segment_manifest.csv", VAL_CACHE_MANIFEST_SHA256).parent
    data = dataset_root()
    receipts = [
        freeze_split(
            freezer=freezer,
            data=data,
            split_manifest=split_manifest,
            split="train",
            cache=train_cache,
            cache_manifest_sha256=TRAIN_CACHE_MANIFEST_SHA256,
            checkpoint=checkpoint,
        ),
        freeze_split(
            freezer=freezer,
            data=data,
            split_manifest=split_manifest,
            split="val",
            cache=val_cache,
            cache_manifest_sha256=VAL_CACHE_MANIFEST_SHA256,
            checkpoint=checkpoint,
        ),
    ]
    result = {
        "schema_version": 1,
        "stage": "x4_s2c_train_val_mask_freeze_kaggle_v1",
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "splits": receipts,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt_path = WORKING / "x4_s2c_mask_freeze_receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "receipt_sha256": sha256(receipt_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
