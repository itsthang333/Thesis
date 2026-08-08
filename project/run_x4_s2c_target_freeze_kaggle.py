from __future__ import annotations

"""Convert the audited S2C train-mask archive into the common X4 target bundle.

This is an I/O-bound CPU stage.  It validates every native train mask with the
same fail-closed loader used by matched students and does not read validation
polygons or test images.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SOURCE_COMMIT = "ffb29546111a4e99238a93585cd6c3edcae4bb5a"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CHECKPOINT_SHA256 = "b46ac9af9a37c8a206ec2de071500600e40bcb6ae7a268ea97a63a01802405f3"
TRAIN_CACHE_MANIFEST_SHA256 = "63029e41066af3b044082c6a9c689612073e92cf6595b21702de672059643c59"
FREEZER_SHA256 = "5eb5b4cebb677ec925f7c8c5446e655bdd5fffbc54c63d91398cad0dc2dad2ef"
TARGET_IO_SHA256 = "fc82186e8530b41d8798ab9df1ce8bd347d017e933fe298f5d8f41ead906cada"
FROZEN_IO_SHA256 = "423c5b9eef87a59d9f457bc4acc1f6795cff9c4b1c875f25d651b5aa88987a2d"
X4_CONTRACT_SHA256 = "cc41be0f8ea5cd675fc3e4ce21b7b47a805446898bc55fb5bda5eeadde5f8bb5"
CANDIDATE_IO_SHA256 = "0770fa8eb6d6b3fe35a4ae2db187d0e174b9b5f0a87f8ee254e2cdc41dbf6c61"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(name: str, expected: str | None = None) -> Path:
    matches = [path for path in INPUT.rglob(name) if path.is_file()]
    if expected is not None:
        matches = [path for path in matches if sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one locked {name}, found {matches}")
    return matches[0]


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def main() -> None:
    started = time.perf_counter()
    os.environ.update({
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    receipt_path = exact_file("x4_s2c_mask_freeze_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "stage": "x4_s2c_train_val_mask_freeze_kaggle_v1",
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if differences:
        raise RuntimeError(f"S2C mask receipt differs: {differences}")
    split_receipts = {item["split"]: item for item in receipt.get("splits", [])}
    if set(split_receipts) != {"train", "val"}:
        raise RuntimeError("S2C receipt does not bind exactly train and val")
    train_receipt = split_receipts["train"]
    expected_train = {
        "images": 2981,
        "cache_manifest_sha256": TRAIN_CACHE_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if any(train_receipt.get(key) != value for key, value in expected_train.items()):
        raise RuntimeError("S2C train receipt differs")
    archive = exact_file("x4_s2c_masks_train.zip")
    if sha256(archive) != train_receipt.get("archive_sha256"):
        raise RuntimeError("S2C train archive differs")
    extracted = WORKING / "s2c_train_extracted"
    safe_extract(archive, extracted)
    source_root = extracted / "x4_s2c_masks_train"
    source_manifest = source_root / "x4_s2c_mask_manifest.csv"
    source_freeze = source_root / "x4_s2c_mask_freeze.json"
    if sha256(source_manifest) != train_receipt.get("manifest_sha256"):
        raise RuntimeError("S2C train manifest differs")
    if sha256(source_freeze) != train_receipt.get("freeze_sha256"):
        raise RuntimeError("S2C train freeze differs")
    upstream = json.loads(source_freeze.read_text(encoding="utf-8"))
    required_upstream = {
        "stage": "x4_s2c_mask_freeze_v1",
        "split": "train",
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "segment_cache_manifest_sha256": TRAIN_CACHE_MANIFEST_SHA256,
        "images": 2981,
        "normal_images": 1493,
        "tumor_images": 1488,
        "training_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if any(upstream.get(key) != value for key, value in required_upstream.items()):
        raise RuntimeError("S2C train freeze violates the X4 boundary")

    freezer = exact_file("freeze_x4_training_targets.py", FREEZER_SHA256)
    project = freezer.parent
    locked = {
        project / "x4_training_targets.py": TARGET_IO_SHA256,
        project / "frozen_io.py": FROZEN_IO_SHA256,
        project / "x4_contract.py": X4_CONTRACT_SHA256,
        project / "pseudo" / "candidate_diagnostics.py": CANDIDATE_IO_SHA256,
    }
    for path, expected in locked.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"locked target-freeze dependency differs: {path}")
    split = exact_file("canonical_split_manifest_85511.csv", SPLIT_SHA256)
    output = WORKING / "x4_s2c_target"
    command = [
        sys.executable,
        str(freezer),
        "--arm", "s2c",
        "--source-kind", "mask_manifest",
        "--repo-root", str(project.parent),
        "--split-manifest", str(split),
        "--source-root", str(source_root),
        "--source-manifest", str(source_manifest),
        "--expected-source-manifest-sha256", str(train_receipt["manifest_sha256"]),
        "--source-freeze", str(source_freeze),
        "--expected-source-freeze-sha256", str(train_receipt["freeze_sha256"]),
        "--source-commit", SOURCE_COMMIT,
        "--output-dir", str(output),
    ]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=project.parent, check=True)

    sys.path.insert(0, str(project))
    from frozen_io import load_split_rows_without_annotations  # noqa: PLC0415
    from x4_training_targets import validate_x4_target_bundle  # noqa: PLC0415

    target_freeze = output / "x4_target_freeze.json"
    target_freeze_sha = sha256(target_freeze)
    canonical_rows = load_split_rows_without_annotations(
        split,
        expected_sha256=SPLIT_SHA256,
        split="train",
        allow_test=False,
    )
    _, freeze = validate_x4_target_bundle(
        output,
        arm="s2c",
        split_sha256=SPLIT_SHA256,
        expected_freeze_sha256=target_freeze_sha,
        canonical_train_rows=canonical_rows,
    )
    if freeze.get("source_freeze_sha256") != train_receipt["freeze_sha256"]:
        raise RuntimeError("S2C target is not bound to the audited mask freeze")
    target_archive = Path(shutil.make_archive(
        str(WORKING / "x4_s2c_target"),
        "zip",
        root_dir=WORKING,
        base_dir=output.name,
    ))
    final_receipt = {
        "schema_version": 1,
        "stage": "x4_s2c_target_freeze_wrapper_v1",
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "cache_manifest_sha256": TRAIN_CACHE_MANIFEST_SHA256,
        "source_mask_receipt_sha256": sha256(receipt_path),
        "source_manifest_sha256": train_receipt["manifest_sha256"],
        "source_freeze_sha256": train_receipt["freeze_sha256"],
        "target_freeze_sha256": target_freeze_sha,
        "target_manifest_sha256": freeze["manifest_sha256"],
        "archive_sha256": sha256(target_archive),
        "images": 2981,
        "tumor_images": 1488,
        "normal_images": 1493,
        "train_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    final_path = WORKING / "x4_s2c_target_receipt.json"
    final_path.write_text(
        json.dumps(final_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(output)
    shutil.rmtree(extracted)
    print(json.dumps({**final_receipt, "receipt_sha256": sha256(final_path)}, indent=2))


if __name__ == "__main__":
    main()
