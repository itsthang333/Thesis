from __future__ import annotations

"""Freeze the canonical 2,981-image Rich-Gallery X4 target bundle.

The bounded CPU stage consumes the immutable Geometry-v3 candidate bank and
the independently frozen train G1 choices.  It never reads spatial annotations
or test images.  The resulting native-resolution target bundle is validated
with the same fail-closed loader used by every matched X4 student.
"""

import csv
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
SOURCE_COMMIT = "458ab52f145583fe97485a419a230c848f68b46d"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
TRAIN_CANDIDATE_MANIFEST_SHA256 = "e260be427d3a35d1b6305f17cc8e2e3ed53eb92641a9f19e6cfa6c8b10f8a436"
TRAIN_PSEUDO_MANIFEST_SHA256 = "649ee4232bbcca930c099e888708fa6894a34229ce08e1b80a17446c745a1f13"
G1_CHECKPOINT_SHA256 = "634e1200330e87692fab4a2e35ba70806790937d7b19ed8b0a3c4968471bfe8c"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    started = time.perf_counter()
    os.environ.update({
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    receipt_path = exact_file("x4_rich_gallery_train_g1_scores_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_receipt = {
        "stage": "x4_rich_gallery_train_g1_score_wrapper_v1",
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "candidate_manifest_sha256": TRAIN_CANDIDATE_MANIFEST_SHA256,
        "pseudo_manifest_sha256": TRAIN_PSEUDO_MANIFEST_SHA256,
        "g1_checkpoint_sha256": G1_CHECKPOINT_SHA256,
        "images": 2981,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": receipt.get(key), "expected": value}
        for key, value in required_receipt.items()
        if receipt.get(key) != value
    }
    if differences:
        raise RuntimeError(f"train-score receipt differs: {differences}")
    score_archive = exact_file("x4_rich_gallery_train_g1_scores.zip")
    if sha256(score_archive) != receipt.get("archive_sha256"):
        raise RuntimeError("train-score archive differs from receipt")
    extracted = WORKING / "train_score_extracted"
    safe_extract(score_archive, extracted)
    score_root = extracted / "x4_rich_gallery_train_g1_scores"
    evidence = score_root / "descriptor_evidence_manifest.csv"
    score_freeze = score_root / "diagnostic_freeze.json"
    if sha256(evidence) != receipt.get("evidence_manifest_sha256"):
        raise RuntimeError("train-score evidence manifest differs")
    if sha256(score_freeze) != receipt.get("freeze_sha256"):
        raise RuntimeError("train-score freeze differs")
    evidence_rows = read_csv(evidence)
    if len(evidence_rows) != 2981 or len({row["image_id"] for row in evidence_rows}) != 2981:
        raise RuntimeError("train-score evidence is not the canonical-sized cohort")

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
    candidate_manifest = exact_file(
        "candidate_diagnostics_manifest.csv", TRAIN_CANDIDATE_MANIFEST_SHA256
    )
    candidate_root = candidate_manifest.parent
    output = WORKING / "x4_rich_gallery_target"
    command = [
        sys.executable,
        str(freezer),
        "--arm", "rich_gallery",
        "--source-kind", "rich_gallery",
        "--repo-root", str(project.parent),
        "--split-manifest", str(split),
        "--source-root", str(score_root),
        "--source-manifest", str(evidence),
        "--expected-source-manifest-sha256", str(receipt["evidence_manifest_sha256"]),
        "--source-freeze", str(score_freeze),
        "--expected-source-freeze-sha256", str(receipt["freeze_sha256"]),
        "--candidate-root", str(candidate_root),
        "--candidate-manifest-sha256", TRAIN_CANDIDATE_MANIFEST_SHA256,
        "--candidate-pseudo-manifest-sha256", TRAIN_PSEUDO_MANIFEST_SHA256,
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
        arm="rich_gallery",
        split_sha256=SPLIT_SHA256,
        expected_freeze_sha256=target_freeze_sha,
        canonical_train_rows=canonical_rows,
    )
    if freeze.get("source_manifest_sha256") != receipt["evidence_manifest_sha256"]:
        raise RuntimeError("target bundle is not bound to the train-score evidence")
    archive = Path(shutil.make_archive(
        str(WORKING / "x4_rich_gallery_target"),
        "zip",
        root_dir=WORKING,
        base_dir=output.name,
    ))
    final_receipt = {
        "schema_version": 1,
        "stage": "x4_rich_gallery_target_freeze_wrapper_v1",
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "source_score_receipt_sha256": sha256(receipt_path),
        "source_manifest_sha256": receipt["evidence_manifest_sha256"],
        "source_freeze_sha256": receipt["freeze_sha256"],
        "candidate_manifest_sha256": TRAIN_CANDIDATE_MANIFEST_SHA256,
        "candidate_pseudo_manifest_sha256": TRAIN_PSEUDO_MANIFEST_SHA256,
        "target_freeze_sha256": target_freeze_sha,
        "target_manifest_sha256": freeze["manifest_sha256"],
        "archive_sha256": sha256(archive),
        "images": 2981,
        "tumor_images": 1488,
        "normal_images": 1493,
        "train_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    final_path = WORKING / "x4_rich_gallery_target_receipt.json"
    final_path.write_text(
        json.dumps(final_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(output)
    shutil.rmtree(extracted)
    print(json.dumps({**final_receipt, "receipt_sha256": sha256(final_path)}, indent=2))


if __name__ == "__main__":
    main()
