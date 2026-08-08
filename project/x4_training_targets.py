from __future__ import annotations

"""Fail-closed training-target bundle used by matched X4 WSSS students."""

import csv
import json
from pathlib import Path

from PIL import Image

from frozen_io import sha256_file


TARGET_ARMS = ("cam", "puzzlecam", "s2c", "rich_gallery")


def inspect_mask(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as handle:
        mask = handle.convert("L")
        width, height = mask.size
        histogram = mask.histogram()
    if sum(histogram[2:255]) != 0:
        raise ValueError(f"X4 target mask is not a binary 2-D image: {path}")
    return {
        "mask_height": int(height),
        "mask_width": int(width),
        "mask_foreground_pixels": int(histogram[1] + histogram[255]),
        "mask_sha256": sha256_file(path),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_x4_target_bundle(
    root: Path,
    *,
    arm: str,
    split_sha256: str,
    expected_freeze_sha256: str,
    canonical_train_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    if arm not in TARGET_ARMS:
        raise ValueError(f"unknown X4 pseudo-target arm: {arm}")
    freeze_path = root / "x4_target_freeze.json"
    manifest_path = root / "x4_target_manifest.csv"
    if sha256_file(freeze_path) != expected_freeze_sha256:
        raise ValueError("X4 target freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("stage") != "x4_train_target_freeze_v1"
        or freeze.get("arm") != arm
        or freeze.get("split_sha256") != split_sha256
        or int(freeze.get("images", -1)) != 2981
        or freeze.get("native_resolution_masks") is not True
        or freeze.get("normal_targets_explicitly_empty") is not True
        or int(freeze.get("train_spatial_annotations_read", -1)) != 0
        or freeze.get("targets_frozen_before_outer_validation_gt") is not True
        or freeze.get("outer_validation_annotations_read") != 0
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
        or freeze.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("X4 target freeze violates the training boundary")
    rows = read_csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    expected = {row["image_id"]: row for row in canonical_train_rows}
    if len(rows) != len(indexed) or set(indexed) != set(expected):
        raise ValueError("X4 target manifest differs from canonical train")
    tumor_empty = 0
    for image_id, row in indexed.items():
        source = expected[image_id]
        if row.get("group_id") != source.get("group_id") or int(row.get("tumor", -1)) != int(
            source.get("tumor", -2)
        ):
            raise ValueError(f"X4 target label/group differs: {image_id}")
        relative = Path(row["mask_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe X4 target mask path: {image_id}")
        inspected = inspect_mask(root / relative)
        if any(str(inspected[key]) != str(row[key]) for key in inspected):
            raise ValueError(f"X4 target mask metadata differs: {image_id}")
        if int(source["tumor"]) == 0 and int(inspected["mask_foreground_pixels"]) != 0:
            raise ValueError(f"normal X4 training target must be empty: {image_id}")
        tumor_empty += int(
            int(source["tumor"]) == 1 and int(inspected["mask_foreground_pixels"]) == 0
        )
    if int(freeze.get("tumor_empty_targets", -1)) != tumor_empty:
        raise ValueError("X4 target tumor-empty count differs")
    return indexed, freeze
