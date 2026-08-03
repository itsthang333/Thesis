from __future__ import annotations

"""Hash-bound, annotation-free I/O shared by the final pipeline."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_sha256(value: str, *, name: str) -> str:
    value = value.strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_split_rows_without_annotations(
    split_manifest: Path,
    *,
    expected_sha256: str,
    split: str,
    allow_test: bool = False,
) -> list[dict[str, str]]:
    allowed = {"train", "val"} | ({"test"} if allow_test else set())
    if split not in allowed:
        raise ValueError(
            "annotation-free split reader may only read train/val; test requires "
            "an explicit final-protocol call with allow_test=True"
        )
    if sha256_file(split_manifest) != validate_sha256(
        expected_sha256, name="expected split manifest SHA-256"
    ):
        raise ValueError("Split manifest SHA-256 mismatch")
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("split") == split and row.get("eligible") == "1"
        ]
    if not rows:
        raise ValueError(f"No eligible rows for split {split}")
    required = {"image_id", "group_id", "tumor", "image_sha256"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Split manifest lacks required fields: {missing}")
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Selected split contains duplicate image IDs")
    if any(row["tumor"] not in {"0", "1"} for row in rows):
        raise ValueError("Selected split contains an invalid binary image label")
    return sorted(rows, key=lambda row: row["image_id"])


def locate_verified_image(dataset_root: Path, row: dict[str, str]) -> Path:
    candidates = (
        dataset_root / "images" / row["image_id"],
        dataset_root / row["image_id"],
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Image not found uniquely: {row['image_id']}")
    if sha256_file(matches[0]) != row["image_sha256"]:
        raise ValueError(f"Source image SHA-256 mismatch: {row['image_id']}")
    return matches[0]


def verify_model_snapshot(
    model_dir: Path,
    *,
    expected_config_sha256: str,
    expected_preprocessor_sha256: str,
    expected_weight_sha256: str,
) -> dict[str, Any]:
    files = {
        "config.json": (
            model_dir / "config.json",
            validate_sha256(expected_config_sha256, name="config SHA-256"),
        ),
        "preprocessor_config.json": (
            model_dir / "preprocessor_config.json",
            validate_sha256(expected_preprocessor_sha256, name="preprocessor SHA-256"),
        ),
        "model.safetensors": (
            model_dir / "model.safetensors",
            validate_sha256(expected_weight_sha256, name="weight SHA-256"),
        ),
    }
    result: dict[str, Any] = {}
    for name, (path, expected) in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model snapshot lacks {name}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen model file hash mismatch for {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return result


def save_float_map(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Cannot save a non-finite or non-2D map")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("Normalized map must lie in [0,1]")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float16, copy=False), allow_pickle=False)
