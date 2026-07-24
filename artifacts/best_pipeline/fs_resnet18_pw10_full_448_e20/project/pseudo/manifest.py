from __future__ import annotations

"""Integrity/provenance manifest for generated pseudo masks.

The manifest is deliberately independent of torch so it can be checked before
the expensive training stack is initialised.  A pseudo-mask directory is not a
valid training target merely because PNG files happen to exist: every expected
image must have one unique, binary, correctly shaped and hash-matched entry.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image


MANIFEST_NAME = "pseudo_mask_manifest.csv"
SUMMARY_NAME = "pseudo_mask_summary.json"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_binary_mask(path: str | Path, expected_shape: tuple[int, int] | None = None) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pseudo-mask file is missing: {path}")
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
    except Exception as exc:
        raise ValueError(f"Unreadable pseudo-mask {path}: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"Pseudo-mask must be 2-D, got shape={array.shape}: {path}")
    shape = (int(array.shape[0]), int(array.shape[1]))
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Pseudo-mask shape mismatch for {path}: expected {expected_shape}, got {shape}")
    values = sorted(int(value) for value in np.unique(array))
    if not set(values).issubset({0, 1, 255}):
        raise ValueError(f"Pseudo-mask is not binary for {path}; unique values={values[:20]}")
    foreground = int((array > 0).sum())
    return {
        "mask_height": shape[0],
        "mask_width": shape[1],
        "mask_unique_values": "|".join(str(value) for value in values),
        "mask_foreground_pixels": foreground,
        "mask_area_ratio": foreground / float(array.size),
        "mask_sha256": sha256_file(path),
    }


def write_pseudo_mask_manifest(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    *,
    expected_image_names: Iterable[str],
    split: str,
    image_size: int,
    run_metadata_sha256: str,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    expected = [str(name) for name in expected_image_names]
    expected_stems = {Path(name).stem for name in expected}
    row_stems = [Path(str(row.get("image_name", ""))).stem for row in rows]
    duplicates = sorted({stem for stem in row_stems if row_stems.count(stem) > 1})
    missing = sorted(expected_stems - set(row_stems))
    unexpected = sorted(set(row_stems) - expected_stems)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "Pseudo-mask generation is incomplete/inconsistent: "
            f"duplicates={duplicates[:5]}, missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    enriched: list[dict[str, object]] = []
    for row in rows:
        stem = Path(str(row["image_name"])).stem
        mask_path = output_dir / "masks" / f"{stem}.png"
        inspected = inspect_binary_mask(mask_path, expected_shape=(image_size, image_size))
        enriched.append(
            {
                **row,
                "split": split,
                "mask_path": str(Path("masks") / f"{stem}.png"),
                **inspected,
            }
        )

    manifest_path = output_dir / MANIFEST_NAME
    fieldnames = sorted({key for row in enriched for key in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    statuses: dict[str, int] = {}
    for row in enriched:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    summary = {
        "schema_version": 2,
        "complete": True,
        "split": split,
        "expected_images": len(expected_stems),
        "manifest_rows": len(enriched),
        "image_size": image_size,
        "statuses": statuses,
        "run_metadata_sha256": run_metadata_sha256,
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def validate_pseudo_mask_manifest(
    mask_dir: str | Path,
    samples: Iterable[Mapping[str, object]],
    *,
    split: str,
    image_size: int,
) -> dict[str, object]:
    mask_dir = Path(mask_dir).resolve()
    output_dir = mask_dir.parent
    manifest_path = output_dir / MANIFEST_NAME
    summary_path = output_dir / SUMMARY_NAME
    run_metadata_path = output_dir / "run_metadata.json"
    for path in (manifest_path, summary_path, run_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Pseudo-mask provenance is incomplete: required file is missing: {path}"
            )

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("schema_version", -1)) != 2:
        raise ValueError(
            "Pseudo-mask manifest schema is not the unambiguous counter schema v2; "
            "regenerate pseudo masks with the current pipeline"
        )
    if not summary.get("complete"):
        raise ValueError(f"Pseudo-mask summary is not marked complete: {summary_path}")
    if str(summary.get("split")) != split:
        raise ValueError(f"Pseudo-mask manifest split={summary.get('split')!r}, expected {split!r}")
    if int(summary.get("image_size", -1)) != int(image_size):
        raise ValueError(
            f"Pseudo-mask manifest image_size={summary.get('image_size')!r}, expected {image_size}"
        )
    actual_manifest_hash = sha256_file(manifest_path)
    if summary.get("manifest_sha256") != actual_manifest_hash:
        raise ValueError("Pseudo-mask manifest hash does not match pseudo_mask_summary.json")
    if summary.get("run_metadata_sha256") != sha256_file(run_metadata_path):
        raise ValueError("Pseudo-mask run_metadata.json changed after the manifest was created")

    expected = {Path(str(sample["image_id"])).stem: sample for sample in samples}
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        required_counters = {
            "above_threshold_candidates",
            "selected_candidates",
            "selected_components",
            "sam_prompt_calls",
            "unique_prompt_points",
        }
        missing_counters = required_counters - set(row)
        if missing_counters:
            raise ValueError(f"Pseudo-mask manifest is missing counters: {sorted(missing_counters)}")
        stem = Path(row.get("image_name", "")).stem
        if not stem or stem in indexed:
            raise ValueError(f"Pseudo-mask manifest contains duplicate/empty image id: {stem!r}")
        indexed[stem] = row
    missing = sorted(set(expected) - set(indexed))
    unexpected = sorted(set(indexed) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Pseudo-mask manifest does not match dataset split: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for stem, sample in expected.items():
        row = indexed[stem]
        if row.get("split") != split:
            raise ValueError(f"Pseudo-mask row {stem} has split={row.get('split')!r}, expected {split!r}")
        mask_path = mask_dir / f"{stem}.png"
        inspected = inspect_binary_mask(mask_path, expected_shape=(image_size, image_size))
        if row.get("mask_sha256") != inspected["mask_sha256"]:
            raise ValueError(f"Pseudo-mask hash mismatch for {stem}")
        if str(row.get("true_tumor", "")) not in {"0", "1"}:
            raise ValueError(f"Pseudo-mask row {stem} has invalid true_tumor={row.get('true_tumor')!r}")
        if int(row["true_tumor"]) != int(bool(sample.get("tumor", 0))):
            raise ValueError(f"Pseudo-mask image-level label mismatch for {stem}")

    return {
        **summary,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_hash,
        "summary_sha256": sha256_file(summary_path),
    }
