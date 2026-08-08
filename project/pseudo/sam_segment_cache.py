from __future__ import annotations

"""Integrity-checked, annotation-free cache of class-agnostic SAM segments.

The cache is a WSSS input artifact.  It contains only image-derived SAM
segments and SAM confidence values; it must never contain BTXRD polygons or
ground-truth masks.
"""

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from pseudo.manifest import sha256_file


SEGMENT_MANIFEST_NAME = "sam_segment_manifest.csv"
SEGMENT_SUMMARY_NAME = "sam_segment_summary.json"
SEGMENT_SCHEMA_VERSION = 2
SUPPORTED_SEGMENT_SCHEMA_VERSIONS = (1, SEGMENT_SCHEMA_VERSION)


def _sam_quality(annotation: Mapping[str, object]) -> float:
    predicted_iou = float(annotation.get("predicted_iou", 0.0))
    stability = float(annotation.get("stability_score", 0.0))
    return float(np.clip(0.5 * predicted_iou + 0.5 * stability, 0.0, 1.0))


def build_disjoint_segment_map(
    annotations: Sequence[Mapping[str, object]],
    *,
    shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert overlapping SAM proposals into deterministic disjoint regions.

    Small regions are assigned first so a large radiograph/bone proposal does
    not overwrite a potentially useful local lesion proposal.  Label 0 is
    reserved for uncovered/ignored pixels; valid segment ids are contiguous
    from 1.  No class label or spatial annotation is used.
    """

    if not 0.0 <= min_area_ratio < max_area_ratio <= 1.0:
        raise ValueError("Require 0 <= min_area_ratio < max_area_ratio <= 1")
    if not 1 <= max_segments <= np.iinfo(np.uint16).max - 1:
        raise ValueError("max_segments must be in [1, 65534]")

    height, width = shape
    image_area = float(height * width)
    candidates: list[tuple[int, float, int, np.ndarray]] = []
    for source_index, annotation in enumerate(annotations):
        mask = np.asarray(annotation.get("segmentation"), dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"SAM proposal {source_index} shape mismatch: expected {shape}, got {mask.shape}"
            )
        area = int(mask.sum())
        ratio = area / image_area
        if area > 0 and min_area_ratio <= ratio <= max_area_ratio:
            candidates.append((area, _sam_quality(annotation), source_index, mask))

    # Stable, deterministic priority: fine regions first, then better SAM
    # quality, then the generator's original order.
    candidates.sort(key=lambda item: (item[0], -item[1], item[2]))
    segment_map = np.zeros(shape, dtype=np.uint16)
    qualities = [0.0]
    area_ratios = [0.0]
    source_indices = [-1]
    for _area, quality, source_index, mask in candidates:
        if len(qualities) - 1 >= max_segments:
            break
        residual = mask & (segment_map == 0)
        residual_area = int(residual.sum())
        if residual_area / image_area < min_area_ratio:
            continue
        segment_id = len(qualities)
        segment_map[residual] = segment_id
        qualities.append(quality)
        area_ratios.append(residual_area / image_area)
        source_indices.append(source_index)

    return (
        segment_map,
        np.asarray(qualities, dtype=np.float32),
        np.asarray(area_ratios, dtype=np.float32),
        np.asarray(source_indices, dtype=np.int32),
    )


def build_proposal_bank(
    annotations: Sequence[Mapping[str, object]],
    *,
    shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    max_proposals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retain overlapping SAM proposals for CPM and final pseudo masks.

    SSC needs a disjoint partition, but CPM benefits from the original
    overlapping proposals: a lesion-shaped mask must not be destroyed merely
    because a different SAM region won the partition assignment.  The bank is
    still class-agnostic and image-derived only.
    """

    if not 0.0 <= min_area_ratio < max_area_ratio <= 1.0:
        raise ValueError("Require 0 <= min_area_ratio < max_area_ratio <= 1")
    if max_proposals < 1:
        raise ValueError("max_proposals must be positive")
    height, width = shape
    image_area = float(height * width)
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for source_index, annotation in enumerate(annotations):
        mask = np.asarray(annotation.get("segmentation"), dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"SAM proposal {source_index} shape mismatch: expected {shape}, got {mask.shape}"
            )
        area = int(mask.sum())
        ratio = area / image_area
        if area > 0 and min_area_ratio <= ratio <= max_area_ratio:
            candidates.append((_sam_quality(annotation), area, source_index, mask))
    # Preserve the most reliable masks first.  Area and source index make ties
    # deterministic without using any class or spatial ground truth.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    candidates = candidates[:max_proposals]
    if not candidates:
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.stack([item[3] for item in candidates]).astype(bool, copy=False),
        np.asarray([item[0] for item in candidates], dtype=np.float32),
        np.asarray([item[1] / image_area for item in candidates], dtype=np.float32),
        np.asarray([item[2] for item in candidates], dtype=np.int32),
    )


def inspect_segment_npz(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] | None = None,
    require_proposal_bank: bool = False,
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SAM segment cache file is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"segment_map", "quality", "area_ratio", "source_index"}
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"missing arrays {sorted(missing)}")
            segment_map = np.asarray(payload["segment_map"])
            quality = np.asarray(payload["quality"])
            area_ratio = np.asarray(payload["area_ratio"])
            source_index = np.asarray(payload["source_index"])
            proposal_required = {
                "proposal_masks",
                "proposal_quality",
                "proposal_area_ratio",
                "proposal_source_index",
            }
            proposal_present = proposal_required & set(payload.files)
            if proposal_present and proposal_present != proposal_required:
                raise ValueError(
                    f"incomplete proposal bank arrays: {sorted(proposal_required - proposal_present)}"
                )
            if require_proposal_bank and proposal_present != proposal_required:
                raise ValueError("missing overlapping SAM proposal bank")
            if proposal_present:
                proposal_masks = np.asarray(payload["proposal_masks"])
                proposal_quality = np.asarray(payload["proposal_quality"])
                proposal_area_ratio = np.asarray(payload["proposal_area_ratio"])
                proposal_source_index = np.asarray(payload["proposal_source_index"])
            else:
                proposal_masks = np.zeros((0, *segment_map.shape), dtype=bool)
                proposal_quality = np.zeros((0,), dtype=np.float32)
                proposal_area_ratio = np.zeros((0,), dtype=np.float32)
                proposal_source_index = np.zeros((0,), dtype=np.int32)
    except Exception as exc:
        raise ValueError(f"Unreadable SAM segment cache {path}: {exc}") from exc

    if segment_map.ndim != 2:
        raise ValueError(f"segment_map must be 2-D in {path}, got {segment_map.shape}")
    shape = (int(segment_map.shape[0]), int(segment_map.shape[1]))
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"SAM segment shape mismatch for {path}: expected {expected_shape}, got {shape}")
    if segment_map.dtype.kind not in "ui":
        raise ValueError(f"segment_map must use an integer dtype in {path}")
    if quality.ndim != 1 or area_ratio.ndim != 1 or source_index.ndim != 1:
        raise ValueError(f"SAM segment metadata arrays must be 1-D in {path}")
    max_id = int(segment_map.max(initial=0))
    expected_length = max_id + 1
    if not (len(quality) == len(area_ratio) == len(source_index) == expected_length):
        raise ValueError(
            f"SAM segment metadata length mismatch in {path}: max_id={max_id}, "
            f"quality={len(quality)}, area_ratio={len(area_ratio)}, source_index={len(source_index)}"
        )
    present = np.unique(segment_map)
    # Label 0 is reserved for uncovered pixels, but need not physically occur
    # when the retained SAM segments cover the complete image.
    expected_present = np.arange(1 if int(present[0]) == 1 else 0, max_id + 1, dtype=present.dtype)
    if int(present[0]) not in (0, 1) or not np.array_equal(present, expected_present):
        raise ValueError(f"SAM positive segment ids are not contiguous in {path}: {present[:20]}")
    if expected_length and (
        float(quality[0]) != 0.0
        or float(area_ratio[0]) != 0.0
        or int(source_index[0]) != -1
    ):
        raise ValueError(f"SAM segment id 0 must have zero quality/area and source_index=-1 in {path}")
    if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
        raise ValueError(f"Invalid SAM quality values in {path}")
    if not np.isfinite(area_ratio).all() or np.any((area_ratio < 0) | (area_ratio > 1)):
        raise ValueError(f"Invalid SAM segment area ratios in {path}")
    if proposal_masks.ndim != 3 or tuple(proposal_masks.shape[1:]) != shape:
        raise ValueError(f"proposal_masks must be [N,{shape[0]},{shape[1]}] in {path}")
    proposal_count = int(proposal_masks.shape[0])
    if not (
        proposal_quality.ndim == proposal_area_ratio.ndim == proposal_source_index.ndim == 1
        and len(proposal_quality) == len(proposal_area_ratio) == len(proposal_source_index) == proposal_count
    ):
        raise ValueError(f"SAM proposal metadata length mismatch in {path}")
    if proposal_masks.dtype.kind not in "bui" or (
        proposal_count and not np.isin(proposal_masks, (0, 1)).all()
    ):
        raise ValueError(f"proposal_masks must be binary in {path}")
    if proposal_count and np.any(proposal_masks.reshape(proposal_count, -1).sum(axis=1) == 0):
        raise ValueError(f"SAM proposal bank contains an empty mask in {path}")
    if not np.isfinite(proposal_quality).all() or np.any((proposal_quality < 0) | (proposal_quality > 1)):
        raise ValueError(f"Invalid SAM proposal quality values in {path}")
    if not np.isfinite(proposal_area_ratio).all() or np.any(
        (proposal_area_ratio <= 0) | (proposal_area_ratio > 1)
    ):
        raise ValueError(f"Invalid SAM proposal area ratios in {path}")

    return {
        "cache_height": shape[0],
        "cache_width": shape[1],
        "segment_count": max_id,
        "proposal_count": proposal_count,
        "covered_area_ratio": float((segment_map > 0).mean()),
        "cache_sha256": sha256_file(path),
    }


def write_sam_segment_cache_manifest(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    *,
    expected_image_names: Iterable[str],
    split: str,
    sam_image_size: int,
    run_metadata_sha256: str,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    expected = [str(name) for name in expected_image_names]
    expected_stems = {Path(name).stem for name in expected}
    row_stems = [Path(str(row.get("image_name", ""))).stem for row in rows]
    if len(row_stems) != len(set(row_stems)):
        raise RuntimeError("SAM segment cache rows contain duplicate image ids")
    missing = sorted(expected_stems - set(row_stems))
    unexpected = sorted(set(row_stems) - expected_stems)
    if missing or unexpected:
        raise RuntimeError(
            f"SAM segment cache is incomplete: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    enriched: list[dict[str, object]] = []
    for row in rows:
        stem = Path(str(row["image_name"])).stem
        relative_path = Path("segments") / f"{stem}.npz"
        inspected = inspect_segment_npz(
            output_dir / relative_path,
            expected_shape=(sam_image_size, sam_image_size),
            require_proposal_bank=True,
        )
        enriched.append({**row, "split": split, "cache_path": str(relative_path), **inspected})

    manifest_path = output_dir / SEGMENT_MANIFEST_NAME
    fieldnames = sorted({key for row in enriched for key in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    summary = {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "complete": True,
        "wsss_supervision": "class-agnostic SAM segments from radiographs; no GT mask/polygon",
        "split": split,
        "expected_images": len(expected_stems),
        "manifest_rows": len(enriched),
        "sam_image_size": int(sam_image_size),
        "total_segments": int(sum(int(row["segment_count"]) for row in enriched)),
        "total_proposals": int(sum(int(row["proposal_count"]) for row in enriched)),
        "run_metadata_sha256": run_metadata_sha256,
        "manifest_sha256": sha256_file(manifest_path),
    }
    summary_path = output_dir / SEGMENT_SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def validate_sam_segment_cache(
    output_dir: str | Path,
    samples: Iterable[Mapping[str, object]],
    *,
    split: str,
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / SEGMENT_MANIFEST_NAME
    summary_path = output_dir / SEGMENT_SUMMARY_NAME
    run_metadata_path = output_dir / "run_metadata.json"
    for path in (manifest_path, summary_path, run_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"SAM segment cache provenance is incomplete: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema_version = int(summary.get("schema_version", -1))
    if schema_version not in SUPPORTED_SEGMENT_SCHEMA_VERSIONS:
        raise ValueError("Unsupported SAM segment cache schema")
    if not summary.get("complete") or str(summary.get("split")) != split:
        raise ValueError(f"SAM segment cache is incomplete or for the wrong split: {summary_path}")
    if summary.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("SAM segment manifest hash does not match summary")
    if summary.get("run_metadata_sha256") != sha256_file(run_metadata_path):
        raise ValueError("SAM segment run_metadata.json changed after cache creation")

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {Path(str(sample["image_id"])).stem: sample for sample in samples}
    indexed: dict[str, dict[str, str]] = {}
    expected_shape = (int(summary["sam_image_size"]), int(summary["sam_image_size"]))
    for row in rows:
        stem = Path(str(row.get("image_name", ""))).stem
        if not stem or stem in indexed:
            raise ValueError(f"SAM segment manifest contains duplicate/empty image id: {stem!r}")
        indexed[stem] = row
    missing = sorted(set(expected) - set(indexed))
    unexpected = sorted(set(indexed) - set(expected))
    if missing or unexpected:
        raise ValueError(f"SAM segment cache does not match split: missing={missing[:5]}, unexpected={unexpected[:5]}")

    for stem, sample in expected.items():
        row = indexed[stem]
        if row.get("split") != split:
            raise ValueError(f"SAM segment row {stem} has wrong split {row.get('split')!r}")
        if int(row.get("tumor", -1)) != int(bool(sample.get("tumor", 0))):
            raise ValueError(f"SAM segment image-label mismatch for {stem}")
        source_hash = str(sample.get("image_sha256", ""))
        if source_hash and row.get("source_image_sha256") != source_hash:
            raise ValueError(f"SAM segment source image hash mismatch for {stem}")
        cache_path = output_dir / str(row["cache_path"])
        inspected = inspect_segment_npz(
            cache_path,
            expected_shape=expected_shape,
            require_proposal_bank=schema_version >= 2,
        )
        if row.get("cache_sha256") != inspected["cache_sha256"]:
            raise ValueError(f"SAM segment cache hash mismatch for {stem}")

    return {
        **summary,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "rows": indexed,
    }


def load_cached_segments(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        segment_map = np.asarray(payload["segment_map"], dtype=np.int64)
        quality = np.asarray(payload["quality"], dtype=np.float32)
    return segment_map, quality


def load_cached_proposal_bank(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load overlapping masks; legacy schema-v1 caches return an empty bank."""

    with np.load(path, allow_pickle=False) as payload:
        if "proposal_masks" not in payload.files:
            shape = tuple(np.asarray(payload["segment_map"]).shape)
            return np.zeros((0, *shape), dtype=bool), np.zeros((0,), dtype=np.float32)
        masks = np.asarray(payload["proposal_masks"], dtype=bool)
        quality = np.asarray(payload["proposal_quality"], dtype=np.float32)
    return masks, quality

