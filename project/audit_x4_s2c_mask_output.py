from __future__ import annotations

"""Independent annotation-free audit for X4 S2C Stage-A masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import resolve_btxrd_root
from frozen_io import load_split_rows_without_annotations, locate_verified_image
from pseudo.manifest import sha256_file
from x4_contract import CANONICAL_SPLIT_SHA256, load_x4_protocol


EXPECTED = {"train": (2981, 1493, 1488), "val": (371, 187, 184)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    protocol, protocol_sha = load_x4_protocol(args.repo_root)
    freeze_path = args.output_root / "x4_s2c_mask_freeze.json"
    manifest_path = args.output_root / "x4_s2c_mask_manifest.csv"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "stage": "x4_s2c_mask_freeze_v1",
        "study": protocol["study"],
        "split": args.split,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "segment_cache_manifest_sha256": args.expected_cache_manifest_sha256,
        "native_resolution_masks": True,
        "normal_targets_explicitly_empty": True,
        "training_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "masks_frozen_before_outer_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {key: {"actual": freeze.get(key), "expected": value}
                   for key, value in required.items() if freeze.get(key) != value}
    if differences:
        raise ValueError(f"X4 S2C freeze contract differs: {differences}")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("X4 S2C auditor checkpoint SHA differs")
    if freeze.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("X4 S2C manifest SHA differs")
    rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=CANONICAL_SPLIT_SHA256,
        split=args.split, allow_test=False,
    )
    count, normals, tumors = EXPECTED[args.split]
    if (len(rows), sum(1-int(r["tumor"]) for r in rows), sum(int(r["tumor"]) for r in rows)) != (count, normals, tumors):
        raise ValueError("X4 S2C canonical cohort differs")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    by_id = {row["image_id"]: row for row in manifest}
    canonical = {row["image_id"]: row for row in rows}
    if len(manifest) != len(by_id) or set(by_id) != set(canonical):
        raise ValueError("X4 S2C output IDs differ")
    root = resolve_btxrd_root(args.dataset_root)
    foreground = tumor_empty = selected_images = selected_segments = 0
    for image_id, source in canonical.items():
        row = by_id[image_id]
        if row["group_id"] != source["group_id"] or int(row["tumor"]) != int(source["tumor"]):
            raise ValueError(f"X4 S2C label/group differs: {image_id}")
        relative = Path(row["mask_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe X4 S2C mask path: {image_id}")
        path = args.output_root / relative
        if sha256_file(path) != row["mask_sha256"]:
            raise ValueError(f"X4 S2C mask SHA differs: {image_id}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"))
        image_path = locate_verified_image(root, source)
        with Image.open(image_path) as handle:
            width, height = handle.size
        if mask.shape != (height, width) or mask.shape != (int(row["mask_height"]), int(row["mask_width"])):
            raise ValueError(f"X4 S2C native geometry differs: {image_id}")
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError(f"X4 S2C mask is not binary: {image_id}")
        positive = int((mask > 0).sum())
        if positive != int(row["mask_foreground_pixels"]):
            raise ValueError(f"X4 S2C foreground differs: {image_id}")
        if int(source["tumor"]) == 0 and positive:
            raise ValueError(f"X4 S2C normal mask is non-empty: {image_id}")
        chosen = int(row["selected_segment_count"])
        foreground += positive
        selected_images += int(chosen > 0)
        selected_segments += chosen
        tumor_empty += int(int(source["tumor"]) == 1 and positive == 0)
    aggregate = {
        "images": count, "normal_images": normals, "tumor_images": tumors,
        "selected_images": selected_images, "selected_segments": selected_segments,
        "tumor_empty_masks": tumor_empty, "total_foreground_pixels": foreground,
    }
    if any(int(freeze.get(key, -1)) != value for key, value in aggregate.items()):
        raise ValueError("X4 S2C aggregate counts differ")
    audit = {
        "schema_version": 1,
        "stage": "independent_x4_s2c_mask_output_audit_v1",
        "status": "pass",
        "split": args.split,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "cache_manifest_sha256": args.expected_cache_manifest_sha256,
        "freeze_sha256": sha256_file(freeze_path),
        "manifest_sha256": sha256_file(manifest_path),
        **aggregate,
        "native_geometry_verified": True,
        "normal_masks_empty": True,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "audit_sha256": sha256_file(args.audit_output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
