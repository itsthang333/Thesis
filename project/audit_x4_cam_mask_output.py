from __future__ import annotations

"""Independent annotation-free audit for one X4 CAM mask-freeze output."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import resolve_btxrd_root
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
)
from x4_contract import CANONICAL_SPLIT_SHA256, load_x4_protocol


EXPECTED = {
    "train": {"images": 2981, "tumor": 1488, "normal": 1493},
    "val": {"images": 371, "tumor": 184, "normal": 187},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    protocol, protocol_sha = load_x4_protocol(args.repo_root)
    freeze_path = args.output_root / "x4_cam_mask_freeze.json"
    manifest_path = args.output_root / "x4_cam_mask_manifest.csv"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("auditor CAM checkpoint SHA-256 mismatch")
    required = {
        "schema_version": 1,
        "stage": "x4_cam_mask_freeze_v1",
        "study": protocol["study"],
        "split": args.split,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_seed": 42,
        "cam_percentile": 90.0,
        "constant_map_rule": "empty",
        "native_resolution_masks": True,
        "normal_targets_explicitly_empty": True,
        "train_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "masks_frozen_before_outer_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": freeze.get(key), "expected": expected}
        for key, expected in required.items()
        if freeze.get(key) != expected
    }
    if differences:
        raise ValueError(f"X4 CAM freeze contract differs: {differences}")
    if freeze.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("X4 CAM manifest SHA-256 differs")

    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split=args.split,
        allow_test=False,
    )
    expected_counts = EXPECTED[args.split]
    counts = {
        "images": len(rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
    }
    if counts != expected_counts:
        raise ValueError("auditor canonical CAM cohort differs")
    manifest = read_csv(manifest_path)
    by_id = {row["image_id"]: row for row in manifest}
    canonical = {row["image_id"]: row for row in rows}
    if len(manifest) != len(by_id) or set(by_id) != set(canonical):
        raise ValueError("auditor CAM output IDs differ from canonical cohort")

    root = resolve_btxrd_root(args.dataset_root)
    foreground = 0
    tumor_empty = 0
    for image_id, source in canonical.items():
        row = by_id[image_id]
        if row["group_id"] != source["group_id"] or row["tumor"] != source["tumor"]:
            raise ValueError(f"auditor CAM label/group differs: {image_id}")
        relative = Path(row["mask_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe CAM output path: {image_id}")
        mask_path = args.output_root / relative
        if sha256_file(mask_path) != row["mask_sha256"]:
            raise ValueError(f"auditor CAM mask hash differs: {image_id}")
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"))
        image_path = locate_verified_image(root, source)
        with Image.open(image_path) as handle:
            width, height = handle.size
        if mask.shape != (height, width) or mask.shape != (
            int(row["mask_height"]), int(row["mask_width"])
        ):
            raise ValueError(f"auditor CAM native geometry differs: {image_id}")
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError(f"auditor CAM mask is not binary: {image_id}")
        positive = int((mask > 0).sum())
        if positive != int(row["mask_foreground_pixels"]):
            raise ValueError(f"auditor CAM foreground count differs: {image_id}")
        if int(source["tumor"]) == 0 and positive != 0:
            raise ValueError(f"auditor CAM normal target is non-empty: {image_id}")
        foreground += positive
        tumor_empty += int(int(source["tumor"]) == 1 and positive == 0)
    if (
        int(freeze.get("images", -1)) != counts["images"]
        or int(freeze.get("tumor_images", -1)) != counts["tumor"]
        or int(freeze.get("normal_images", -1)) != counts["normal"]
        or int(freeze.get("total_foreground_pixels", -1)) != foreground
    ):
        raise ValueError("auditor CAM aggregate counts differ")

    audit = {
        "schema_version": 1,
        "stage": "independent_x4_cam_mask_output_audit_v1",
        "status": "pass",
        "split": args.split,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "freeze_sha256": sha256_file(freeze_path),
        "manifest_sha256": sha256_file(manifest_path),
        "images": counts["images"],
        "tumor_images": counts["tumor"],
        "normal_images": counts["normal"],
        "tumor_empty_masks": tumor_empty,
        "total_foreground_pixels": foreground,
        "native_geometry_verified": True,
        "normal_masks_empty": True,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**audit, "audit_sha256": sha256_file(args.audit_output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
