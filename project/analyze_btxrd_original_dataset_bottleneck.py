from __future__ import annotations

"""Describe the canonical BTXRD train/validation cohort for WSSS research.

The script is deliberately diagnostic-only.  It reads polygon annotations only
for the canonical validation split, never opens the test split, and never emits
training targets.  Its purpose is to quantify the spatial resolution and cohort
confounding constraints that any image-label-only method must overcome.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


GRID_SIZES = (10, 14, 16, 32, 64, 80, 112, 128)
SUBGROUPS = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subgroup(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def load_polygon_mask(annotation_path: Path, width: int, height: int) -> tuple[np.ndarray, int]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    polygon_count = 0
    for shape in payload.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if len(points) < 3:
            continue
        draw.polygon(points, outline=1, fill=1)
        polygon_count += 1
    return np.asarray(image, dtype=bool), polygon_count


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "area_ratio": quantiles([float(row["gt_area_ratio"]) for row in rows]),
        "native_width": quantiles([float(row["width"]) for row in rows]),
        "native_height": quantiles([float(row["height"]) for row in rows]),
        "bbox_width_at_320": quantiles([float(row["bbox_width_at_320"]) for row in rows]),
        "bbox_height_at_320": quantiles([float(row["bbox_height_at_320"]) for row in rows]),
        "polygon_count": dict(sorted(Counter(int(row["polygon_count"]) for row in rows).items())),
        "view": dict(sorted(Counter(str(row["view"]) for row in rows).items())),
        "tumor_type": dict(
            sorted(Counter(f"{row['tumor_type']}:{row['tumor_type_name']}" for row in rows).items())
        ),
        "benign": int(sum(int(row["benign"]) for row in rows)),
        "malignant": int(sum(int(row["malignant"]) for row in rows)),
        "expected_positive_cells": {
            str(grid): quantiles(
                [float(row["gt_area_ratio"]) * float(grid * grid) for row in rows]
            )
            for grid in GRID_SIZES
        },
    }


def main() -> None:
    args = parse_args()
    images_dir = args.dataset_root / "images"
    annotations_dir = args.dataset_root / "Annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError("dataset root must contain images/ and Annotations/")

    with args.split_manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if int(row["eligible"]) == 1]
    split_counts = Counter((row["split"], int(row["tumor"])) for row in manifest_rows)
    expected = {
        ("train", 0): 1493,
        ("train", 1): 1488,
        ("val", 0): 187,
        ("val", 1): 184,
    }
    if any(split_counts[key] != value for key, value in expected.items()):
        raise ValueError(f"canonical train/validation counts changed: {split_counts}")

    val_tumor = [row for row in manifest_rows if row["split"] == "val" and int(row["tumor"]) == 1]
    per_image: list[dict[str, Any]] = []
    for row in val_tumor:
        image_id = str(row["image_id"])
        width = int(row["width"])
        height = int(row["height"])
        image_path = images_dir / image_id
        annotation_path = annotations_dir / f"{Path(image_id).stem}.json"
        if not image_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError(f"missing canonical validation input for {image_id}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(f"manifest/native size mismatch for {image_id}")
        mask, polygon_count = load_polygon_mask(annotation_path, width, height)
        area = int(mask.sum())
        if area <= 0:
            raise ValueError(f"tumor validation polygon is empty: {image_id}")
        ys, xs = np.where(mask)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        area_ratio = float(area / float(width * height))
        per_image.append(
            {
                "image_id": image_id,
                "group_id": str(row["group_id"]),
                "size_group": subgroup(area_ratio),
                "gt_area_ratio": area_ratio,
                "width": width,
                "height": height,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_width_at_320": float(320.0 * bbox_width / width),
                "bbox_height_at_320": float(320.0 * bbox_height / height),
                "polygon_count": polygon_count,
                "tumor_type": int(row["tumor_type"]),
                "tumor_type_name": str(row["tumor_type_name"]),
                "benign": int(row["benign"]),
                "malignant": int(row["malignant"]),
                "view": str(row["view"]),
                "anatomy": str(row["anatomy"]),
            }
        )

    subgroup_counts = Counter(row["size_group"] for row in per_image)
    if subgroup_counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise ValueError(f"canonical subgroup counts changed: {subgroup_counts}")

    group_summary: dict[str, Any] = {}
    for split in ("train", "val"):
        split_rows = [row for row in manifest_rows if row["split"] == split]
        sizes = Counter(str(row["group_id"]) for row in split_rows)
        group_summary[split] = {
            "images": len(split_rows),
            "groups": len(sizes),
            "multi_image_groups": int(sum(value > 1 for value in sizes.values())),
            "max_group_size": int(max(sizes.values())),
        }

    result = {
        "stage": "btxrd_original_dataset_bottleneck_analysis_v1",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "scope": "canonical train/validation only",
        "counts": {
            "train": {"normal": 1493, "tumor": 1488, "total": 2981},
            "val": {"normal": 187, "tumor": 184, "total": 371},
            "val_tumor_subgroups": dict(sorted(subgroup_counts.items())),
        },
        "groups": group_summary,
        "val_tumor": {
            "overall": summarize_rows(per_image),
            **{
                group: summarize_rows([row for row in per_image if row["size_group"] == group])
                for group in SUBGROUPS
            },
        },
        "academic_status": {
            "spatial_gt_role": "retrospective validation-only dataset diagnosis",
            "training_targets_written": False,
            "test_rows_selected": 0,
            "test_images_opened": 0,
            "test_annotations_opened": 0,
            "test_evaluated": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image_val_tumor_geometry.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "summary_sha256": sha256_file(summary_path),
        "per_image_sha256": sha256_file(per_image_path),
        "validation_tumor": 184,
        "test_images_opened": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
