from __future__ import annotations

"""Evaluate frozen YOLO binary-union masks with the common X4 native evaluator."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import _decode_labelme_polygon_mask, resolve_btxrd_root
from evaluation.segmentation_metrics import (
    bootstrap_group_confidence_intervals,
    json_safe,
    segmentation_metrics,
    subgroup_summaries,
    summarize_segmentation_rows,
)
from frozen_io import load_split_rows_without_annotations, sha256_file, validate_sha256
from x4_contract import load_x4_protocol


SIZE_GROUPS = ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return SIZE_GROUPS[0]
    if area_ratio < 0.05:
        return SIZE_GROUPS[1]
    return SIZE_GROUPS[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--expected-training-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    split_sha = validate_sha256(args.expected_split_sha256, name="split SHA-256")
    freeze_path = args.prediction_root / "prediction_freeze.json"
    manifest_path = args.prediction_root / "prediction_manifest.csv"
    if sha256_file(freeze_path) != validate_sha256(
        args.expected_prediction_freeze_sha256, name="prediction freeze SHA-256"
    ):
        raise ValueError("X4 YOLO prediction freeze changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "x4_yolo_prediction_freeze_v1"
        or freeze.get("split_sha256") != split_sha
        or freeze.get("images") != 371
        or freeze.get("predictions_frozen_before_spatial_ground_truth") is not True
        or freeze.get("validation_annotations_read") != 0
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
        or freeze.get("prediction_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("X4 YOLO Stage-A freeze violates the protocol")
    training_report_path = args.training_root / "training_report.json"
    if sha256_file(training_report_path) != validate_sha256(
        args.expected_training_report_sha256, name="training report SHA-256"
    ):
        raise ValueError("X4 YOLO training report changed")
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training.get("test_images_read") != 0 or training.get("test_evaluated") is not False:
        raise ValueError("X4 YOLO training report accessed test")

    split_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=split_sha, split="val", allow_test=False
    )
    rows = read_csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != 371 or set(indexed) != {row["image_id"] for row in split_rows}:
        raise ValueError("X4 YOLO prediction cohort differs")
    for split_row in split_rows:
        row = indexed[split_row["image_id"]]
        relative = Path(row["mask_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe YOLO prediction path")
        path = args.prediction_root / relative
        if sha256_file(path) != row["mask_sha256"]:
            raise ValueError(f"X4 YOLO prediction changed: {split_row['image_id']}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"))
        expected_shape = (int(row["native_height"]), int(row["native_width"]))
        if mask.shape != expected_shape or not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError(f"X4 YOLO native mask geometry differs: {split_row['image_id']}")

    protocol, protocol_sha = load_x4_protocol(Path(__file__).resolve().parents[1])
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    per_image: list[dict[str, object]] = []
    annotations_opened = 0
    for split_row in split_rows:
        image_id = split_row["image_id"]
        row = indexed[image_id]
        height, width = int(row["native_height"]), int(row["native_width"])
        if int(split_row["tumor"]):
            target = _decode_labelme_polygon_mask(
                btxrd_root / "Annotations" / f"{Path(image_id).stem}.json",
                height=height,
                width=width,
            )
            annotations_opened += 1
        else:
            target = np.zeros((height, width), dtype=bool)
        with Image.open(args.prediction_root / row["mask_path"]) as handle:
            prediction = np.asarray(handle.convert("L")) > 0
        area_ratio = float(target.mean())
        per_image.append(
            {
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "group_source": split_row.get("group_source", ""),
                "tumor": split_row["tumor"],
                "arm": "yolov8s_seg",
                "seed": training["seed"],
                "evaluation_grid": "native",
                "native_height": height,
                "native_width": width,
                "native_size_group": size_group(area_ratio) if area_ratio else "normal",
                **segmentation_metrics(prediction, target, compute_boundary=True),
            }
        )
    if annotations_opened != 184:
        raise ValueError("X4 YOLO evaluator did not open exactly 184 validation annotations")
    counts = {
        group: sum(row["native_size_group"] == group for row in per_image) for group in SIZE_GROUPS
    }
    if counts != {"small_lt_1pct": 94, "medium_1_to_5pct": 72, "large_ge_5pct": 18}:
        raise ValueError(f"X4 YOLO native subgroup counts differ: {counts}")
    summary = summarize_segmentation_rows(per_image)
    summary["native_subgroups"] = {
        group: summarize_segmentation_rows(
            [row for row in per_image if row["native_size_group"] == group]
        )
        for group in SIZE_GROUPS
    }
    summary["group_bootstrap_ci95"] = bootstrap_group_confidence_intervals(
        per_image,
        iterations=int(protocol["paired_bootstrap"]["iterations"]),
        seed=int(protocol["paired_bootstrap"]["seed"]),
    )
    args.output_dir.mkdir(parents=True)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(json_safe(per_image))
    subgroup_rows = subgroup_summaries(per_image)
    subgroup_path = args.output_dir / "subgroups.csv"
    with subgroup_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(subgroup_rows))
    report = {
        "schema_version": 1,
        "stage": "x4_yolov8s_seg_common_evaluation_v1",
        "model": "YOLOv8s-seg",
        "seed": training["seed"],
        "split_sha256": split_sha,
        "x4_protocol_sha256": protocol_sha,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "native_subgroup_counts": counts,
        "native_ultralytics_validation": training["native_ultralytics_validation"],
        "common_binary_union_summary": summary,
        "prediction_bytes_verified_before_annotations": True,
        "validation_annotations_opened": annotations_opened,
        "test_images_read": 0,
        "test_evaluated": False,
        "per_image_sha256": sha256_file(per_image_path),
        "subgroups_sha256": sha256_file(subgroup_path),
    }
    report_path = args.output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "evaluation_report_sha256": sha256_file(report_path)}, indent=2))


if __name__ == "__main__":
    main()
