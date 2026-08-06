from __future__ import annotations

"""Evaluate one frozen G4 pseudo-mask variant on canonical validation.

The integrity boundary is deliberate: all 371 prediction files and their
provenance records are verified before the first spatial annotation is opened.
The evaluator never accepts test rows.
"""

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
    summarize_segmentation_rows,
)
from frozen_io import load_split_rows_without_annotations, sha256_file
from pseudo.manifest import validate_pseudo_mask_manifest


SIZE_GROUPS = ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Directory containing masks/, pseudo_mask_manifest.csv and provenance files.",
    )
    parser.add_argument("--expected-pseudo-summary-sha256", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def _resize(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return np.asarray(mask, dtype=bool)
    height, width = shape
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0


def _size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small_lt_1pct"
    if area_ratio < 0.05:
        return "medium_1_to_5pct"
    return "large_ge_5pct"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("G4 evaluation output must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("G4 variant evaluation requires canonical 371/184 validation")

    pseudo_summary_path = args.prediction_root / "pseudo_mask_summary.json"
    if sha256_file(pseudo_summary_path) != args.expected_pseudo_summary_sha256:
        raise ValueError("Pseudo-mask summary SHA-256 mismatch")

    # Normalize the label type because the generic manifest checker correctly
    # treats labels as booleans, while csv.DictReader represents zero as "0".
    validation_samples = [
        {**row, "tumor": int(row["tumor"])} for row in split_rows
    ]
    integrity = validate_pseudo_mask_manifest(
        args.prediction_root / "masks",
        validation_samples,
        split="val",
        image_size=None,
    )
    source_size = int(integrity["source_image_size"])

    # Every prediction byte is now verified. Spatial annotations may only be
    # opened below this point.
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    per_image: list[dict[str, object]] = []
    opened_annotations = 0
    for split_row in split_rows:
        image_id = str(split_row["image_id"])
        tumor = int(split_row["tumor"]) == 1
        with Image.open(btxrd_root / "images" / image_id) as image:
            native_width, native_height = image.size
        native_shape = (native_height, native_width)
        if tumor:
            target_native = _decode_labelme_polygon_mask(
                btxrd_root / "Annotations" / f"{Path(image_id).stem}.json",
                height=native_height,
                width=native_width,
            ).astype(bool)
            opened_annotations += 1
        else:
            target_native = np.zeros(native_shape, dtype=bool)

        with Image.open(args.prediction_root / "masks" / f"{Path(image_id).stem}.png") as image:
            prediction = np.asarray(image.convert("L")) > 0
        if prediction.shape != (source_size, source_size):
            raise ValueError(f"Unexpected frozen mask shape for {image_id}: {prediction.shape}")
        target = _resize(target_native, prediction.shape)
        native_area_ratio = float(target_native.mean())
        metrics = segmentation_metrics(prediction, target, compute_boundary=True)
        per_image.append(
            {
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "variant": args.variant_name,
                "evaluation_grid": f"{source_size}x{source_size}",
                "native_height": native_height,
                "native_width": native_width,
                "native_gt_area_ratio": native_area_ratio,
                "native_size_group": _size_group(native_area_ratio) if tumor else "normal",
                **metrics,
            }
        )

    if opened_annotations != 184:
        raise ValueError(f"opened {opened_annotations} annotations instead of 184")
    subgroup_counts = {
        group: sum(
            bool(row["gt_positive"]) and row["native_size_group"] == group
            for row in per_image
        )
        for group in SIZE_GROUPS
    }
    if subgroup_counts != {
        "small_lt_1pct": 94,
        "medium_1_to_5pct": 72,
        "large_ge_5pct": 18,
    }:
        raise ValueError(f"Canonical native-area subgroup counts differ: {subgroup_counts}")

    summary = summarize_segmentation_rows(per_image)
    tumor_rows = [row for row in per_image if bool(row["gt_positive"])]
    summary["native_subgroups"] = {
        group: summarize_segmentation_rows(
            [row for row in tumor_rows if row["native_size_group"] == group]
        )
        for group in SIZE_GROUPS
    }
    summary["group_bootstrap_ci95"] = bootstrap_group_confidence_intervals(
        per_image,
        iterations=args.bootstrap_iterations,
        seed=20260806,
    )

    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(json_safe(per_image))
    report = {
        "schema_version": 1,
        "study": "G4 frozen pseudo-mask variant actual segmentation evaluation",
        "variant": args.variant_name,
        "evaluation_grid": f"{source_size}x{source_size}",
        "split_sha256": args.expected_split_sha256,
        "images": 371,
        "tumor_images": 184,
        "native_subgroup_counts": subgroup_counts,
        "summary": summary,
        "prediction_bytes_verified_before_annotations": True,
        "validation_annotations_opened": opened_annotations,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "pass": True,
        "variant": args.variant_name,
        "split_sha256": args.expected_split_sha256,
        "pseudo_summary_sha256": args.expected_pseudo_summary_sha256,
        "pseudo_manifest_sha256": integrity["manifest_sha256"],
        "prediction_bytes_verified_before_annotations": True,
        "validation_annotations_opened": opened_annotations,
        "test_images_read": 0,
        "test_evaluated": False,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(report_path),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
