from __future__ import annotations

"""Evaluate an audited X4 four-arm gate freeze on canonical validation GT."""

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
from frozen_io import load_split_rows_without_annotations, sha256_file
from x4_contract import CANONICAL_SPLIT_SHA256, GATE_ARMS, STUDENT_SEEDS, load_x4_protocol


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
    parser.add_argument("--seed", type=int, choices=STUDENT_SEEDS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    protocol, protocol_sha = load_x4_protocol(Path(__file__).resolve().parents[1])
    freeze_path = args.prediction_root / "prediction_freeze.json"
    manifest_path = args.prediction_root / "prediction_manifest.csv"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("X4 gate freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("stage") != "x4_gate_prediction_freeze_v1"
        or int(freeze.get("seed", -1)) != args.seed
        or tuple(freeze.get("arms", ())) != GATE_ARMS
        or freeze.get("split_sha256") != CANONICAL_SPLIT_SHA256
        or freeze.get("x4_protocol_sha256") != protocol_sha
        or int(freeze.get("images_per_arm", -1)) != 371
        or freeze.get("prediction_manifest_sha256") != sha256_file(manifest_path)
        or freeze.get("predictions_frozen_before_spatial_ground_truth") is not True
        or freeze.get("validation_annotations_read") != 0
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("X4 gate freeze violates Stage A")
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split="val",
        allow_test=False,
    )
    manifest_rows = read_csv(manifest_path)
    indexed = {(row["arm"], row["image_id"]): row for row in manifest_rows}
    expected_keys = {(arm, row["image_id"]) for arm in GATE_ARMS for row in split_rows}
    if len(manifest_rows) != len(indexed) or set(indexed) != expected_keys:
        raise ValueError("X4 gate prediction manifest cohort differs")
    for arm, image_id in sorted(expected_keys):
        row = indexed[(arm, image_id)]
        relative = Path(row["mask_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe X4 gate mask path")
        mask_path = args.prediction_root / relative
        if sha256_file(mask_path) != row["mask_sha256"]:
            raise ValueError(f"X4 gate mask changed: {arm}/{image_id}")
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"))
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError(f"X4 gate mask is not binary: {arm}/{image_id}")

    # Annotation boundary: every arm's 1,484 mask bytes are immutable above this line.
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    per_image: list[dict[str, object]] = []
    annotations_opened = 0
    for split_row in split_rows:
        image_id = split_row["image_id"]
        with Image.open(btxrd_root / "images" / image_id) as handle:
            width, height = handle.size
        if int(split_row["tumor"]):
            target = _decode_labelme_polygon_mask(
                btxrd_root / "Annotations" / f"{Path(image_id).stem}.json",
                height=height,
                width=width,
            )
            annotations_opened += 1
        else:
            target = np.zeros((height, width), dtype=bool)
        area_ratio = float(target.mean())
        native_group = size_group(area_ratio) if area_ratio else "normal"
        for arm in GATE_ARMS:
            manifest = indexed[(arm, image_id)]
            with Image.open(args.prediction_root / manifest["mask_path"]) as handle:
                prediction = np.asarray(
                    handle.convert("L").resize((width, height), Image.Resampling.NEAREST)
                ) > 0
            per_image.append(
                {
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "group_source": split_row.get("group_source", ""),
                    "center": split_row.get("center", ""),
                    "anatomy": split_row.get("anatomy", ""),
                    "view": split_row.get("view", ""),
                    "tumor_type_name": split_row.get("tumor_type_name", ""),
                    "arm": arm,
                    "seed": args.seed,
                    "evaluation_grid": "native",
                    "native_height": height,
                    "native_width": width,
                    "native_size_group": native_group,
                    "gate_probability": manifest["gate_probability"],
                    "gate_positive": manifest["gate_positive"],
                    **segmentation_metrics(prediction, target, compute_boundary=True),
                }
            )
    if annotations_opened != 184:
        raise ValueError("X4 gate evaluator did not open exactly 184 validation annotations")

    arm_summaries: dict[str, object] = {}
    subgroup_rows: list[dict[str, object]] = []
    for arm in GATE_ARMS:
        rows = [row for row in per_image if row["arm"] == arm]
        counts = {group: sum(row["native_size_group"] == group for row in rows) for group in SIZE_GROUPS}
        if counts != {"small_lt_1pct": 94, "medium_1_to_5pct": 72, "large_ge_5pct": 18}:
            raise ValueError(f"X4 gate subgroup counts differ for {arm}: {counts}")
        summary = summarize_segmentation_rows(rows)
        summary["native_subgroups"] = {
            group: summarize_segmentation_rows(
                [row for row in rows if row["native_size_group"] == group]
            )
            for group in SIZE_GROUPS
        }
        summary["group_bootstrap_ci95"] = bootstrap_group_confidence_intervals(
            rows,
            iterations=int(protocol["paired_bootstrap"]["iterations"]),
            seed=int(protocol["paired_bootstrap"]["seed"]),
        )
        arm_summaries[arm] = summary
        subgroup_rows.extend({"arm": arm, **row} for row in subgroup_summaries(rows))

    args.output_dir.mkdir(parents=True)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(json_safe(per_image))
    subgroup_path = args.output_dir / "subgroups.csv"
    with subgroup_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(subgroup_rows))
    report = {
        "schema_version": 1,
        "study": "X4 inference-label assumption ablation",
        "seed": args.seed,
        "arms": list(GATE_ARMS),
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "x4_protocol_sha256": protocol_sha,
        "images_per_arm": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "summary_by_arm": arm_summaries,
        "prediction_bytes_verified_before_annotations": True,
        "validation_annotations_opened": annotations_opened,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "pass": True,
        "stage": "x4_gate_native_evaluation_audit_v1",
        "seed": args.seed,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "subgroups_sha256": sha256_file(subgroup_path),
        "summary_sha256": sha256_file(summary_path),
        "images_per_arm": 371,
        "validation_annotations_opened": annotations_opened,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
