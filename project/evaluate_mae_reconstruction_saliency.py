from __future__ import annotations

"""Evaluate frozen MAE maps; GT is opened only after the full hash audit."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__:
    from .datasets.btxrd import BTXRDSegmentationDataset
    from .mae_reconstruction_io import sha256_file, validate_sha256
else:
    from datasets.btxrd import BTXRDSegmentationDataset
    from mae_reconstruction_io import sha256_file, validate_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "pixel_ap", "pixel_auroc", "argmax_hit", "saliency_mass_in_gt",
        "dice_p90", "dice_p95", "dice_p97", "dice_p99",
    ]
    return {
        "n": len(rows),
        **{
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in metrics
        },
    }


def _size_group(area_ratio: float) -> str:
    return "small" if area_ratio < 0.01 else ("medium" if area_ratio < 0.05 else "large")


def main() -> None:
    args = parse_args()
    manifest_path = args.prediction_dir / "prediction_manifest.csv"
    expected = validate_sha256(args.expected_manifest_sha256, name="manifest SHA-256")
    if sha256_file(manifest_path) != expected:
        raise ValueError("Prediction manifest differs from frozen caller hash")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 371 or len({row["image_id"] for row in manifest}) != 371:
        raise ValueError("Frozen validation prediction cohort must contain 371 unique images")
    for row in manifest:
        path = args.prediction_dir / row["map_path"]
        if not path.is_file() or sha256_file(path) != row["map_sha256"]:
            raise ValueError(f"Frozen map missing or hash-mismatched: {row['image_id']}")
        values = np.load(path, allow_pickle=False)
        if values.shape != (320, 320) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen map: {row['image_id']}")
    generation_metadata_path = args.prediction_dir / "generation_metadata.json"
    generation_metadata = json.loads(generation_metadata_path.read_text(encoding="utf-8"))
    if generation_metadata["validation_gt_read"] or generation_metadata["test_evaluated"]:
        raise ValueError("Prediction generation violates locked data-access contract")

    # GT access begins only after all 371 prediction files and hashes pass.
    dataset = BTXRDSegmentationDataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    gt_by_name: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _, mask, image_name = dataset[index]
        gt_by_name[str(image_name)] = mask[0].numpy() > 0.5
    if set(gt_by_name) != {row["image_id"] for row in manifest}:
        raise RuntimeError("GT cohort differs from frozen prediction cohort")

    evaluated: list[dict[str, object]] = []
    image_scores: list[float] = []
    image_labels: list[int] = []
    for row in manifest:
        values = np.load(args.prediction_dir / row["map_path"], allow_pickle=False).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        image_scores.append(float(row["raw_p99"]))
        image_labels.append(int(row["tumor"]))
        if not int(row["tumor"]):
            continue
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        area_ratio = float(target.mean())
        mass_total = float(values.sum())
        record: dict[str, object] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": area_ratio,
            "size_group": _size_group(area_ratio),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(target.reshape(-1)[int(np.argmax(flat_values))]),
            "saliency_mass_in_gt": float(values[target].sum()) / max(mass_total, 1e-12),
        }
        for percentile in (90, 95, 97, 99):
            record[f"dice_p{percentile}"] = _dice(
                values >= float(np.percentile(values, percentile)), target
            )
        evaluated.append(record)

    if len(evaluated) != 184:
        raise ValueError(f"Expected 184 validation tumor images, got {len(evaluated)}")
    subgroup_counts = {
        group: sum(row["size_group"] == group for row in evaluated)
        for group in ("small", "medium", "large")
    }
    if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
        raise ValueError(f"Unexpected subgroup contract: {subgroup_counts}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    summary = {
        "scientific_role": "SKELEX-inspired reconstruction mechanism feasibility; not final segmentation",
        "prediction_manifest_sha256": expected,
        "generation_metadata_sha256": sha256_file(generation_metadata_path),
        "per_image_sha256": sha256_file(per_image_path),
        "cohort": {"validation": 371, "tumor": 184, **subgroup_counts},
        "image_level_auroc_from_raw_p99": float(roc_auc_score(image_labels, image_scores)),
        "tumor_localization": {
            "overall": _summarize(evaluated),
            **{
                group: _summarize([row for row in evaluated if row["size_group"] == group])
                for group in ("small", "medium", "large")
            },
        },
        "thresholds": "fixed diagnostic percentiles only; no threshold selected",
        "consumer_trained": False,
        "complete_misses_included": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
