from __future__ import annotations

"""Post-freeze evaluation for the RAD-DINO geodesic seed-expansion probe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets.btxrd import BTXRDSegmentationDataset
from mae_reconstruction_io import sha256_file, validate_sha256


METRICS = (
    "pixel_ap",
    "pixel_auroc",
    "argmax_hit",
    "saliency_mass_in_gt",
    "dice_p90",
    "dice_p95",
    "dice_p97",
    "dice_p99",
)
ABSOLUTE_MINIMUM = {
    "image_level_auroc_from_raw_p99": 0.65,
    "overall_pixel_auroc": 0.75,
    "small_pixel_auroc": 0.77,
    "overall_dice_p90": 0.10,
    "small_dice_p97": 0.03,
    "medium_dice_p90": 0.12,
    "large_dice_p90": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-generation-sha256", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return (
        1.0
        if denominator == 0
        else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator
    )


def _size_group(area_ratio: float) -> str:
    return "small" if area_ratio < 0.01 else (
        "medium" if area_ratio < 0.05 else "large"
    )


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n": len(rows),
        **{
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        },
    }


def paired_group_report(
    rows: list[tuple[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    groups: dict[str, list[float]] = {}
    for group_id, delta in rows:
        groups.setdefault(group_id, []).append(float(delta))
    group_ids = sorted(groups)
    if not group_ids or iterations <= 0:
        raise ValueError("Paired bootstrap requires groups and iterations")
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [value for group in sampled for value in groups[str(group)]]
        boot[index] = float(np.mean(values))
    return {
        "mean_delta": float(np.mean([delta for _, delta in rows])),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "images": len(rows),
        "groups": len(group_ids),
        "iterations": iterations,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    expected_manifest = validate_sha256(
        args.expected_manifest_sha256, name="prediction manifest SHA-256"
    )
    expected_generation = validate_sha256(
        args.expected_generation_sha256, name="generation metadata SHA-256"
    )
    expected_freeze = validate_sha256(
        args.expected_freeze_sha256, name="prediction freeze SHA-256"
    )
    expected_baseline = validate_sha256(
        args.expected_baseline_sha256, name="baseline per-image SHA-256"
    )
    prediction_dir = args.prediction_root / "predictions"
    manifest_path = prediction_dir / "prediction_manifest.csv"
    generation_path = prediction_dir / "generation_metadata.json"
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("Prediction manifest differs from frozen hash")
    if sha256_file(generation_path) != expected_generation:
        raise ValueError("Generation metadata differs from frozen hash")
    if sha256_file(freeze_path) != expected_freeze:
        raise ValueError("Prediction freeze differs from frozen hash")

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if generation.get("stage") != "prediction-first RAD-DINO geodesic seed expansion":
        raise ValueError("Generation stage mismatch")
    if (
        generation.get("validation_gt_read") is not False
        or generation.get("consumer_trained") is not False
        or generation.get("test_evaluated") is not False
    ):
        raise ValueError("Generation metadata violates GT/consumer/test locks")
    expected_freeze_bindings = {
        "prediction_manifest_sha256": expected_manifest,
        "generation_metadata_sha256": expected_generation,
        "validation_predictions": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    for key, expected in expected_freeze_bindings.items():
        if freeze.get(key) != expected:
            raise ValueError(f"Prediction freeze mismatch: {key}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if (
        len(manifest) != 371
        or len({row["image_id"] for row in manifest}) != 371
        or sum(int(row["tumor"]) for row in manifest) != 184
    ):
        raise ValueError("Frozen prediction cohort differs from 371/184/187")
    physical_bytes = 0
    for row in manifest:
        path = prediction_dir / row["map_path"]
        if not path.is_file() or sha256_file(path) != row["map_sha256"]:
            raise ValueError(f"Frozen map missing/hash-mismatched: {row['image_id']}")
        values = np.load(path, allow_pickle=False)
        if (
            values.dtype != np.float16
            or values.shape != (320, 320)
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"Invalid frozen map: {row['image_id']}")
        if int(row["tumor"]) == 0 and np.count_nonzero(values):
            raise ValueError(f"Normal prediction is non-empty: {row['image_id']}")
        physical_bytes += path.stat().st_size
    if int(freeze.get("physical_map_bytes", -1)) != physical_bytes:
        raise ValueError("Physical map byte count differs from prediction freeze")

    # First validation-spatial-GT access: all predictions and hashes are frozen.
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
        values = np.load(
            prediction_dir / row["map_path"], allow_pickle=False
        ).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        image_scores.append(float(row["raw_p99"]))
        image_labels.append(int(row["tumor"]))
        if not int(row["tumor"]):
            continue
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        area_ratio = float(target.mean())
        record: dict[str, object] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": area_ratio,
            "size_group": _size_group(area_ratio),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(target.reshape(-1)[int(np.argmax(flat_values))]),
            "saliency_mass_in_gt": float(values[target].sum())
            / max(float(values.sum()), 1e-12),
        }
        for percentile in (90, 95, 97, 99):
            record[f"dice_p{percentile}"] = _dice(
                values >= float(np.percentile(values, percentile)),
                target,
            )
        evaluated.append(record)
    subgroup_counts = {
        group: sum(row["size_group"] == group for row in evaluated)
        for group in ("small", "medium", "large")
    }
    if len(evaluated) != 184 or subgroup_counts != {
        "small": 94,
        "medium": 72,
        "large": 18,
    }:
        raise ValueError("Tumor/subgroup evaluation cohort mismatch")

    if sha256_file(args.baseline_per_image) != expected_baseline:
        raise ValueError("Affinity-decoder baseline hash mismatch")
    with args.baseline_per_image.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    if len(baseline_rows) != 184:
        raise ValueError("Affinity-decoder baseline must contain 184 tumors")
    baseline = {row["image_id"]: row for row in baseline_rows}
    candidate = {str(row["image_id"]): row for row in evaluated}
    if set(baseline) != set(candidate):
        raise ValueError("Candidate and affinity-decoder baseline cohorts differ")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    localization = {
        "overall": _summarize(evaluated),
        **{
            group: _summarize(
                [row for row in evaluated if row["size_group"] == group]
            )
            for group in ("small", "medium", "large")
        },
    }
    image_auroc = float(roc_auc_score(image_labels, image_scores))
    summary = {
        "scientific_role": "prediction-first geodesic mechanism probe; not final segmentation",
        "prediction_manifest_sha256": expected_manifest,
        "generation_metadata_sha256": expected_generation,
        "prediction_freeze_sha256": expected_freeze,
        "per_image_sha256": sha256_file(per_image_path),
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "subgroups": subgroup_counts,
        "image_level_auroc_from_raw_p99": image_auroc,
        "tumor_localization": localization,
        "thresholds": "fixed diagnostic percentiles only; no threshold selected",
        "complete_misses_included": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison_metrics: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        strata: dict[str, object] = {}
        for stratum_index, stratum in enumerate(
            ("overall", "small", "medium", "large")
        ):
            names = [
                name
                for name, row in candidate.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            strata[stratum] = paired_group_report(
                [
                    (
                        str(candidate[name]["group_id"]),
                        float(candidate[name][metric]) - float(baseline[name][metric]),
                    )
                    for name in names
                ],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + metric_index * 10 + stratum_index,
            )
        comparison_metrics[metric] = strata
    comparison = {
        "comparison": "geodesic refinement minus frozen affinity-decoder source map",
        "method": "paired complete-group bootstrap",
        "iterations": args.bootstrap_iterations,
        "seed_family": args.bootstrap_seed,
        "metrics": comparison_metrics,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    comparison_path = args.output_dir / "paired_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observed = {
        "image_level_auroc_from_raw_p99": image_auroc,
        "overall_pixel_auroc": localization["overall"]["pixel_auroc"],
        "small_pixel_auroc": localization["small"]["pixel_auroc"],
        "overall_dice_p90": localization["overall"]["dice_p90"],
        "small_dice_p97": localization["small"]["dice_p97"],
        "medium_dice_p90": localization["medium"]["dice_p90"],
        "large_dice_p90": localization["large"]["dice_p90"],
    }
    absolute_checks = {
        key: {
            "observed": float(observed[key]),
            "minimum": float(minimum),
            "pass": float(observed[key]) >= float(minimum),
        }
        for key, minimum in ABSOLUTE_MINIMUM.items()
    }
    p90 = comparison_metrics["dice_p90"]
    relative_checks = {
        "overall_dice_p90_ci95_low_above_zero": {
            "observed": float(p90["overall"]["ci95_low"]),
            "minimum_exclusive": 0.0,
            "pass": float(p90["overall"]["ci95_low"]) > 0.0,
        },
        "no_subgroup_mean_dice_p90_decrease": {
            "observed": {
                group: float(p90[group]["mean_delta"])
                for group in ("small", "medium", "large")
            },
            "minimum": 0.0,
            "pass": all(
                float(p90[group]["mean_delta"]) >= 0.0
                for group in ("small", "medium", "large")
            ),
        },
    }
    gate_pass = all(check["pass"] for check in absolute_checks.values()) and all(
        check["pass"] for check in relative_checks.values()
    )
    gate = {
        "gate_id": "rad_dino_geodesic_seed_expansion_gate_v1",
        "status": "PASS" if gate_pass else "FAIL",
        "all_checks_required": True,
        "absolute_checks": absolute_checks,
        "relative_checks": relative_checks,
        "decision": (
            "AUTHORIZE_SEPARATE_PARTIAL_LABEL_CONSUMER_PROTOCOL"
            if gate_pass
            else "REJECT_FIXED_GEODESIC_CONFIGURATION"
        ),
        "consumer_trained": False,
        "test_evaluated": False,
    }
    gate_path = args.output_dir / "gate_decision.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_sha256": sha256_file(summary_path),
                "comparison_sha256": sha256_file(comparison_path),
                "gate_sha256": sha256_file(gate_path),
                "gate_status": gate["status"],
                "decision": gate["decision"],
                "consumer_trained": False,
                "test_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
