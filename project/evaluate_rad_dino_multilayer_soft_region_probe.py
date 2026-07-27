from __future__ import annotations

"""Post-freeze validation evaluator for the multi-layer soft-region probe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_nominal_patch_memory_arms import METRICS, paired_group_bootstrap
from mae_reconstruction_io import sha256_file


GATE_THRESHOLDS = {
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
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-size", type=int, default=320)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return sha256_file(path)


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return (
        2.0
        * float(np.logical_and(prediction, target).sum())
        / float(denominator)
    )


def _subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def verify_prediction_freeze(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    freeze_path = args.output_dir / "prediction_freeze.json"
    run_manifest_path = args.output_dir / "run_manifest.json"
    prediction_dir = args.output_dir / "predictions"
    manifest_path = prediction_dir / "prediction_manifest.csv"
    if not freeze_path.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError("Prediction freeze/run manifest is missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        freeze["source_commit"] != args.source_commit
        or freeze["protocol_sha256"] != args.protocol_sha256
        or freeze["split_sha256"] != args.expected_split_sha256
        or freeze["validation_predictions"] != 371
        or freeze["validation_gt_read"] is not False
        or freeze["consumer_trained"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise RuntimeError("Prediction freeze contract mismatch")
    if (
        run_manifest["source_commit"] != args.source_commit
        or run_manifest["protocol_sha256"] != args.protocol_sha256
        or run_manifest["validation_gt_read"] is not False
        or run_manifest["consumer_trained"] is not False
        or run_manifest["test_evaluated"] is not False
    ):
        raise RuntimeError("Run manifest contract mismatch")
    if sha256(manifest_path) != freeze["prediction_manifest_sha256"]:
        raise RuntimeError("Frozen prediction-manifest hash mismatch")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if (
        len(manifest) != 371
        or len({row["image_id"] for row in manifest}) != 371
        or len({row["map_path"] for row in manifest}) != 371
    ):
        raise RuntimeError("Prediction manifest must contain 371 unique maps")
    expected_paths = {row["map_path"] for row in manifest}
    observed_paths = {
        path.relative_to(prediction_dir).as_posix()
        for path in (prediction_dir / "maps").glob("*.npy")
    }
    if observed_paths != expected_paths:
        raise RuntimeError("Frozen manifest and physical map set differ")
    for row in manifest:
        map_path = prediction_dir / row["map_path"]
        if sha256(map_path) != row["map_sha256"]:
            raise RuntimeError(f"Prediction map hash mismatch: {row['image_id']}")
    return manifest, freeze


def evaluate_frozen_predictions(
    args: argparse.Namespace,
    manifest: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from datasets.btxrd import BTXRDSegmentationDataset

    dataset = BTXRDSegmentationDataset(
        root=args.dataset_root,
        split="val",
        image_size=args.output_size,
        augment=False,
        split_manifest=args.split_manifest,
    )
    gt_by_name: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _image, mask, name = dataset[index]
        gt_by_name[str(name)] = mask[0].numpy() > 0.5
    if set(gt_by_name) != {row["image_id"] for row in manifest}:
        raise RuntimeError("Frozen prediction and validation-GT cohorts differ")
    prediction_dir = args.output_dir / "predictions"
    evaluated: list[dict[str, object]] = []
    for row in manifest:
        if row["tumor"] != "1":
            continue
        values = np.load(
            prediction_dir / row["map_path"],
            allow_pickle=False,
        ).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        item: dict[str, object] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": float(target.mean()),
            "size_group": _subgroup(float(target.mean())),
            "pixel_ap": float(
                average_precision_score(flat_target, flat_values)
            ),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(
                target.reshape(-1)[int(np.argmax(flat_values))]
            ),
            "saliency_mass_in_gt": float(values[target].sum())
            / max(float(values.sum()), 1.0e-12),
        }
        for percentile in (90, 95, 97, 99):
            item[f"dice_p{percentile}"] = _dice(
                values >= np.percentile(values, percentile),
                target,
            )
        evaluated.append(item)
    if len(evaluated) != 184:
        raise RuntimeError(f"Expected 184 tumor evaluations, got {len(evaluated)}")
    subgroup_counts = {
        name: sum(row["size_group"] == name for row in evaluated)
        for name in ("small", "medium", "large")
    }
    if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"Subgroup contract drift: {subgroup_counts}")
    evaluation_dir = args.output_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=False)
    per_image_path = evaluation_dir / "per_image.csv"
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    metrics = list(METRICS)
    image_labels = np.asarray(
        [int(row["tumor"]) for row in manifest],
        dtype=np.uint8,
    )
    image_scores = np.asarray(
        [float(row["raw_p99"]) for row in manifest],
        dtype=np.float64,
    )
    summary: dict[str, object] = {
        "arm": "rad_dino_multilayer_soft_region_decoder",
        "cohort": {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
        },
        "subgroups": subgroup_counts,
        "image_level_auroc_from_raw_p99": float(
            roc_auc_score(image_labels, image_scores)
        ),
        "tumor_localization": {},
        "complete_misses": {},
    }
    for name in ("overall", "small", "medium", "large"):
        selected = [
            row
            for row in evaluated
            if name == "overall" or row["size_group"] == name
        ]
        summary["tumor_localization"][name] = {
            "n": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in metrics
            },
        }
    for percentile in (90, 95, 97, 99):
        metric = f"dice_p{percentile}"
        summary["complete_misses"][metric] = {
            name: sum(
                float(row[metric]) == 0.0
                for row in evaluated
                if name == "overall" or row["size_group"] == name
            )
            for name in ("overall", "small", "medium", "large")
        }
    summary.update(
        {
            "prediction_manifest_sha256": sha256(
                prediction_dir / "prediction_manifest.csv"
            ),
            "per_image_sha256": sha256(per_image_path),
            "validation_gt_read_only_after_prediction_freeze": True,
            "complete_misses_included": True,
            "consumer_trained": False,
            "test_evaluated": False,
        }
    )
    summary_path = evaluation_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return evaluated, summary


def compare_to_affinity_source(
    candidate_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    if sha256(args.baseline_per_image) != args.expected_baseline_sha256:
        raise RuntimeError("Frozen affinity baseline hash mismatch")
    with args.baseline_per_image.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        baseline_rows = list(csv.DictReader(handle))
    baseline = {str(row["image_id"]): row for row in baseline_rows}
    candidate = {str(row["image_id"]): row for row in candidate_rows}
    if baseline.keys() != candidate.keys() or len(candidate) != 184:
        raise RuntimeError("Candidate and affinity baseline cohorts differ")
    results: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        results[metric] = {}
        for stratum_index, stratum in enumerate(
            ("overall", "small", "medium", "large")
        ):
            names = [
                name
                for name, row in candidate.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            statistics = paired_group_bootstrap(
                [
                    (
                        str(candidate[name]["group_id"]),
                        float(candidate[name][metric])
                        - float(baseline[name][metric]),
                    )
                    for name in names
                ],
                replicates=10_000,
                seed=20260827 + metric_index * 10 + stratum_index,
            )
            statistics["delta_candidate_minus_affinity"] = statistics.pop(
                "delta_multiscale_minus_single_scale"
            )
            results[metric][stratum] = statistics
    return {
        "comparison": (
            "multi-layer soft-region decoder minus frozen RAD-DINO "
            "affinity-decoder v3"
        ),
        "baseline_per_image_sha256": args.expected_baseline_sha256,
        "method": "paired complete-group bootstrap",
        "replicates": 10_000,
        "seed_family": 20260827,
        "metrics": results,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def apply_gate(
    summary: dict[str, object],
    comparison: dict[str, object],
) -> dict[str, object]:
    localization = summary["tumor_localization"]
    observed = {
        "image_level_auroc_from_raw_p99": summary[
            "image_level_auroc_from_raw_p99"
        ],
        "overall_pixel_auroc": localization["overall"]["pixel_auroc"],
        "small_pixel_auroc": localization["small"]["pixel_auroc"],
        "overall_dice_p90": localization["overall"]["dice_p90"],
        "small_dice_p97": localization["small"]["dice_p97"],
        "medium_dice_p90": localization["medium"]["dice_p90"],
        "large_dice_p90": localization["large"]["dice_p90"],
    }
    absolute_checks = {
        name: {
            "observed": float(observed[name]),
            "minimum": float(minimum),
            "pass": bool(float(observed[name]) >= float(minimum)),
        }
        for name, minimum in GATE_THRESHOLDS.items()
    }
    dice_comparison = comparison["metrics"]["dice_p90"]
    overall_ci = dice_comparison["overall"]["ci95"]
    if not isinstance(overall_ci, list) or len(overall_ci) != 2:
        raise ValueError("Overall Dice bootstrap CI must be [low, high]")
    overall_ci_low = float(overall_ci[0])
    subgroup_deltas = {
        name: float(
            dice_comparison[name]["delta_candidate_minus_affinity"]
        )
        for name in ("small", "medium", "large")
    }
    relative_checks = {
        "overall_dice_p90_ci95_low_above_zero": {
            "observed": overall_ci_low,
            "minimum_exclusive": 0.0,
            "pass": overall_ci_low > 0.0,
        },
        "no_subgroup_mean_dice_p90_decrease": {
            "observed": subgroup_deltas,
            "minimum": 0.0,
            "pass": all(value >= 0.0 for value in subgroup_deltas.values()),
        },
    }
    passed = all(
        check["pass"] for check in absolute_checks.values()
    ) and all(check["pass"] for check in relative_checks.values())
    return {
        "gate_id": "rad_dino_multilayer_soft_region_prediction_gate_v1",
        "status": "PASS" if passed else "FAIL",
        "all_checks_required": True,
        "absolute_checks": absolute_checks,
        "relative_checks": relative_checks,
        "on_pass": (
            "authorize only a separately predeclared partial-soft-label "
            "consumer protocol; do not train it automatically"
        ),
        "on_fail": (
            "reject this exact configuration without altering hidden layers, "
            "soft-region thresholds, losses, weights, TTA, or gate"
        ),
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if args.output_size != 320:
        raise ValueError("Frozen evaluation size is 320")
    if sha256(args.split_manifest) != args.expected_split_sha256:
        raise RuntimeError("Frozen split hash mismatch")
    manifest, freeze = verify_prediction_freeze(args)
    evaluated, summary = evaluate_frozen_predictions(args, manifest)
    comparison = compare_to_affinity_source(evaluated, args)
    evaluation_dir = args.output_dir / "evaluation"
    comparison_path = evaluation_dir / "paired_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    gate = apply_gate(summary, comparison)
    gate_path = evaluation_dir / "gate_decision.json"
    gate_path.write_text(
        json.dumps(gate, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": sha256(
            args.output_dir / "prediction_freeze.json"
        ),
        "prediction_manifest_sha256": freeze[
            "prediction_manifest_sha256"
        ],
        "summary_sha256": sha256(evaluation_dir / "summary.json"),
        "per_image_sha256": sha256(evaluation_dir / "per_image.csv"),
        "paired_comparison_sha256": sha256(comparison_path),
        "gate_decision_sha256": sha256(gate_path),
        "cohort": {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            "small": 94,
            "medium": 72,
            "large": 18,
        },
        "complete_misses_included": True,
        "bootstrap_replicates": 10_000,
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    audit_path = evaluation_dir / "evaluation_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "comparison": comparison,
                "gate": gate,
                "audit": audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
