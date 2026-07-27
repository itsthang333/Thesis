"""Post-freeze validation evaluator for the global-local MIL probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_nominal_patch_memory_arms import METRICS, paired_group_bootstrap
from evaluate_rad_dino_multilayer_soft_region_probe import _dice, _subgroup
from mae_reconstruction_io import sha256_file


GATE_THRESHOLDS = {
    "image_level_auroc_from_raw_p99": 0.75,
    "overall_dice_p90": 0.145,
    "small_dice_p90": 0.025,
    "small_dice_p99": 0.060,
    "medium_dice_p90": 0.217,
    "large_dice_p90": 0.518,
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
    for record in (freeze, run_manifest):
        if (
            record["source_commit"] != args.source_commit
            or record["protocol_sha256"] != args.protocol_sha256
            or record["validation_gt_read"] is not False
            or record["consumer_trained"] is not False
            or record["test_evaluated"] is not False
        ):
            raise RuntimeError("Pre-GT provenance contract mismatch")
    if (
        freeze["split_sha256"] != args.expected_split_sha256
        or freeze["validation_predictions"] != 371
        or sha256(manifest_path) != freeze["prediction_manifest_sha256"]
    ):
        raise RuntimeError("Prediction freeze binding mismatch")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 371 or len({row["image_id"] for row in manifest}) != 371:
        raise RuntimeError("Prediction manifest cohort mismatch")
    expected_fused = {row["map_path"] for row in manifest}
    expected_local = {row["local_map_path"] for row in manifest}
    observed_fused = {
        path.relative_to(prediction_dir).as_posix()
        for path in (prediction_dir / "maps").glob("*.npy")
    }
    observed_local = {
        path.relative_to(prediction_dir).as_posix()
        for path in (prediction_dir / "local_maps").glob("*.npy")
    }
    if observed_fused != expected_fused or observed_local != expected_local:
        raise RuntimeError("Manifest and physical fused/local maps differ")
    for row in manifest:
        checks = (
            (row["map_path"], row["map_sha256"]),
            (row["local_map_path"], row["local_map_sha256"]),
        )
        for relative, expected in checks:
            if sha256(prediction_dir / relative) != expected:
                raise RuntimeError(f"Map hash mismatch: {row['image_id']}")
    return manifest, freeze


def load_validation_gt(
    args: argparse.Namespace,
    manifest: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    from datasets.btxrd import BTXRDSegmentationDataset

    dataset = BTXRDSegmentationDataset(
        root=args.dataset_root,
        split="val",
        image_size=args.output_size,
        augment=False,
        split_manifest=args.split_manifest,
    )
    gt_by_name = {
        str(name): mask[0].numpy() > 0.5 for _image, mask, name in dataset
    }
    if set(gt_by_name) != {row["image_id"] for row in manifest}:
        raise RuntimeError("Frozen prediction and validation-GT cohorts differ")
    return gt_by_name


def evaluate_arm(
    args: argparse.Namespace,
    manifest: list[dict[str, str]],
    gt_by_name: dict[str, np.ndarray],
    *,
    map_field: str,
) -> list[dict[str, object]]:
    prediction_dir = args.output_dir / "predictions"
    evaluated: list[dict[str, object]] = []
    for row in manifest:
        if row["tumor"] != "1":
            continue
        values = np.load(
            prediction_dir / row[map_field], allow_pickle=False
        ).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        item: dict[str, object] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": float(target.mean()),
            "size_group": _subgroup(float(target.mean())),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(target.reshape(-1)[int(np.argmax(flat_values))]),
            "saliency_mass_in_gt": float(values[target].sum())
            / max(float(values.sum()), 1.0e-12),
        }
        for percentile in (90, 95, 97, 99):
            item[f"dice_p{percentile}"] = _dice(
                values >= np.percentile(values, percentile), target
            )
        evaluated.append(item)
    counts = {
        name: sum(row["size_group"] == name for row in evaluated)
        for name in ("small", "medium", "large")
    }
    if len(evaluated) != 184 or counts != {
        "small": 94,
        "medium": 72,
        "large": 18,
    }:
        raise RuntimeError("Tumor/subgroup cohort mismatch")
    return evaluated


def summarize_arm(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "tumor_localization": {},
        "complete_misses": {},
    }
    for name in ("overall", "small", "medium", "large"):
        selected = [
            row
            for row in rows
            if name == "overall" or row["size_group"] == name
        ]
        summary["tumor_localization"][name] = {
            "n": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in METRICS
            },
        }
    for percentile in (90, 95, 97, 99):
        metric = f"dice_p{percentile}"
        summary["complete_misses"][metric] = {
            name: sum(
                float(row[metric]) == 0.0
                for row in rows
                if name == "overall" or row["size_group"] == name
            )
            for name in ("overall", "small", "medium", "large")
        }
    return summary


def evaluate_frozen_predictions(
    args: argparse.Namespace,
    manifest: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    gt_by_name = load_validation_gt(args, manifest)
    fused = evaluate_arm(args, manifest, gt_by_name, map_field="map_path")
    local = evaluate_arm(args, manifest, gt_by_name, map_field="local_map_path")
    evaluation_dir = args.output_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=False)
    for filename, rows in (("per_image.csv", fused), ("local_per_image.csv", local)):
        with (evaluation_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    image_labels = np.asarray([int(row["tumor"]) for row in manifest])
    image_scores = np.asarray([float(row["raw_p99"]) for row in manifest])
    summary = {
        "arm": "rad_dino_global_local_mil_fused",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "subgroups": {"small": 94, "medium": 72, "large": 18},
        "image_level_auroc_from_raw_p99": float(
            roc_auc_score(image_labels, image_scores)
        ),
        **summarize_arm(fused),
        "local_only_diagnostic": summarize_arm(local),
        "prediction_manifest_sha256": sha256(
            args.output_dir / "predictions/prediction_manifest.csv"
        ),
        "per_image_sha256": sha256(evaluation_dir / "per_image.csv"),
        "local_per_image_sha256": sha256(
            evaluation_dir / "local_per_image.csv"
        ),
        "validation_gt_read_only_after_prediction_freeze": True,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return fused, summary


def compare_to_global(
    candidate_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    if sha256(args.baseline_per_image) != args.expected_baseline_sha256:
        raise RuntimeError("Frozen global baseline hash mismatch")
    with args.baseline_per_image.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        baseline = {row["image_id"]: row for row in csv.DictReader(handle)}
    candidate = {str(row["image_id"]): row for row in candidate_rows}
    if baseline.keys() != candidate.keys() or len(candidate) != 184:
        raise RuntimeError("Candidate and global baseline cohorts differ")
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
                seed=20260901 + metric_index * 10 + stratum_index,
            )
            statistics["delta_candidate_minus_global"] = statistics.pop(
                "delta_multiscale_minus_single_scale"
            )
            results[metric][stratum] = statistics
    return {
        "comparison": "global-local fused minus frozen multi-layer global v3",
        "baseline_per_image_sha256": args.expected_baseline_sha256,
        "method": "paired complete-group bootstrap",
        "replicates": 10_000,
        "seed_family": 20260901,
        "metrics": results,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def apply_gate(
    summary: dict[str, object], comparison: dict[str, object]
) -> dict[str, object]:
    localization = summary["tumor_localization"]
    observed = {
        "image_level_auroc_from_raw_p99": summary[
            "image_level_auroc_from_raw_p99"
        ],
        "overall_dice_p90": localization["overall"]["dice_p90"],
        "small_dice_p90": localization["small"]["dice_p90"],
        "small_dice_p99": localization["small"]["dice_p99"],
        "medium_dice_p90": localization["medium"]["dice_p90"],
        "large_dice_p90": localization["large"]["dice_p90"],
    }
    absolute = {
        name: {
            "observed": float(observed[name]),
            "minimum": float(minimum),
            "pass": float(observed[name]) >= minimum,
        }
        for name, minimum in GATE_THRESHOLDS.items()
    }
    dice = comparison["metrics"]["dice_p90"]
    deltas = {
        name: float(dice[name]["delta_candidate_minus_global"])
        for name in ("overall", "small", "medium", "large")
    }
    small_ci_low = float(dice["small"]["ci95"][0])
    small_misses = int(summary["complete_misses"]["dice_p90"]["small"])
    relative = {
        "small_dice_p90_ci95_low_above_zero": {
            "observed": small_ci_low,
            "minimum_exclusive": 0.0,
            "pass": small_ci_low > 0.0,
        },
        "no_mean_dice_p90_decrease": {
            "observed": deltas,
            "minimum": 0.0,
            "pass": all(value >= 0.0 for value in deltas.values()),
        },
        "small_complete_misses_below_global": {
            "observed": small_misses,
            "maximum_exclusive": 35,
            "pass": small_misses < 35,
        },
    }
    passed = all(check["pass"] for check in absolute.values()) and all(
        check["pass"] for check in relative.values()
    )
    return {
        "gate_id": "rad_dino_global_local_mil_prediction_gate_v1",
        "status": "PASS" if passed else "FAIL",
        "all_checks_required": True,
        "absolute_checks": absolute,
        "relative_checks": relative,
        "on_pass": (
            "authorize only a separately predeclared size-balanced pseudo-mask "
            "consumer; do not train it automatically"
        ),
        "on_fail": "reject the exact global-local configuration without retuning",
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
    comparison = compare_to_global(evaluated, args)
    evaluation_dir = args.output_dir / "evaluation"
    comparison_path = evaluation_dir / "paired_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    gate = apply_gate(summary, comparison)
    gate_path = evaluation_dir / "gate_decision.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    audit = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": sha256(
            args.output_dir / "prediction_freeze.json"
        ),
        "prediction_manifest_sha256": freeze["prediction_manifest_sha256"],
        "summary_sha256": sha256(evaluation_dir / "summary.json"),
        "per_image_sha256": sha256(evaluation_dir / "per_image.csv"),
        "local_per_image_sha256": sha256(
            evaluation_dir / "local_per_image.csv"
        ),
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
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "comparison": comparison, "gate": gate, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
