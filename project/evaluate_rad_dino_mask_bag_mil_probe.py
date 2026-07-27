from __future__ import annotations

"""Post-freeze evaluator for the RAD-DINO mask-bag MIL probe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    sha256_file,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


SUBGROUPS = ("overall", "small", "medium", "large")
OPERATIONAL_GOALS = {
    "overall": 0.34024039,
    "small": 0.17895493,
    "medium": 0.51244178,
    "large": 0.49370336,
}
ABSOLUTE_GATE = {
    "overall": 0.250,
    "small": 0.130,
    "medium": 0.370,
    "large": 0.380,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-baseline-per-image-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    return parser.parse_args()


def _size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def _load_and_verify_predictions(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("Prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("source_commit") != args.expected_source_commit
        or freeze.get("protocol_sha256") != args.expected_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Prediction freeze provenance/safety contract mismatch")
    manifest_path = args.prediction_root / "predictions" / "prediction_manifest.csv"
    if sha256_file(manifest_path) != freeze.get("prediction_manifest_sha256"):
        raise ValueError("Prediction manifest differs from prediction freeze")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    expected = {row["image_id"]: row for row in val_rows}
    indexed = {row["image_id"]: row for row in predictions}
    if len(predictions) != 371 or len(indexed) != 371 or set(indexed) != set(expected):
        raise ValueError("Prediction manifest does not cover the complete validation split")
    for image_id, prediction in indexed.items():
        if (
            prediction["group_id"] != expected[image_id]["group_id"]
            or prediction["tumor"] != expected[image_id]["tumor"]
        ):
            raise ValueError(f"Prediction identity/label mismatch: {image_id}")
        map_path = args.prediction_root / "predictions" / prediction["map_path"]
        if not map_path.is_file() or sha256_file(map_path) != prediction["map_sha256"]:
            raise ValueError(f"Prediction map hash mismatch: {image_id}")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"Prediction map content mismatch: {image_id}")
    return [indexed[row["image_id"]] for row in val_rows], freeze


def _paired_group_bootstrap(
    candidate: list[float],
    baseline: list[float],
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(candidate, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    group_values: dict[str, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        group_values.setdefault(group, []).append(float(value))
    unique = sorted(group_values)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.integers(0, len(unique), size=len(unique))
        rows = [value for position in sampled for value in group_values[unique[position]]]
        bootstrap[index] = float(np.mean(rows))
    return {
        "delta_candidate_minus_baseline": float(values.mean()),
        "ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_images": len(values),
        "n_groups": len(unique),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10000:
        raise ValueError("The frozen evaluator requires exactly 10,000 replicates")
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    predictions, freeze = _load_and_verify_predictions(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("Validation candidate audit must cover the full split")
    if (
        freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
    ):
        raise ValueError("Prediction freeze is not bound to this candidate gallery")
    if sha256_file(args.baseline_per_image) != args.expected_baseline_per_image_sha256:
        raise ValueError("Frozen baseline per-image SHA-256 mismatch")

    # Protocol boundary: all prediction/candidate files and hashes above are
    # verified before constructing the segmentation dataset or opening a mask.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    prediction_by_id = {row["image_id"]: row for row in predictions}
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_name = dataset[index]
        prediction = prediction_by_id[str(image_name)]
        if prediction["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        map_path = args.prediction_root / "predictions" / prediction["map_path"]
        values = np.load(map_path, allow_pickle=False).astype(np.float32)
        selected = values > 0.0
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        candidate_manifest_row = candidate_rows[Path(str(image_name)).stem]
        candidate_path = args.val_candidate_root / candidate_manifest_row["diagnostic_path"]
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"].astype(bool)
        proposal_dice = [_dice(proposal, target) for proposal in proposals]
        oracle = max(proposal_dice, default=0.0)
        area_ratio = float(target.mean())
        per_image.append(
            {
                "image_id": str(image_name),
                "group_id": prediction["group_id"],
                "gt_area_ratio": area_ratio,
                "size_group": _size_group(area_ratio),
                "dice": _dice(selected, target),
                "oracle_best_single_dice": oracle,
                "pixel_ap": float(average_precision_score(flat_target, flat_values)),
                "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
                "complete_miss": int(not np.logical_and(selected, target).any()),
                "selected_area_ratio": float(selected.mean()),
            }
        )
    if len(per_image) != 184:
        raise RuntimeError("Post-freeze tumor cohort must contain 184 images")
    subgroup_counts = {
        name: sum(row["size_group"] == name for row in per_image)
        for name in ("small", "medium", "large")
    }
    if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"Frozen subgroup cohort mismatch: {subgroup_counts}")

    with args.baseline_per_image.open("r", encoding="utf-8-sig", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    baseline_by_id = {
        row["image_name"]: row
        for row in baseline_rows
        if row.get("gt_positive", "").lower() == "true"
    }
    if set(baseline_by_id) != {str(row["image_id"]) for row in per_image}:
        raise ValueError("Frozen baseline tumor identities differ from candidate")

    summary_metrics: dict[str, dict[str, float | int]] = {}
    oracle_metrics: dict[str, float] = {}
    paired: dict[str, dict[str, object]] = {}
    baseline_misses: dict[str, int] = {}
    for subgroup in SUBGROUPS:
        rows = [
            row
            for row in per_image
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        candidate_values = [float(row["dice"]) for row in rows]
        baseline_values = [float(baseline_by_id[str(row["image_id"])]["dice"]) for row in rows]
        groups = [str(row["group_id"]) for row in rows]
        summary_metrics[subgroup] = {
            "n": len(rows),
            "dice": float(np.mean(candidate_values)),
            "pixel_ap": float(np.mean([float(row["pixel_ap"]) for row in rows])),
            "pixel_auroc": float(
                np.mean([float(row["pixel_auroc"]) for row in rows])
            ),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in rows)),
        }
        oracle_metrics[subgroup] = float(
            np.mean([float(row["oracle_best_single_dice"]) for row in rows])
        )
        paired[subgroup] = _paired_group_bootstrap(
            candidate_values,
            baseline_values,
            groups,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + SUBGROUPS.index(subgroup),
        )
        baseline_misses[subgroup] = int(sum(value == 0.0 for value in baseline_values))

    image_labels = np.asarray([int(row["tumor"]) for row in predictions], dtype=np.int32)
    image_scores = np.asarray(
        [float(row["bag_probability"]) for row in predictions], dtype=np.float64
    )
    image_auroc = float(roc_auc_score(image_labels, image_scores))
    absolute_checks = {
        subgroup: {
            "observed": summary_metrics[subgroup]["dice"],
            "minimum": ABSOLUTE_GATE[subgroup],
            "pass": summary_metrics[subgroup]["dice"] >= ABSOLUTE_GATE[subgroup],
        }
        for subgroup in SUBGROUPS
    }
    oracle_checks = {
        subgroup: {
            "observed": oracle_metrics[subgroup],
            "minimum": OPERATIONAL_GOALS[subgroup],
            "pass": oracle_metrics[subgroup] >= OPERATIONAL_GOALS[subgroup],
        }
        for subgroup in SUBGROUPS
    }
    relative_checks = {
        "overall_ci95_low_above_zero": {
            "observed": paired["overall"]["ci95"][0],
            "minimum_exclusive": 0.0,
            "pass": paired["overall"]["ci95"][0] > 0.0,
        },
        "no_subgroup_mean_decrease": {
            "observed": {
                subgroup: paired[subgroup]["delta_candidate_minus_baseline"]
                for subgroup in ("small", "medium", "large")
            },
            "minimum": 0.0,
            "pass": all(
                paired[subgroup]["delta_candidate_minus_baseline"] >= 0.0
                for subgroup in ("small", "medium", "large")
            ),
        },
        "no_complete_miss_increase": {
            "observed": {
                subgroup: summary_metrics[subgroup]["complete_misses"]
                - baseline_misses[subgroup]
                for subgroup in SUBGROUPS
            },
            "maximum": 0,
            "pass": all(
                summary_metrics[subgroup]["complete_misses"]
                <= baseline_misses[subgroup]
                for subgroup in SUBGROUPS
            ),
        },
    }
    image_check = {
        "observed": image_auroc,
        "minimum": 0.75,
        "pass": image_auroc >= 0.75,
    }
    passed = (
        all(check["pass"] for check in absolute_checks.values())
        and all(check["pass"] for check in oracle_checks.values())
        and all(check["pass"] for check in relative_checks.values())
        and image_check["pass"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary = {
        "arm": "rad_dino_mask_bag_mil_selected_sam_proposal",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "subgroups": subgroup_counts,
        "image_level_auroc": image_auroc,
        "tumor_localization": summary_metrics,
        "candidate_oracle": oracle_metrics,
        "complete_misses_included": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    paired_payload = {
        "comparison": "mask-bag MIL selected proposal minus promoted flip-TTA selector",
        "baseline_per_image_sha256": args.expected_baseline_per_image_sha256,
        "method": "paired complete-group bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed_family": args.bootstrap_seed,
        "metrics": {"dice": paired},
        "consumer_trained": False,
        "test_evaluated": False,
    }
    gate = {
        "gate_id": "rad_dino_mask_bag_mil_prediction_gate_v1",
        "status": "PASS" if passed else "FAIL",
        "all_checks_required": True,
        "image_level_auroc": image_check,
        "absolute_dice_checks": absolute_checks,
        "candidate_oracle_checks": oracle_checks,
        "relative_checks": relative_checks,
        "on_pass": "authorize only a separately predeclared pseudo-mask consumer; do not train automatically",
        "on_fail": "reject the exact mask-bag configuration without validation retuning",
        "consumer_trained": False,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    paired_path = args.output_dir / "paired_comparison.json"
    gate_path = args.output_dir / "gate_decision.json"
    for path, payload in (
        (summary_path, summary),
        (paired_path, paired_payload),
        (gate_path, gate),
    ):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    audit = {
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "prediction_manifest_sha256": freeze["prediction_manifest_sha256"],
        "candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "baseline_per_image_sha256": args.expected_baseline_per_image_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "paired_comparison_sha256": sha256_file(paired_path),
        "gate_decision_sha256": sha256_file(gate_path),
        "cohort": {"validation": 371, "tumor": 184, "normal": 187, **subgroup_counts},
        "complete_misses_included": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
