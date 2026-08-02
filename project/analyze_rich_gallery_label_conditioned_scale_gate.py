from __future__ import annotations

"""Test whether supplied BTXRD image labels identify a useful extent expert.

This is a retrospective validation diagnostic.  Candidate choices and Dice
for a fixed beta grid were frozen by the earlier scale-feasibility analysis.
The present script only asks whether tumor subtype or benign/malignant labels
generalize as an extent-routing variable under group-separated folds.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BETAS = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
GROUPS = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-per-image", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-support", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_for_group(group_id: str, folds: int) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def beta_key(beta: float) -> str:
    return f"beta_{beta:g}_dice"


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def evaluate_label(
    rows: list[dict[str, Any]],
    label_key: str,
    folds: int,
    min_train_support: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        training = [
            candidate
            for candidate in rows
            if int(candidate["fold"]) != int(row["fold"])
            and str(candidate[label_key]) == str(row[label_key])
        ]
        if len(training) < min_train_support:
            selected_beta = 0.0
        else:
            means = {beta: mean(training, beta_key(beta)) for beta in BETAS}
            selected_beta = max(BETAS, key=lambda beta: (means[beta], -abs(beta), -beta))
        routed = float(row[beta_key(selected_beta)])
        predictions.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "fold": row["fold"],
                "size_group": row["size_group"],
                "label_name": label_key,
                "label_value": row[label_key],
                "selected_beta": selected_beta,
                "baseline_dice": float(row["baseline_dice"]),
                "routed_dice": routed,
                "delta": routed - float(row["baseline_dice"]),
                "outer_train_support": len(training),
            }
        )

    result: dict[str, Any] = {
        "overall": {
            "baseline_dice": mean(predictions, "baseline_dice"),
            "routed_dice": mean(predictions, "routed_dice"),
            "delta": mean(predictions, "delta"),
            "changed_beta_images": int(sum(float(row["selected_beta"]) != 0.0 for row in predictions)),
            "selected_beta_counts": dict(
                sorted(Counter(float(row["selected_beta"]) for row in predictions).items())
            ),
        },
        "subgroups": {},
    }
    for group in GROUPS:
        subset = [row for row in predictions if row["size_group"] == group]
        result["subgroups"][group] = {
            "n": len(subset),
            "baseline_dice": mean(subset, "baseline_dice"),
            "routed_dice": mean(subset, "routed_dice"),
            "delta": mean(subset, "delta"),
        }
    return result, predictions


def main() -> None:
    args = parse_args()
    with args.scale_per_image.open(newline="", encoding="utf-8-sig") as handle:
        scale_rows = list(csv.DictReader(handle))
    with args.split_manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = {
            str(row["image_id"]): row
            for row in csv.DictReader(handle)
            if row["split"] == "val" and int(row["eligible"]) == 1 and int(row["tumor"]) == 1
        }
    if len(scale_rows) != 184 or len(manifest) != 184:
        raise ValueError("expected exact 184-image canonical validation tumor cohort")

    rows: list[dict[str, Any]] = []
    for raw in scale_rows:
        image_id = str(raw["image_id"])
        if image_id not in manifest:
            raise ValueError(f"scale row not in canonical validation cohort: {image_id}")
        meta = manifest[image_id]
        row: dict[str, Any] = {
            "image_id": image_id,
            "group_id": str(raw["group_id"]),
            "fold": fold_for_group(str(raw["group_id"]), args.folds),
            "size_group": str(raw["true_size_group"]),
            "baseline_dice": float(raw["baseline_dice"]),
            "tumor_type": int(meta["tumor_type"]),
            "tumor_type_name": str(meta["tumor_type_name"]),
            "benign": int(meta["benign"]),
            "malignant": int(meta["malignant"]),
        }
        for beta in BETAS:
            row[beta_key(beta)] = float(raw[beta_key(beta)])
        rows.append(row)

    if Counter(row["size_group"] for row in rows) != Counter({"small": 94, "medium": 72, "large": 18}):
        raise ValueError("canonical subgroup counts changed")

    label_results: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for label_key in ("tumor_type", "benign", "malignant"):
        result, predictions = evaluate_label(
            rows,
            label_key=label_key,
            folds=args.folds,
            min_train_support=args.min_train_support,
        )
        label_results[label_key] = result
        all_predictions.extend(predictions)

    type_table: list[dict[str, Any]] = []
    for tumor_type in sorted({int(row["tumor_type"]) for row in rows}):
        subset = [row for row in rows if int(row["tumor_type"]) == tumor_type]
        beta_means = {beta: mean(subset, beta_key(beta)) for beta in BETAS}
        best_beta = max(BETAS, key=lambda beta: (beta_means[beta], -abs(beta), -beta))
        type_table.append(
            {
                "tumor_type": tumor_type,
                "tumor_type_name": subset[0]["tumor_type_name"],
                "n": len(subset),
                "size_counts": dict(sorted(Counter(row["size_group"] for row in subset).items())),
                "baseline_dice": mean(subset, "baseline_dice"),
                "same_cohort_best_beta": best_beta,
                "same_cohort_best_dice": beta_means[best_beta],
                "same_cohort_delta": beta_means[best_beta] - mean(subset, "baseline_dice"),
            }
        )

    result = {
        "stage": "rich_gallery_label_conditioned_scale_gate_diagnostic_v1",
        "scale_per_image_sha256": sha256_file(args.scale_per_image),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "folds": args.folds,
        "fold_assignment": "sha256(group_id) modulo folds",
        "min_outer_train_support": args.min_train_support,
        "cohort": {"tumor": 184, "small": 94, "medium": 72, "large": 18},
        "label_results": label_results,
        "same_cohort_type_feasibility": type_table,
        "academic_status": {
            "candidate_choices_frozen_before_this_analysis": True,
            "validation_gt_used_retrospectively_for_expert_utility": True,
            "same_cohort_type_table_is_not_deployable": True,
            "group_separated_result_is_diagnostic_not_confirmatory": True,
            "test_images_read": 0,
            "test_evaluated": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = args.output_dir / "nested_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    audit = {
        "audit_pass": True,
        "summary_sha256": sha256_file(summary_path),
        "predictions_sha256": sha256_file(predictions_path),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
