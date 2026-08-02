from __future__ import annotations

"""Retrospective feasibility analysis for a scale-conditional gallery policy.

This script consumes an already frozen validation candidate table.  Candidate
Dice and lesion-size groups are used only to estimate upper bounds and failure
modes; outputs are not a deployable selector and cannot be used as test claims.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from models.rich_gallery_g2_objective import (
    average_percentile_rank,
    rank_fusion_scores,
    stable_select,
)


BETAS = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
GROUPS = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--expected-per-candidate-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inferred_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    args = parse_args()
    if sha256_file(args.per_candidate) != args.expected_per_candidate_sha256:
        raise ValueError("per-candidate table hash mismatch")

    bags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.per_candidate.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            bags[str(raw["image_id"])].append(
                {
                    "image_id": str(raw["image_id"]),
                    "group_id": str(raw["group_id"]),
                    "size_group": str(raw["size_group"]),
                    "candidate_local_index": int(raw["candidate_local_index"]),
                    "source": str(raw["source"]),
                    "area": float(raw["candidate_area_ratio"]),
                    "dice": float(raw["candidate_dice"]),
                    "g1": float(raw["g1_logit"]),
                    "upstream": float(raw["upstream_score"]),
                    "baseline_selected": int(raw["is_baseline_selected"]),
                }
            )
    if len(bags) != 184:
        raise ValueError(f"expected 184 tumor bags, got {len(bags)}")

    per_image: list[dict[str, Any]] = []
    selected_by_beta: dict[float, list[dict[str, Any]]] = {beta: [] for beta in BETAS}
    for image_id, rows in sorted(bags.items()):
        groups = {str(row["size_group"]) for row in rows}
        group_ids = {str(row["group_id"]) for row in rows}
        if len(groups) != 1 or len(group_ids) != 1 or next(iter(groups)) not in GROUPS:
            raise ValueError(f"inconsistent bag metadata: {image_id}")
        g1 = np.asarray([row["g1"] for row in rows], dtype=np.float64)
        upstream = np.asarray([row["upstream"] for row in rows], dtype=np.float64)
        areas = np.asarray([row["area"] for row in rows], dtype=np.float64)
        dice_values = np.asarray([row["dice"] for row in rows], dtype=np.float64)
        baseline_scores = rank_fusion_scores(g1, upstream)
        area_rank = average_percentile_rank(np.log(np.clip(areas, 1.0e-8, None)))
        baseline_local = stable_select(baseline_scores, g1)
        frozen_selected = [index for index, row in enumerate(rows) if row["baseline_selected"] == 1]
        if frozen_selected != [baseline_local]:
            raise ValueError(f"baseline selection did not reproduce: {image_id}")

        order = np.argsort(-baseline_scores, kind="stable")
        top_five = order[: min(5, len(order))]
        source_best_areas: list[float] = []
        for source in sorted({str(row["source"]) for row in rows}):
            members = [index for index, row in enumerate(rows) if row["source"] == source]
            local = max(members, key=lambda index: (baseline_scores[index], g1[index], -index))
            source_best_areas.append(float(areas[local]))
        image_row: dict[str, Any] = {
            "image_id": image_id,
            "group_id": next(iter(group_ids)),
            "true_size_group": next(iter(groups)),
            "candidate_count": len(rows),
            "baseline_dice": float(dice_values[baseline_local]),
            "baseline_selected_area": float(areas[baseline_local]),
            "top5_median_area": float(np.median(areas[top_five])),
            "source_best_median_area": float(np.median(source_best_areas)),
            "eligible_oracle_dice": float(dice_values.max()),
        }
        for beta in BETAS:
            scores = baseline_scores + float(beta) * (area_rank - 0.5)
            selected = stable_select(scores, g1)
            record = {
                "image_id": image_id,
                "group_id": image_row["group_id"],
                "size_group": image_row["true_size_group"],
                "beta": beta,
                "dice": float(dice_values[selected]),
                "selected_area": float(areas[selected]),
                "candidate_local_index": int(rows[selected]["candidate_local_index"]),
            }
            selected_by_beta[beta].append(record)
            image_row[f"beta_{beta:g}_dice"] = record["dice"]
            image_row[f"beta_{beta:g}_area"] = record["selected_area"]
        per_image.append(image_row)

    counts = Counter(str(row["true_size_group"]) for row in per_image)
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise ValueError(f"subgroup counts changed: {counts}")

    beta_metrics: dict[str, dict[str, float]] = {}
    for beta, rows in selected_by_beta.items():
        beta_metrics[f"{beta:g}"] = {
            group: mean(
                [row for row in rows if group == "overall" or row["size_group"] == group],
                "dice",
            )
            for group in ("overall", *GROUPS)
        }
    best_beta_by_group = {
        group: max(BETAS, key=lambda beta: beta_metrics[f"{beta:g}"][group])
        for group in GROUPS
    }
    true_group_routed = [
        float(row[f"beta_{best_beta_by_group[str(row['true_size_group'])]:g}_dice"])
        for row in per_image
    ]
    expert_betas = sorted(set(best_beta_by_group.values()))
    expert_oracle = [
        max(float(row[f"beta_{beta:g}_dice"]) for beta in expert_betas)
        for row in per_image
    ]

    gate_results: dict[str, Any] = {}
    for feature in ("baseline_selected_area", "top5_median_area", "source_best_median_area"):
        routed: list[float] = []
        confusion: Counter[tuple[str, str]] = Counter()
        for row in per_image:
            predicted = inferred_group(float(row[feature]))
            truth = str(row["true_size_group"])
            confusion[(truth, predicted)] += 1
            beta = best_beta_by_group[predicted]
            routed.append(float(row[f"beta_{beta:g}_dice"]))
        gate_results[feature] = {
            "overall_dice": float(np.mean(routed)),
            "subgroup_dice": {
                group: float(
                    np.mean(
                        [value for value, row in zip(routed, per_image, strict=True) if row["true_size_group"] == group]
                    )
                )
                for group in GROUPS
            },
            "group_accuracy": float(
                np.mean([inferred_group(float(row[feature])) == row["true_size_group"] for row in per_image])
            ),
            "confusion": {
                f"{truth}->{prediction}": int(value)
                for (truth, prediction), value in sorted(confusion.items())
            },
        }

    result = {
        "stage": "rich_gallery_scale_conditional_retrospective_feasibility_v1",
        "input_sha256": args.expected_per_candidate_sha256,
        "cohort": {"tumor": 184, "small": 94, "medium": 72, "large": 18},
        "baseline": beta_metrics["0"],
        "eligible_oracle": {
            group: float(
                np.mean(
                    [row["eligible_oracle_dice"] for row in per_image if group == "overall" or row["true_size_group"] == group]
                )
            )
            for group in ("overall", *GROUPS)
        },
        "beta_metrics": beta_metrics,
        "best_beta_by_true_group": best_beta_by_group,
        "true_group_routing_dice": float(np.mean(true_group_routed)),
        "three_expert_per_image_oracle_dice": float(np.mean(expert_oracle)),
        "candidate_derived_gate_results": gate_results,
        "scale_identifiability": {
            feature: {
                "spearman_with_true_size_ordinal": float(
                    spearmanr(
                        np.log(np.clip([float(row[feature]) for row in per_image], 1.0e-8, None)),
                        [GROUPS.index(str(row["true_size_group"])) for row in per_image],
                    ).statistic
                ),
                "median_by_true_group": {
                    group: float(np.median([row[feature] for row in per_image if row["true_size_group"] == group]))
                    for group in GROUPS
                },
            }
            for feature in ("baseline_selected_area", "top5_median_area", "source_best_median_area")
        },
        "academic_status": {
            "candidate_scores_were_frozen_before_gt": True,
            "gt_used_retrospectively_for_beta_and_oracle_analysis": True,
            "true_group_routing_is_not_deployable": True,
            "candidate_derived_gates_are_exploratory_validation_results": True,
            "test_images_read": 0,
            "test_evaluated": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image_scale_features.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "input_sha256": args.expected_per_candidate_sha256,
        "summary_sha256": sha256_file(summary_path),
        "per_image_sha256": sha256_file(per_image_path),
        "tumor_images": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
