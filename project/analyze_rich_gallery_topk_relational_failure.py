from __future__ import annotations

"""Paired post-freeze decomposition of the top-k relational diagnostic."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank


BASELINE = "g1_upstream_baseline"
RELATIONAL = "top10_cross_source_relational_product"
SUBGROUPS = ("overall", "small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def correlation(left: list[float], right: list[float]) -> float:
    a = average_percentile_rank(np.asarray(left, dtype=np.float64))
    b = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def group_bootstrap_ci(rows: list[dict[str, object]], *, seed: int = 20260801) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(float(row["delta_dice"]))
    identifiers = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(10000, dtype=np.float64)
    for index in range(len(samples)):
        drawn = rng.choice(identifiers, size=len(identifiers), replace=True)
        values = [value for group in drawn for value in grouped[str(group)]]
        samples[index] = float(np.mean(values))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def selected(rows: list[dict[str, object]], subgroup: str) -> list[dict[str, object]]:
    return [row for row in rows if subgroup == "overall" or row["size_group"] == subgroup]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = np.asarray([float(row["delta_dice"]) for row in rows])
    changed = [row for row in rows if row["choice_changed"]]
    wins = [row for row in rows if float(row["delta_dice"]) > 1.0e-12]
    losses = [row for row in rows if float(row["delta_dice"]) < -1.0e-12]
    ties = len(rows) - len(wins) - len(losses)
    transition = Counter(row["miss_transition"] for row in rows)
    sources: dict[str, dict[str, float | int]] = {}
    for name in sorted({str(row["source_transition"]) for row in rows}):
        subset = [row for row in rows if row["source_transition"] == name]
        sources[name] = {
            "n": len(subset),
            "delta_sum": float(sum(float(row["delta_dice"]) for row in subset)),
            "delta_mean": float(np.mean([float(row["delta_dice"]) for row in subset])),
        }
    return {
        "n": len(rows),
        "mean_delta_dice": float(deltas.mean()),
        "median_delta_dice": float(np.median(deltas)),
        "group_bootstrap_ci95": group_bootstrap_ci(rows),
        "wins": len(wins),
        "losses": len(losses),
        "ties": ties,
        "positive_delta_mass": float(sum(float(row["delta_dice"]) for row in wins)),
        "negative_delta_mass": float(sum(float(row["delta_dice"]) for row in losses)),
        "choice_changed": len(changed),
        "choice_changed_fraction": float(len(changed) / len(rows)),
        "changed_mean_delta": float(np.mean([float(row["delta_dice"]) for row in changed])) if changed else 0.0,
        "baseline_misses": int(sum(int(row["baseline_miss"]) for row in rows)),
        "relational_misses": int(sum(int(row["relational_miss"]) for row in rows)),
        "miss_transitions": dict(sorted(transition.items())),
        "baseline_area_ratio_median": float(np.median([float(row["baseline_area_ratio"]) for row in rows])),
        "relational_area_ratio_median": float(np.median([float(row["relational_area_ratio"]) for row in rows])),
        "area_ratio_increased_fraction": float(np.mean([
            float(row["relational_area_ratio"]) > float(row["baseline_area_ratio"])
            for row in rows
        ])),
        "relational_support_median": float(np.median([float(row["relational_support"]) for row in rows])),
        "support_vs_relational_dice_spearman": correlation(
            [float(row["relational_support"]) for row in rows],
            [float(row["relational_dice"]) for row in rows],
        ),
        "support_vs_delta_spearman": correlation(
            [float(row["relational_support"]) for row in rows],
            [float(row["delta_dice"]) for row in rows],
        ),
        "source_transitions": dict(
            sorted(sources.items(), key=lambda item: (-item[1]["n"], item[0]))
        ),
    }


def main() -> None:
    args = parse_args()
    with args.selection_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        selection_rows = list(csv.DictReader(handle))
    with args.per_image.open("r", newline="", encoding="utf-8-sig") as handle:
        metric_rows = list(csv.DictReader(handle))
    selections = {(row["variant"], row["image_id"]): row for row in selection_rows}
    metrics = {(row["variant"], row["image_id"]): row for row in metric_rows}
    images = sorted({row["image_id"] for row in metric_rows})
    if len(images) != 184 or len(selections) != 742 or len(metrics) != 368:
        raise ValueError("paired relational evidence has an invalid cohort")

    paired: list[dict[str, object]] = []
    for image_id in images:
        base_selection = selections[(BASELINE, image_id)]
        relation_selection = selections[(RELATIONAL, image_id)]
        base = metrics[(BASELINE, image_id)]
        relation = metrics[(RELATIONAL, image_id)]
        if base["group_id"] != relation["group_id"] or base["size_group"] != relation["size_group"]:
            raise ValueError(f"paired identity mismatch: {image_id}")
        base_miss = int(base["complete_miss"])
        relation_miss = int(relation["complete_miss"])
        miss_transition = {
            (0, 0): "hit_to_hit",
            (0, 1): "hit_to_miss",
            (1, 0): "miss_to_hit",
            (1, 1): "miss_to_miss",
        }[(base_miss, relation_miss)]
        paired.append({
            "image_id": image_id,
            "group_id": base["group_id"],
            "size_group": base["size_group"],
            "baseline_dice": float(base["dice"]),
            "relational_dice": float(relation["dice"]),
            "delta_dice": float(relation["dice"]) - float(base["dice"]),
            "baseline_miss": base_miss,
            "relational_miss": relation_miss,
            "miss_transition": miss_transition,
            "baseline_area_ratio": float(base["selected_gt_area_ratio"]),
            "relational_area_ratio": float(relation["selected_gt_area_ratio"]),
            "baseline_source": base["selected_source"],
            "relational_source": relation["selected_source"],
            "source_transition": f"{base['selected_source']}->{relation['selected_source']}",
            "baseline_candidate_index": int(base_selection["selected_candidate_index"]),
            "relational_candidate_index": int(relation_selection["selected_candidate_index"]),
            "choice_changed": int(base_selection["selected_candidate_index"] != relation_selection["selected_candidate_index"]),
            "relational_support": float(relation_selection["selected_relational_support"]),
            "oracle_dice": float(base["oracle_dice"]),
        })

    report = {
        "stage": "rich_gallery_top10_relational_failure_decomposition_v1",
        "decision": "failed_overall_and_small_noninferiority; no sweep",
        "subgroups": {group: summarize(selected(paired, group)) for group in SUBGROUPS},
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "evaluation_per_image_sha256": sha256_file(args.per_image),
        "analysis_per_image_sha256": sha256_file(per_image_path),
        "analysis_summary_sha256": sha256_file(summary_path),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
