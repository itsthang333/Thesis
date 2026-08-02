from __future__ import annotations

"""Fail-closed deep analysis of matched-normal Stage-B evidence.

This script consumes only the already evaluated Stage-B tables.  It does not
open images, annotations or test data, and it cannot alter any frozen choice.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank


BASELINE = "g1_upstream_baseline"
PRIMARY = "baseline_transplant_three_to_one"
RANDOM_CONTROL = "baseline_random_control_three_to_one"
EXPECTED_BASELINE_DICE = 0.28872948670665205
GROUPS = ("overall", "small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _summary(values) -> dict[str, float]:
    array = _finite(values)
    if not len(array):
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    return {
        "n": float(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _transition_summary(
    baseline_rows: dict[str, dict[str, str]],
    changed_rows: dict[str, dict[str, str]],
    *,
    group: str,
) -> dict[str, object]:
    image_ids = [
        image_id
        for image_id, row in baseline_rows.items()
        if group == "overall" or row["size_group"] == group
    ]
    records = [(baseline_rows[image_id], changed_rows[image_id]) for image_id in image_ids]
    candidate_changed = [
        pair for pair in records if pair[0]["selected_candidate_index"] != pair[1]["selected_candidate_index"]
    ]
    deltas = np.asarray(
        [float(right["dice"]) - float(left["dice"]) for left, right in records],
        dtype=np.float64,
    )
    area_multipliers = np.asarray(
        [
            float(right["selected_area_ratio"])
            / max(1.0e-12, float(left["selected_area_ratio"]))
            for left, right in candidate_changed
        ],
        dtype=np.float64,
    )
    return {
        "n": len(records),
        "candidate_changed": len(candidate_changed),
        "source_changed": int(sum(left["selected_source"] != right["selected_source"] for left, right in records)),
        "improved": int((deltas > 1.0e-12).sum()),
        "worsened": int((deltas < -1.0e-12).sum()),
        "unchanged": int((np.abs(deltas) <= 1.0e-12).sum()),
        "mean_dice_delta": float(deltas.mean()),
        "median_dice_delta": float(np.median(deltas)),
        "baseline_misses_recovered": int(
            sum(int(left["complete_miss"]) and not int(right["complete_miss"]) for left, right in records)
        ),
        "new_misses_created": int(
            sum(not int(left["complete_miss"]) and int(right["complete_miss"]) for left, right in records)
        ),
        "changed_candidate_area_multiplier_median": (
            float(np.median(area_multipliers)) if len(area_multipliers) else 1.0
        ),
    }


def _score_identity_analysis(candidate_rows: list[dict[str, str]]) -> dict[str, object]:
    by_image: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_image[row["image_id"]].append(row)
    if len(by_image) != 184:
        raise ValueError("candidate table does not contain 184 tumor images")

    score_rows: list[dict[str, object]] = []
    for image_id, rows in by_image.items():
        quality = np.asarray([float(row["candidate_dice"]) for row in rows])
        area = np.asarray([float(row["candidate_area_ratio"]) for row in rows])
        g1 = np.asarray([float(row["g1_logit"]) for row in rows])
        upstream = np.asarray([float(row["upstream_score"]) for row in rows])
        matched = np.asarray([float(row["matched_logit_delta"]) for row in rows])
        random = np.asarray([float(row["random_logit_delta"]) for row in rows])
        class_inside = np.asarray([float(row["matched_class_inside_delta"]) for row in rows])
        norm5 = np.asarray(
            [float(row["norm5_matched_relative_feature_l2_contrast"]) for row in rows]
        )
        baseline = 0.5 * (average_percentile_rank(g1) + average_percentile_rank(upstream))
        primary = 0.75 * average_percentile_rank(baseline) + 0.25 * average_percentile_rank(matched)
        signals = {
            "g1": g1,
            "upstream": upstream,
            "baseline_fixed_fusion": baseline,
            "matched_transplant": matched,
            "random_transplant": random,
            "matched_minus_random": matched - random,
            "matched_class_inside": class_inside,
            "norm5_relative_l2_contrast": norm5,
            "primary_fixed_fusion": primary,
        }
        oracle = int(quality.argmax())
        record: dict[str, object] = {
            "image_id": image_id,
            "size_group": rows[0]["size_group"],
        }
        for name, signal in signals.items():
            record[f"{name}_quality_corr"] = _rank_correlation(signal, quality)
            record[f"{name}_area_corr"] = _rank_correlation(signal, area)
            record[f"{name}_oracle_percentile"] = float(
                average_percentile_rank(signal)[oracle]
            )
        score_rows.append(record)

    result: dict[str, object] = {}
    for group in GROUPS:
        chosen = [
            row for row in score_rows if group == "overall" or row["size_group"] == group
        ]
        group_result: dict[str, object] = {"n": len(chosen), "signals": {}}
        for signal in (
            "g1",
            "upstream",
            "baseline_fixed_fusion",
            "matched_transplant",
            "random_transplant",
            "matched_minus_random",
            "matched_class_inside",
            "norm5_relative_l2_contrast",
            "primary_fixed_fusion",
        ):
            group_result["signals"][signal] = {
                "quality_rank_correlation": _summary(
                    row[f"{signal}_quality_corr"] for row in chosen
                ),
                "area_rank_correlation": _summary(
                    row[f"{signal}_area_corr"] for row in chosen
                ),
                "eligible_oracle_percentile": _summary(
                    row[f"{signal}_oracle_percentile"] for row in chosen
                ),
            }
        result[group] = group_result
    return result


def _fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _markdown(result: dict[str, object]) -> str:
    variants = result["variants"]
    transition = result["selection_transitions"]["primary_vs_baseline"]
    score = result["score_identity"]["overall"]["signals"]
    layers = result["layerwise_overall"]
    lines = [
        "# Matched-normal transplant failure dossier",
        "",
        "## Actual endpoint",
        "",
        "| Variant | Dice | IoU | <1% | 1–<5% | >=5% | Misses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (BASELINE, "transplant_only", "baseline_transplant_equal", PRIMARY, RANDOM_CONTROL):
        item = variants[name]
        lines.append(
            f"| `{name}` | {_fmt(item['overall']['dice'])} | {_fmt(item['overall']['iou'])} | "
            f"{_fmt(item['small']['dice'])} | {_fmt(item['medium']['dice'])} | "
            f"{_fmt(item['large']['dice'])} | {item['overall']['complete_misses']} |"
        )
    lines.extend(
        [
            "",
            "The fixed 3:1 matched fusion does not promote: it is below the immutable baseline, "
            "and its small-lesion Dice is materially worse.  Test remains locked.",
            "",
            "## Exact baseline bottleneck",
            "",
            f"- Proposal-supply regret: `{_fmt(result['baseline_failure']['overall']['proposal_supply_regret'])}`.",
            f"- Eligible selector regret: `{_fmt(result['baseline_failure']['overall']['selector_regret_within_eligible_gallery'])}`.",
            f"- Complete misses: `{result['baseline_failure']['overall']['complete_misses']}/184`; wrong oracle source: "
            f"`{result['baseline_failure']['overall']['wrong_source']}/184`.",
            f"- Small lesions: median selected/GT area ratio "
            f"`{_fmt(result['baseline_failure']['subgroups']['small']['median_selected_gt_area_ratio'], 3)}` "
            "(over-extent dominates).",
            f"- Large lesions: median selected/GT area ratio "
            f"`{_fmt(result['baseline_failure']['subgroups']['large']['median_selected_gt_area_ratio'], 3)}` "
            "(under-extent dominates).",
            "",
            "## What the transplant changed",
            "",
            "| Group | Changed candidate | Improved | Worsened | Recovered misses | New misses | Mean Dice delta | Area multiplier on changed choices |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        item = transition[group]
        lines.append(
            f"| {group} | {item['candidate_changed']} | {item['improved']} | {item['worsened']} | "
            f"{item['baseline_misses_recovered']} | {item['new_misses_created']} | "
            f"{_fmt(item['mean_dice_delta'])} | {_fmt(item['changed_candidate_area_multiplier_median'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate identity versus area",
            "",
            "| Signal | Median quality rank corr | Median area rank corr | Median oracle percentile |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "g1",
        "upstream",
        "baseline_fixed_fusion",
        "matched_transplant",
        "random_transplant",
        "matched_minus_random",
        "matched_class_inside",
        "norm5_relative_l2_contrast",
        "primary_fixed_fusion",
    ):
        item = score[name]
        lines.append(
            f"| `{name}` | {_fmt(item['quality_rank_correlation']['median'])} | "
            f"{_fmt(item['area_rank_correlation']['median'])} | "
            f"{_fmt(item['eligible_oracle_percentile']['median'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer trajectory",
            "",
            "| Layer | Matched oracle percentile | Random oracle percentile | Matched-random gain | Quality corr | Area corr | Recipient CV |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in ("pool0", "transition1", "transition2", "transition3", "norm5"):
        item = layers[stage]
        lines.append(
            f"| `{stage}` | {_fmt(item['matched_oracle_percentile'])} | "
            f"{_fmt(item['random_oracle_percentile'])} | {_fmt(item['matched_random_gain'])} | "
            f"{_fmt(item['matched_quality_corr'])} | {_fmt(item['matched_area_corr'])} | "
            f"{_fmt(item['oracle_recipient_cv'])} |"
        )
    lines.extend(
        [
            "",
            "## Mechanistic conclusion",
            "",
            "1. The gallery is not the limiting resource: supply regret is negligible; selection accounts for essentially the full remaining gap.",
            "2. Transplanted content is not a reliable candidate-identity observable at the stem.  Matched and random donor trajectories are nearly indistinguishable.",
            "3. The weak mid-level signal peaks around transition2, then the final representation becomes area-dominated rather than overlap-dominated.  This explains improved medium/large extent and the severe small-lesion collapse.",
            "4. The next improvement must preserve G1+upstream and add a candidate-specific tumor-identity observable that is independent of area, plus a signed scale/extent cue.  More proposals, global alpha tuning, transplant weighting or post-hoc morphology are not supported.",
            "",
            "No test data were accessed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("failure-analysis output must not exist")
    summary_path = args.stage_b_root / "evaluation_summary.json"
    if sha256_file(summary_path) != args.expected_summary_sha256:
        raise ValueError("Stage-B summary SHA-256 mismatch")
    stage_b = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        stage_b.get("stage") != "rich_gallery_matched_normal_transplant_stage_b_v2"
        or stage_b.get("validation_images") != 371
        or stage_b.get("tumor_images_evaluated") != 184
        or stage_b.get("test_evaluated") is not False
        or stage_b.get("candidate_scores_frozen_before_validation_gt") is not True
    ):
        raise ValueError("Stage-B scientific boundary mismatch")
    file_contract = {
        "per_image_results.csv": stage_b["per_image_results_sha256"],
        "per_candidate_layerwise.csv": stage_b["per_candidate_layerwise_sha256"],
        "per_image_layerwise_summary.csv": stage_b["per_image_layerwise_summary_sha256"],
    }
    for name, expected in file_contract.items():
        if sha256_file(args.stage_b_root / name) != expected:
            raise ValueError(f"Stage-B table SHA-256 mismatch: {name}")
    per_image = _read_csv(args.stage_b_root / "per_image_results.csv")
    per_candidate = _read_csv(args.stage_b_root / "per_candidate_layerwise.csv")
    indexed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in per_image:
        indexed[row["variant"]][row["image_id"]] = row
    if any(len(indexed[name]) != 184 for name in (BASELINE, PRIMARY, RANDOM_CONTROL)):
        raise ValueError("Stage-B selector cohort incomplete")
    transitions = {
        "primary_vs_baseline": {
            group: _transition_summary(indexed[BASELINE], indexed[PRIMARY], group=group)
            for group in GROUPS
        },
        "primary_vs_random_control": {
            group: _transition_summary(indexed[RANDOM_CONTROL], indexed[PRIMARY], group=group)
            for group in GROUPS
        },
    }
    layerwise = stage_b["layerwise_bottleneck"]["overall"]["stages"]
    layer_table = {}
    for stage, item in layerwise.items():
        layer_table[stage] = {
            "matched_oracle_percentile": item["matched"]["oracle_percentile"]["median"],
            "random_oracle_percentile": item["random"]["oracle_percentile"]["median"],
            "matched_random_gain": item["matched_minus_random"]["oracle_percentile"]["median"],
            "matched_quality_corr": item["matched"]["quality_rank_correlation"]["median"],
            "matched_area_corr": item["matched"]["area_rank_correlation"]["median"],
            "oracle_recipient_cv": item["matched"]["oracle_recipient_cv"]["median"],
        }
    result = {
        "stage": "rich_gallery_matched_normal_transplant_failure_analysis_v1",
        "stage_b_summary_sha256": args.expected_summary_sha256,
        "variants": stage_b["variants"],
        "promotion": stage_b["promotion"],
        "bootstrap": stage_b["bootstrap"],
        "baseline_failure": stage_b["baseline_failure_decomposition"],
        "selection_transitions": transitions,
        "score_identity": _score_identity_analysis(per_candidate),
        "layerwise_overall": layer_table,
        "layerwise_strata": stage_b["layerwise_bottleneck"]["strata"],
        "identified_first_failure_branch": stage_b["layerwise_bottleneck"][
            "identified_first_failure_branch"
        ],
        "decision": "retire_matched_normal_transplant_selector_without_sweep",
        "next_required_observables": [
            "candidate_specific_tumor_identity_independent_of_area",
            "scale_dependent_signed_extent",
        ],
        "validation_gt_used_only_via_audited_stage_b_tables": True,
        "test_evaluated": False,
    }
    if abs(float(result["variants"][BASELINE]["overall"]["dice"]) - EXPECTED_BASELINE_DICE) > 1.0e-12:
        raise ValueError("immutable baseline does not reproduce")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "failure_analysis.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output_dir / "MATCHED_NORMAL_TRANSPLANT_FAILURE_DOSSIER.md"
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": True,
                "failure_analysis_sha256": sha256_file(json_path),
                "dossier_sha256": sha256_file(markdown_path),
                "decision": result["decision"],
                "test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
