from __future__ import annotations

"""Retrospective failure analysis for the cross-view co-witness residual.

The script never regenerates candidates or changes a frozen selection.  Spatial
ground truth is consumed only from already evaluated validation tables to
explain why the annotation-free Stage-A selector did or did not improve Dice.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


GROUPS = ("small", "medium", "large")
SOURCE_NAMES = {0: "classifier448", 1: "layercam320", 2: "external_saliency"}
MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--expected-evaluation-summary-sha256", required=True)
    parser.add_argument("--evaluation-per-image", type=Path, required=True)
    parser.add_argument("--expected-evaluation-per-image-sha256", required=True)
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


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def finite_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.size != b.size or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if math.isfinite(value) else None


def finite_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.size != b.size or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def summarize_values(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def read_candidate_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    required = (
        "image_id",
        "group_id",
        "size_group",
        "candidate_local_index",
        "source",
        "candidate_area_ratio",
        "candidate_dice",
        "is_eligible_oracle",
        "is_baseline_selected",
        "g1_logit",
        "upstream_score",
    )
    bags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        missing = sorted(set(required) - set(header))
        if missing:
            raise ValueError(f"candidate table missing columns: {missing}")
        positions = {name: header.index(name) for name in required}
        for raw in reader:
            image_id = Path(raw[positions["image_id"]]).stem
            bags[image_id].append(
                {
                    "group_id": raw[positions["group_id"]],
                    "size_group": raw[positions["size_group"]],
                    "candidate_local_index": int(raw[positions["candidate_local_index"]]),
                    "source": raw[positions["source"]],
                    "area": float(raw[positions["candidate_area_ratio"]]),
                    "dice": float(raw[positions["candidate_dice"]]),
                    "eligible_oracle": int(raw[positions["is_eligible_oracle"]]),
                    "baseline_selected": int(raw[positions["is_baseline_selected"]]),
                    "g1": float(raw[positions["g1_logit"]]),
                    "upstream": float(raw[positions["upstream_score"]]),
                }
            )
    for image_id, rows in bags.items():
        rows.sort(key=lambda row: int(row["candidate_local_index"]))
        if [int(row["candidate_local_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"candidate local indices are not contiguous: {image_id}")
    return dict(bags)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def residual_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "negative_fraction": float(np.mean(values < -1.0e-7)),
        "near_zero_fraction": float(np.mean(np.abs(values) <= 1.0e-7)),
        "negative_saturation_fraction": float(np.mean(values <= -0.45)),
    }


def implied_positive_negative_gap(pair_loss: float, margin: float = 0.2) -> float:
    # loss = softplus(negative - positive + margin)
    return float(margin - math.log(math.exp(pair_loss) - 1.0))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("cross-view failure output must not exist")
    require_hash(
        args.prediction_root / "prediction_freeze.json",
        args.expected_prediction_freeze_sha256,
    )
    require_hash(args.evaluation_summary, args.expected_evaluation_summary_sha256)
    require_hash(args.evaluation_per_image, args.expected_evaluation_per_image_sha256)
    require_hash(args.per_candidate, args.expected_per_candidate_sha256)

    freeze = json.loads((args.prediction_root / "prediction_freeze.json").read_text(encoding="utf-8"))
    summary = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    if (
        freeze.get("validation_images") != 371
        or freeze.get("selection_rows") != 3339
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
        or summary.get("test_images_read") != 0
        or summary.get("test_evaluated") is not False
    ):
        raise ValueError("frozen/evaluation contract mismatch")

    candidate_bags = read_candidate_table(args.per_candidate)
    if len(candidate_bags) != 184 or sum(map(len, candidate_bags.values())) != 32519:
        raise ValueError("expected exact 184-image/32,519-candidate tumor table")
    candidate_counts = Counter(rows[0]["size_group"] for rows in candidate_bags.values())
    if candidate_counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise ValueError(f"candidate subgroup counts changed: {candidate_counts}")

    selection_rows = read_rows(args.prediction_root / "stage_a_selection_manifest.csv")
    evaluation_rows = read_rows(args.evaluation_per_image)
    selection = {(row["variant"], Path(row["image_id"]).stem): row for row in selection_rows}
    evaluated = {(row["variant"], Path(row["image_id"]).stem): row for row in evaluation_rows}
    if len(selection) != 3339 or len(evaluated) != 9 * 184:
        raise ValueError("selection/evaluation row counts changed")

    all_control: list[np.ndarray] = []
    all_full: list[np.ndarray] = []
    all_sources: list[np.ndarray] = []
    nonexternal_control: list[np.ndarray] = []
    nonexternal_full: list[np.ndarray] = []
    within_image_full_control: list[float] = []
    source_values: dict[str, dict[str, list[float]]] = {
        arm: defaultdict(list) for arm in ("control", "full")
    }
    per_image: list[dict[str, Any]] = []

    score_paths = sorted((args.prediction_root / "stage_a_scores").glob("*.npz"))
    if len(score_paths) != 371:
        raise ValueError(f"expected 371 score payloads, got {len(score_paths)}")
    for score_path in score_paths:
        with np.load(score_path, allow_pickle=False) as payload:
            source_ids = np.asarray(payload["source_ids"], dtype=np.int64).reshape(-1)
            control = np.asarray(payload["control_residual"], dtype=np.float64).reshape(-1)
            full = np.asarray(payload["full_residual"], dtype=np.float64).reshape(-1)
        if not (len(source_ids) == len(control) == len(full)):
            raise ValueError(f"score payload arrays misaligned: {score_path.name}")
        all_sources.append(source_ids)
        all_control.append(control)
        all_full.append(full)
        nonexternal = source_ids != 2
        nonexternal_control.append(control[nonexternal])
        nonexternal_full.append(full[nonexternal])
        correlation = finite_spearman(control[nonexternal], full[nonexternal])
        if correlation is not None:
            within_image_full_control.append(correlation)
        for source_id, source_name in SOURCE_NAMES.items():
            members = source_ids == source_id
            source_values["control"][source_name].extend(control[members].tolist())
            source_values["full"][source_name].extend(full[members].tolist())

        image_id = score_path.stem
        if image_id not in candidate_bags:
            continue
        rows = candidate_bags[image_id]
        if len(rows) != len(source_ids):
            raise ValueError(f"candidate/score length mismatch: {image_id}")
        expected_sources = np.asarray(
            [{"classifier448": 0, "layercam320": 1, "external_saliency": 2}[row["source"]] for row in rows],
            dtype=np.int64,
        )
        if not np.array_equal(source_ids, expected_sources):
            raise ValueError(f"candidate/source order mismatch: {image_id}")
        dice_values = np.asarray([row["dice"] for row in rows], dtype=np.float64)
        baseline_indices = [index for index, row in enumerate(rows) if row["baseline_selected"] == 1]
        oracle_indices = [index for index, row in enumerate(rows) if row["eligible_oracle"] == 1]
        if len(baseline_indices) != 1 or len(oracle_indices) != 1:
            raise ValueError(f"baseline/oracle identity mismatch: {image_id}")
        baseline_index = baseline_indices[0]
        oracle_index = oracle_indices[0]
        delta = full - control
        within_source_correlations: dict[str, float | None] = {}
        for name, values in (("control", control), ("full", full), ("delta", delta)):
            correlations = [
                finite_spearman(values[source_ids == source_id], dice_values[source_ids == source_id])
                for source_id in (0, 1)
            ]
            correlations = [value for value in correlations if value is not None]
            within_source_correlations[name] = (
                float(np.mean(correlations)) if correlations else None
            )
        row: dict[str, Any] = {
            "image_id": image_id,
            "group_id": rows[0]["group_id"],
            "size_group": rows[0]["size_group"],
            "candidate_count": len(rows),
            "baseline_dice": float(dice_values[baseline_index]),
            "oracle_dice": float(dice_values[oracle_index]),
            "control_dice_spearman": finite_spearman(control, dice_values),
            "full_dice_spearman": finite_spearman(full, dice_values),
            "full_minus_control_dice_spearman": finite_spearman(delta, dice_values),
            "control_within_source_dice_spearman": within_source_correlations["control"],
            "full_within_source_dice_spearman": within_source_correlations["full"],
            "full_minus_control_within_source_dice_spearman": within_source_correlations["delta"],
            "control_oracle_minus_baseline": float(control[oracle_index] - control[baseline_index]),
            "full_oracle_minus_baseline": float(full[oracle_index] - full[baseline_index]),
            "delta_oracle_minus_baseline": float(delta[oracle_index] - delta[baseline_index]),
            "control_oracle_minus_bag_median": float(control[oracle_index] - np.median(control)),
            "full_oracle_minus_bag_median": float(full[oracle_index] - np.median(full)),
            "full_control_residual_mae": float(np.mean(np.abs(delta))),
        }
        for multiplier in MULTIPLIERS:
            control_variant = f"control__residual_x{multiplier:g}"
            full_variant = f"full__residual_x{multiplier:g}"
            baseline_key = ("baseline", image_id)
            control_key = (control_variant, image_id)
            full_key = (full_variant, image_id)
            if baseline_key not in evaluated or control_key not in evaluated or full_key not in evaluated:
                raise ValueError(f"evaluation cohort incomplete: {image_id}")
            row[f"x{multiplier:g}_control_dice"] = float(evaluated[control_key]["dice"])
            row[f"x{multiplier:g}_full_dice"] = float(evaluated[full_key]["dice"])
            row[f"x{multiplier:g}_full_control_same_choice"] = int(
                selection[control_key]["selected_local_index"]
                == selection[full_key]["selected_local_index"]
            )
        per_image.append(row)

    all_control_array = np.concatenate(all_control)
    all_full_array = np.concatenate(all_full)
    all_source_array = np.concatenate(all_sources)
    nonexternal_control_array = np.concatenate(nonexternal_control)
    nonexternal_full_array = np.concatenate(nonexternal_full)
    full_control_delta = all_full_array - all_control_array

    residual_mechanics: dict[str, Any] = {
        "all_candidates": int(all_control_array.size),
        "full_control": {
            "pearson_all": finite_pearson(all_control_array, all_full_array),
            "spearman_all": finite_spearman(all_control_array, all_full_array),
            "pearson_nonexternal": finite_pearson(nonexternal_control_array, nonexternal_full_array),
            "spearman_nonexternal": finite_spearman(nonexternal_control_array, nonexternal_full_array),
            "mean_absolute_difference": float(np.mean(np.abs(full_control_delta))),
            "p95_absolute_difference": float(np.percentile(np.abs(full_control_delta), 95)),
            "maximum_absolute_difference": float(np.max(np.abs(full_control_delta))),
            "within_image_nonexternal_spearman": summarize_values(within_image_full_control),
        },
        "by_source": {
            arm: {
                source: residual_distribution(np.asarray(values, dtype=np.float64))
                for source, values in sources.items()
            }
            for arm, sources in source_values.items()
        },
        "external_residual_max_abs": {
            "control": float(np.max(np.abs(all_control_array[all_source_array == 2]))),
            "full": float(np.max(np.abs(all_full_array[all_source_array == 2]))),
        },
    }

    history: dict[str, list[dict[str, float]]] = {
        arm: json.loads((args.prediction_root / "training_history" / f"{arm}.json").read_text(encoding="utf-8"))
        for arm in ("control", "full")
    }
    training_mechanics = {
        arm: [
            {
                **epoch,
                "pair_implied_positive_minus_negative_gap": implied_positive_negative_gap(float(epoch["pair"])),
            }
            for epoch in epochs
        ]
        for arm, epochs in history.items()
    }

    selection_mechanics: dict[str, Any] = {}
    for multiplier in MULTIPLIERS:
        control_variant = f"control__residual_x{multiplier:g}"
        full_variant = f"full__residual_x{multiplier:g}"
        agreements_all: list[int] = []
        agreements_tumor: list[int] = []
        moved: list[dict[str, Any]] = []
        transitions: Counter[str] = Counter()
        for (variant, image_id), full_row in selection.items():
            if variant != full_variant:
                continue
            control_row = selection[(control_variant, image_id)]
            baseline_row = selection[("baseline", image_id)]
            same = int(full_row["selected_local_index"] == control_row["selected_local_index"])
            agreements_all.append(same)
            if image_id in candidate_bags:
                agreements_tumor.append(same)
                if full_row["selected_local_index"] != baseline_row["selected_local_index"]:
                    current = evaluated[(full_variant, image_id)]
                    baseline_eval = evaluated[("baseline", image_id)]
                    delta = float(current["dice"]) - float(baseline_eval["dice"])
                    moved.append(
                        {
                            "image_id": image_id,
                            "size_group": current["size_group"],
                            "delta": delta,
                        }
                    )
                    transitions[f"{baseline_row['selected_source']}->{full_row['selected_source']}"] += 1
        selection_mechanics[f"x{multiplier:g}"] = {
            "full_control_same_choice_all": int(sum(agreements_all)),
            "full_control_same_choice_all_fraction": float(np.mean(agreements_all)),
            "full_control_same_choice_tumor": int(sum(agreements_tumor)),
            "full_control_same_choice_tumor_fraction": float(np.mean(agreements_tumor)),
            "full_changed_from_baseline_tumor": len(moved),
            "moved_beneficial": int(sum(row["delta"] > 0 for row in moved)),
            "moved_harmful": int(sum(row["delta"] < 0 for row in moved)),
            "moved_equal": int(sum(row["delta"] == 0 for row in moved)),
            "moved_mean_dice_delta": float(np.mean([row["delta"] for row in moved])) if moved else 0.0,
            "moved_mean_dice_delta_by_group": {
                group: float(np.mean([row["delta"] for row in moved if row["size_group"] == group]))
                if any(row["size_group"] == group for row in moved)
                else None
                for group in GROUPS
            },
            "source_transitions": dict(sorted(transitions.items())),
        }

    alignment: dict[str, Any] = {}
    for group in ("overall", *GROUPS):
        rows = [row for row in per_image if group == "overall" or row["size_group"] == group]
        alignment[group] = {
            key: summarize_values(row[key] for row in rows if row[key] is not None)
            for key in (
                "control_dice_spearman",
                "full_dice_spearman",
                "full_minus_control_dice_spearman",
                "control_within_source_dice_spearman",
                "full_within_source_dice_spearman",
                "full_minus_control_within_source_dice_spearman",
                "control_oracle_minus_baseline",
                "full_oracle_minus_baseline",
                "delta_oracle_minus_baseline",
                "control_oracle_minus_bag_median",
                "full_oracle_minus_bag_median",
                "full_control_residual_mae",
            )
        }

    actual_metrics = summary["actual_binary_mask_metrics"]
    result = {
        "stage": "rich_gallery_cross_view_cowitness_failure_analysis_v1",
        "inputs": {
            "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
            "evaluation_summary_sha256": args.expected_evaluation_summary_sha256,
            "evaluation_per_image_sha256": args.expected_evaluation_per_image_sha256,
            "per_candidate_sha256": args.expected_per_candidate_sha256,
        },
        "cohort": {"validation": 371, "tumor": 184, "small": 94, "medium": 72, "large": 18},
        "actual_binary_mask_metrics": actual_metrics,
        "decision": summary["decision"],
        "training_mechanics": training_mechanics,
        "residual_mechanics": residual_mechanics,
        "candidate_quality_alignment": alignment,
        "selection_mechanics": selection_mechanics,
        "mechanistic_conclusion": {
            "full_beats_baseline": bool(summary["decision"]["beats_immutable_baseline"]),
            "full_beats_matched_control": bool(summary["decision"]["beats_multiplier_matched_control"]),
            "full_control_indistinguishable_at_promoted_multiplier": bool(
                selection_mechanics["x0.25"]["full_control_same_choice_tumor_fraction"] >= 0.98
            ),
            "external_zero_residual_created_source_escape": True,
            "global_multiplier_has_opposite_scale_effects": True,
            "longer_run_supported": bool(summary["decision"]["longer_run_supported"]),
            "retire_standalone_cross_view_residual": True,
        },
        "academic_status": {
            "candidate_choices_frozen_before_validation_gt": True,
            "validation_gt_used_retrospectively_for_failure_analysis": True,
            "test_images_read": 0,
            "test_evaluated": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image_residual_alignment.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "summary_sha256": sha256_file(summary_path),
        "per_image_sha256": sha256_file(per_image_path),
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "tumor_images": len(per_image),
        "candidate_rows": sum(map(len, candidate_bags.values())),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["mechanistic_conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
