"""Post-hoc, read-only failure analysis for the frozen S7 selector pair.

This utility never creates predictions, trains a model, selects a rescue arm, or
opens BTXRD test.  It consumes only the already-frozen validation evaluation
tables plus GT-blind producer/cache diagnostics and prints one JSON dossier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260802-codex-s7-global-local-instance-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(array.size, dtype=np.float64)
    cursor = 0
    while cursor < array.size:
        stop = cursor + 1
        while stop < array.size and array[order[stop]] == array[order[cursor]]:
            stop += 1
        result[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return result


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 3:
        return None
    left = rankdata(first)
    right = rankdata(second)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def top_margin(logits: np.ndarray) -> float:
    ordered = np.sort(np.asarray(logits, dtype=np.float64).reshape(-1))
    return float(ordered[-1] - ordered[-2]) if ordered.size > 1 else float("inf")


def winner_position(indices: np.ndarray, candidate_index: int) -> int:
    matches = np.flatnonzero(indices == candidate_index)
    if matches.size != 1:
        raise RuntimeError(f"candidate index {candidate_index} is not unique")
    return int(matches[0])


def score_payloads(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(root / "candidate_score_manifest.csv")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = root / row["score_path"]
        with np.load(path, allow_pickle=False) as payload:
            result[row["image_id"]] = {
                "indices": payload["candidate_indices"].astype(np.int64),
                "logits": payload["candidate_logits"].astype(np.float64),
                "selected": int(row["selected_candidate_index"]),
            }
    return result


def cache_rows(cache_root: Path) -> dict[str, dict[str, Any]]:
    rows = [row for row in read_csv(cache_root / "selector_cache_manifest.csv") if row["split"] == "val"]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = cache_root / row["cache_path"]
        with np.load(path, allow_pickle=False) as payload:
            result[row["image_id"]] = {
                "candidate_indices": payload["candidate_indices"].astype(np.int64),
                "family_ids": payload["family_ids"].astype(np.int64),
                "proposal_source_ids": payload["proposal_source_ids"].astype(str),
                "prompt_modes": payload["prompt_modes"].astype(str),
                "shape_features": payload["shape_features"].astype(np.float64),
            }
    return result


def subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["changed"]]
    return {
        "n": len(rows),
        "changed_count": len(changed),
        "changed_fraction": len(changed) / len(rows) if rows else None,
        "candidate_count_mean": mean([row["candidate_count"] for row in rows]),
        "flip_agreement_mean": mean([row["flip_agreement"] for row in rows]),
        "mean_absolute_residual_mean": mean([row["mean_absolute_residual"] for row in rows]),
        "maximum_absolute_residual_mean": mean([row["maximum_absolute_residual"] for row in rows]),
        "base_top_margin_mean": mean([row["base_top_margin"] for row in rows]),
        "primary_top_margin_mean": mean([row["primary_top_margin"] for row in rows]),
    }


def outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["changed"]]
    deltas = [row["dice_delta"] for row in rows]
    changed_deltas = [row["dice_delta"] for row in changed]
    return {
        "n": len(rows),
        "changed_count": len(changed),
        "changed_fraction": len(changed) / len(rows) if rows else None,
        "identity_dice_mean": mean([row["identity_dice"] for row in rows]),
        "primary_dice_mean": mean([row["primary_dice"] for row in rows]),
        "dice_delta_mean": mean(deltas),
        "dice_delta_sum": float(sum(deltas)),
        "dice_delta_quantiles": quantiles(deltas),
        "changed_dice_delta_mean": mean(changed_deltas),
        "changed_dice_delta_median": median(changed_deltas),
        "changed_improved": sum(row["dice_delta"] > 1.0e-12 for row in changed),
        "changed_worsened": sum(row["dice_delta"] < -1.0e-12 for row in changed),
        "changed_tied": sum(abs(row["dice_delta"]) <= 1.0e-12 for row in changed),
        "misses_recovered": sum(row["identity_miss"] and not row["primary_miss"] for row in rows),
        "overlaps_lost": sum(not row["identity_miss"] and row["primary_miss"] for row in rows),
        "selected_area_ratio_identity_mean": mean([row["identity_area"] for row in rows]),
        "selected_area_ratio_primary_mean": mean([row["primary_area"] for row in rows]),
        "selected_area_ratio_delta_mean": mean([row["area_delta"] for row in rows]),
        "changed_selected_area_ratio_delta_mean": mean([row["area_delta"] for row in changed]),
        "selected_to_gt_area_ratio_identity_median": median([row["identity_area"] / row["gt_area"] for row in rows]),
        "selected_to_gt_area_ratio_primary_median": median([row["primary_area"] / row["gt_area"] for row in rows]),
    }


def local_top_dynamics(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tops = [{int(key): int(value) for key, value in row["target"]["local_top_indices"].items()} for row in history]
    keys = sorted(tops[0])
    if any(sorted(row) != keys for row in tops):
        raise RuntimeError("positive-bag local-top key sets differ across epochs")

    def changed(first: int, second: int) -> float:
        return sum(tops[first][key] != tops[second][key] for key in keys) / len(keys)

    consecutive = [changed(index, index + 1) for index in range(len(tops) - 1)]
    unique_counts = [len({row[key] for row in tops}) for key in keys]
    stabilized_unique = [len({row[key] for row in tops[20:]}) for key in keys]
    return {
        "positive_bags": len(keys),
        "changed_fraction_epoch_1_to_2": changed(0, 1),
        "changed_fraction_epoch_1_to_20": changed(0, 19),
        "changed_fraction_epoch_1_to_21": changed(0, 20),
        "changed_fraction_epoch_20_to_21": changed(19, 20),
        "changed_fraction_epoch_21_to_40": changed(20, 39),
        "changed_fraction_epoch_1_to_40": changed(0, 39),
        "consecutive_changed_fraction_mean": mean(consecutive),
        "consecutive_changed_fraction_min": float(min(consecutive)),
        "consecutive_changed_fraction_max": float(max(consecutive)),
        "bags_ever_changing_fraction": sum(value > 1 for value in unique_counts) / len(unique_counts),
        "unique_winners_over_40_epochs_mean": mean(unique_counts),
        "unique_winners_over_40_epochs_median": median(unique_counts),
        "bags_changing_after_mass_stabilized_fraction": sum(value > 1 for value in stabilized_unique) / len(stabilized_unique),
        "unique_winners_epochs_21_to_40_mean": mean(stabilized_unique),
    }


def top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"transition": key, "count": count} for key, count in counter.most_common(limit)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--terminal-result-audit", type=Path, required=True)
    args = parser.parse_args()

    diagnostics_path = args.output_root / "gt_blind_diagnostics.csv"
    history_path = args.output_root / "training_history.json"
    identity_eval_path = args.evaluation_root / "geometry_v3_identity" / "per_image.csv"
    primary_eval_path = args.evaluation_root / "global_local_instance" / "per_image.csv"
    paired_path = args.decision_root / "paired_per_image.csv"
    input_paths = {
        "gt_blind_diagnostics.csv": diagnostics_path,
        "training_history.json": history_path,
        "identity_per_image.csv": identity_eval_path,
        "primary_per_image.csv": primary_eval_path,
        "paired_per_image.csv": paired_path,
        "selector_cache_manifest.csv": args.cache_root / "selector_cache_manifest.csv",
        "identity_candidate_score_manifest.csv": args.output_root / "geometry_v3_identity" / "candidate_scores" / "candidate_score_manifest.csv",
        "primary_candidate_score_manifest.csv": args.output_root / "global_local_instance" / "candidate_scores" / "candidate_score_manifest.csv",
        "terminal_result_audit.json": args.terminal_result_audit,
    }

    terminal_result = json.loads(args.terminal_result_audit.read_text(encoding="utf-8"))
    if terminal_result.get("experiment_id") != EXPERIMENT_ID or terminal_result.get("status") != "COMPLETE_GATE_FAIL":
        raise RuntimeError("terminal result audit is not the frozen S7 gate failure")

    diagnostics = {row["image_id"]: row for row in read_csv(diagnostics_path)}
    identity_eval = {row["image_id"]: row for row in read_csv(identity_eval_path)}
    primary_eval = {row["image_id"]: row for row in read_csv(primary_eval_path)}
    paired = read_csv(paired_path)
    identity_scores = score_payloads(args.output_root / "geometry_v3_identity" / "candidate_scores")
    primary_scores = score_payloads(args.output_root / "global_local_instance" / "candidate_scores")
    cache = cache_rows(args.cache_root)
    if not (set(diagnostics) == set(identity_scores) == set(primary_scores) == set(cache)):
        raise RuntimeError("validation image sets do not match")
    if set(identity_eval) != set(primary_eval) or len(identity_eval) != 184:
        raise RuntimeError("tumor evaluation image sets do not match")

    rows: list[dict[str, Any]] = []
    source_transitions: Counter[str] = Counter()
    prompt_transitions: Counter[str] = Counter()
    for image_id in sorted(diagnostics):
        diag = diagnostics[image_id]
        base = identity_scores[image_id]
        primary = primary_scores[image_id]
        if not np.array_equal(base["indices"], primary["indices"]):
            raise RuntimeError(f"candidate index mismatch for {image_id}")
        indices = base["indices"]
        base_position = winner_position(indices, base["selected"])
        primary_position = winner_position(indices, primary["selected"])
        changed_selection = base_position != primary_position
        if changed_selection != bool(int(diag["selection_changed"])):
            raise RuntimeError(f"selection-change mismatch for {image_id}")
        residual = primary["logits"] - base["logits"]
        cached = cache[image_id]
        if not np.array_equal(indices, cached["candidate_indices"]):
            raise RuntimeError(f"cache candidate index mismatch for {image_id}")
        if changed_selection:
            source_transitions[f"{cached['proposal_source_ids'][base_position]} -> {cached['proposal_source_ids'][primary_position]}"] += 1
            prompt_transitions[f"{cached['prompt_modes'][base_position]} -> {cached['prompt_modes'][primary_position]}"] += 1
        row: dict[str, Any] = {
            "image_id": image_id,
            "tumor": bool(int(diag["tumor"])),
            "changed": changed_selection,
            "candidate_count": int(diag["candidate_count"]),
            "flip_agreement": float(diag["original_flip_agreement"]),
            "mean_absolute_residual": float(diag["mean_absolute_residual"]),
            "maximum_absolute_residual": float(diag["maximum_absolute_residual"]),
            "residual_signed_mean": float(residual.mean()),
            "residual_std": float(residual.std()),
            "base_top_margin": top_margin(base["logits"]),
            "primary_top_margin": top_margin(primary["logits"]),
            "same_family": bool(cached["family_ids"][base_position] == cached["family_ids"][primary_position]),
            "same_source": bool(cached["proposal_source_ids"][base_position] == cached["proposal_source_ids"][primary_position]),
            "same_prompt": bool(cached["prompt_modes"][base_position] == cached["prompt_modes"][primary_position]),
            "identity_area": float(cached["shape_features"][base_position, 0]),
            "primary_area": float(cached["shape_features"][primary_position, 0]),
            "area_delta": float(cached["shape_features"][primary_position, 0] - cached["shape_features"][base_position, 0]),
            "residual_advantage_new_over_old": float(residual[primary_position] - residual[base_position]) if changed_selection else 0.0,
            "base_penalty_new_vs_old": float(base["logits"][base_position] - base["logits"][primary_position]) if changed_selection else 0.0,
            "primary_margin_new_vs_old": float(primary["logits"][primary_position] - primary["logits"][base_position]) if changed_selection else 0.0,
        }
        if row["tumor"]:
            identity = identity_eval[image_id]
            selected = primary_eval[image_id]
            if int(identity["selected_candidate_index"]) != base["selected"] or int(selected["selected_candidate_index"]) != primary["selected"]:
                raise RuntimeError(f"evaluation winner mismatch for {image_id}")
            row.update(
                {
                    "size_group": identity["size_group"],
                    "gt_area": float(identity["gt_area_ratio"]),
                    "identity_dice": float(identity["dice"]),
                    "primary_dice": float(selected["dice"]),
                    "dice_delta": float(selected["dice"]) - float(identity["dice"]),
                    "identity_miss": bool(int(identity["complete_miss"])),
                    "primary_miss": bool(int(selected["complete_miss"])),
                    "score_quality_delta": (
                        optional_float(selected["score_quality_spearman"])
                        - optional_float(identity["score_quality_spearman"])
                        if optional_float(selected["score_quality_spearman"]) is not None
                        and optional_float(identity["score_quality_spearman"]) is not None
                        else None
                    ),
                    "regret_delta": float(selected["selected_to_oracle_regret"]) - float(identity["selected_to_oracle_regret"]),
                }
            )
        rows.append(row)

    tumor_rows = [row for row in rows if row["tumor"]]
    normal_rows = [row for row in rows if not row["tumor"]]
    changed_rows = [row for row in rows if row["changed"]]
    changed_tumor = [row for row in tumor_rows if row["changed"]]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epoch_indices = [0, 9, 19, 20, 29, 39]
    epoch_summary: list[dict[str, Any]] = []
    for index in epoch_indices:
        item = history[index]
        target = item["target"]
        epoch_summary.append(
            {
                "epoch": int(item["epoch"]),
                "target_mass": float(target["target_positive_mass"]),
                "realized_mass_after_local": float(target["realized_mass_after_local"]),
                "local_mass_uplift": float(target["realized_mass_after_local"] - target["projected_mass_before_local"]),
                "instance_loss": float(item["instance"]),
                "consistency_raw": float(item["consistency"]),
                "consistency_weighted": 0.1 * float(item["consistency"]),
                "drift_raw": float(item["drift"]),
                "drift_weighted": 0.001 * float(item["drift"]),
                "total_loss": float(item["total"]),
            }
        )

    dossier = {
        "analysis_id": "mask_bag_global_local_instance_s7_failure_analysis_v1",
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "analysis_scope": "post-hoc diagnosis of a frozen rejected arm; not model selection or rescue",
        "experiment_id": EXPERIMENT_ID,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()},
        "safety": {
            "consumer_trained": False,
            "predictions_modified": False,
            "post_hoc_rescue_or_sweep_performed": False,
            "test_evaluated": False,
            "validation_gt_used_only_from_frozen_evaluation_tables": True,
        },
        "selection_churn": {
            "all": subset_summary(rows),
            "tumor": subset_summary(tumor_rows),
            "normal": subset_summary(normal_rows),
            "changed": subset_summary(changed_rows),
            "unchanged": subset_summary([row for row in rows if not row["changed"]]),
        },
        "tumor_outcomes": {
            "overall": outcome_summary(tumor_rows),
            "small": outcome_summary([row for row in tumor_rows if row["size_group"] == "small"]),
            "medium": outcome_summary([row for row in tumor_rows if row["size_group"] == "medium"]),
            "large": outcome_summary([row for row in tumor_rows if row["size_group"] == "large"]),
        },
        "changed_candidate_transitions": {
            "n_all": len(changed_rows),
            "n_tumor": len(changed_tumor),
            "same_family_fraction_all": mean([float(row["same_family"]) for row in changed_rows]),
            "same_source_fraction_all": mean([float(row["same_source"]) for row in changed_rows]),
            "same_prompt_fraction_all": mean([float(row["same_prompt"]) for row in changed_rows]),
            "same_family_fraction_tumor": mean([float(row["same_family"]) for row in changed_tumor]),
            "same_source_fraction_tumor": mean([float(row["same_source"]) for row in changed_tumor]),
            "same_prompt_fraction_tumor": mean([float(row["same_prompt"]) for row in changed_tumor]),
            "residual_advantage_new_over_old_mean": mean([row["residual_advantage_new_over_old"] for row in changed_rows]),
            "base_penalty_new_vs_old_mean": mean([row["base_penalty_new_vs_old"] for row in changed_rows]),
            "primary_margin_new_vs_old_mean": mean([row["primary_margin_new_vs_old"] for row in changed_rows]),
            "top_source_transitions_all": top_counter(source_transitions),
            "top_prompt_transitions_all": top_counter(prompt_transitions),
        },
        "diagnostic_correlations": {
            "tumor_dice_delta_vs_mean_absolute_residual": spearman([row["dice_delta"] for row in tumor_rows], [row["mean_absolute_residual"] for row in tumor_rows]),
            "tumor_dice_delta_vs_maximum_absolute_residual": spearman([row["dice_delta"] for row in tumor_rows], [row["maximum_absolute_residual"] for row in tumor_rows]),
            "tumor_dice_delta_vs_candidate_count": spearman([row["dice_delta"] for row in tumor_rows], [row["candidate_count"] for row in tumor_rows]),
            "changed_tumor_dice_delta_vs_residual_advantage": spearman([row["dice_delta"] for row in changed_tumor], [row["residual_advantage_new_over_old"] for row in changed_tumor]),
            "changed_tumor_dice_delta_vs_base_penalty": spearman([row["dice_delta"] for row in changed_tumor], [row["base_penalty_new_vs_old"] for row in changed_tumor]),
            "changed_tumor_dice_delta_vs_area_delta": spearman([row["dice_delta"] for row in changed_tumor], [row["area_delta"] for row in changed_tumor]),
        },
        "ranking_quality": {
            "score_quality_spearman_delta_mean": mean([row["score_quality_delta"] for row in tumor_rows if row["score_quality_delta"] is not None]),
            "score_quality_spearman_delta_changed_mean": mean([row["score_quality_delta"] for row in changed_tumor if row["score_quality_delta"] is not None]),
            "selected_to_oracle_regret_delta_mean": mean([row["regret_delta"] for row in tumor_rows]),
            "selected_to_oracle_regret_delta_changed_mean": mean([row["regret_delta"] for row in changed_tumor]),
        },
        "training_dynamics": {
            "epochs": 40,
            "positive_bags": int(history[0]["target"]["positive_bags"]),
            "negative_bags": int(history[0]["target"]["negative_bags"]),
            "positive_bag_candidates": int(history[0]["target"]["positive_bag_candidates"]),
            "locally_forced_candidates_each_epoch": int(history[0]["target"]["locally_forced_candidates"]),
            "selected_epochs": epoch_summary,
            "drift_raw_epoch_40_over_epoch_1": float(history[39]["drift"] / history[0]["drift"]),
            "local_top_dynamics": local_top_dynamics(history),
        },
        "paired_table_rows": len(paired),
    }
    print(json.dumps(dossier, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
