"""Read-only failure analysis for the frozen S8 selector comparison.

The utility consumes only physically frozen producer artifacts and validation
evaluation tables.  It never creates predictions, opens raw validation GT,
touches BTXRD test, trains a consumer, or selects a post-hoc rescue arm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260802-codex-s8-skelex-reconstruction-randomization-v1"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank"
EPSILON = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


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


def classify_delta(delta: float) -> str:
    if delta > EPSILON:
        return "win"
    if delta < -EPSILON:
        return "loss"
    return "tie"


def mask_pair_dice(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first) > 0
    right = np.asarray(second) > 0
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(left, right).sum() / denominator)


def subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta_dice"]) for row in rows if row.get("delta_dice") is not None]
    return {
        "n": len(rows),
        "wins": sum(classify_delta(value) == "win" for value in deltas),
        "losses": sum(classify_delta(value) == "loss" for value in deltas),
        "ties": sum(classify_delta(value) == "tie" for value in deltas),
        "delta_sum": float(sum(deltas)) if deltas else None,
        "positive_delta_sum": float(sum(max(value, 0.0) for value in deltas)) if deltas else None,
        "negative_delta_sum": float(sum(min(value, 0.0) for value in deltas)) if deltas else None,
        "permutation_p_value_median": median([float(row["permutation_p_value"]) for row in rows]),
        "observed_improvement_median": median([float(row["observed_improvement"]) for row in rows]),
        "new_to_old_area_ratio_median": median([float(row["new_to_old_area_ratio"]) for row in rows]),
        "mask_pair_dice_median": median([float(row["mask_pair_dice"]) for row in rows]),
        "same_family_count": sum(bool(row["same_family"]) for row in rows),
        "area_shrink_count": sum(float(row["new_to_old_area_ratio"]) < 1.0 for row in rows),
        "area_below_half_count": sum(float(row["new_to_old_area_ratio"]) < 0.5 for row in rows),
        "near_disjoint_count": sum(float(row["mask_pair_dice"]) < 0.1 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--control-evaluation-root", type=Path, required=True)
    parser.add_argument("--primary-evaluation-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output}")

    decision_path = args.decision_root / "gate_decision.json"
    comparison_path = args.decision_root / "paired_comparison.json"
    paired_path = args.decision_root / "paired_per_image.csv"
    decision_audit_path = args.decision_root / "decision_audit.json"
    control_manifest_path = args.producer_root / CONTROL_ARM / "predictions" / "prediction_manifest.csv"
    primary_manifest_path = args.producer_root / PRIMARY_ARM / "predictions" / "prediction_manifest.csv"
    evidence_manifest_path = args.producer_root / "reconstruction_evidence" / "evidence_manifest.json"
    pair_freeze_path = args.producer_root / "prediction_pair_freeze.json"
    control_eval_path = args.control_evaluation_root / "per_image.csv"
    primary_eval_path = args.primary_evaluation_root / "per_image.csv"

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision_audit = json.loads(decision_audit_path.read_text(encoding="utf-8"))
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    pair_freeze = json.loads(pair_freeze_path.read_text(encoding="utf-8"))
    if decision.get("status") != "FAIL" or decision.get("operational_pass") is not False:
        raise RuntimeError("decision is not the frozen S8 gate failure")
    for payload, name in [
        (decision_audit, "decision audit"),
        (evidence_manifest, "evidence manifest"),
        (pair_freeze, "pair freeze"),
    ]:
        if payload.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"{name} experiment mismatch")
    if decision_audit.get("ground_truth_reopened_for_matched_comparison") is not False:
        raise RuntimeError("matched comparison unexpectedly reopened GT")
    if decision.get("consumer_authorized") is not False or decision.get("test_evaluated") is not False:
        raise RuntimeError("S8 safety state is not fail-closed")

    control_rows = read_csv(control_manifest_path)
    primary_rows = read_csv(primary_manifest_path)
    control = {row["image_id"]: row for row in control_rows}
    primary = {row["image_id"]: row for row in primary_rows}
    paired = {row["image_id"]: row for row in read_csv(paired_path)}
    control_eval = {row["image_id"]: row for row in read_csv(control_eval_path)}
    primary_eval = {row["image_id"]: row for row in read_csv(primary_eval_path)}
    evidence = {row["image_id"]: row for row in evidence_manifest["rows"]}
    if not (len(control) == len(primary) == len(evidence) == 371):
        raise RuntimeError("producer cohort is not exactly 371 images")
    if set(control) != set(primary) or set(control) != set(evidence):
        raise RuntimeError("producer/evidence image sets differ")
    if not (len(paired) == len(control_eval) == len(primary_eval) == 184):
        raise RuntimeError("evaluation cohort is not exactly 184 tumor images")
    if set(paired) != set(control_eval) or set(paired) != set(primary_eval):
        raise RuntimeError("frozen evaluation image sets differ")

    changed: list[dict[str, Any]] = []
    for image_id in sorted(control):
        old = control[image_id]
        new = primary[image_id]
        old_index = int(old["selected_candidate_index"])
        new_index = int(new["selected_candidate_index"])
        if old_index == new_index:
            continue
        evidence_row = evidence[image_id]
        evidence_path = args.producer_root / evidence_row["evidence_path"]
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise RuntimeError(f"evidence hash mismatch for {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            indices = payload["candidate_indices"].astype(np.int64)
            old_position = np.flatnonzero(indices == old_index)
            new_position = np.flatnonzero(indices == new_index)
            if old_position.size != 1 or new_position.size != 1:
                raise RuntimeError(f"candidate index mismatch for {image_id}")
            old_position = int(old_position[0])
            new_position = int(new_position[0])
            if int(payload["switched"]) != 1 or int(payload["selected_index"]) != new_index:
                raise RuntimeError(f"evidence switch mismatch for {image_id}")
            old_family = int(payload["family_ids"][old_position])
            new_family = int(payload["family_ids"][new_position])
            old_base = float(payload["base_scores"][old_position])
            new_base = float(payload["base_scores"][new_position])
            observed_improvement = float(payload["observed_improvement"])
            permutation_p_value = float(payload["permutation_p_value"])

        old_map_path = args.producer_root / CONTROL_ARM / "predictions" / old["map_path"]
        new_map_path = args.producer_root / PRIMARY_ARM / "predictions" / new["map_path"]
        if sha256_file(old_map_path) != old["map_sha256"] or sha256_file(new_map_path) != new["map_sha256"]:
            raise RuntimeError(f"prediction map hash mismatch for {image_id}")
        old_area = float(old["selected_area_ratio"])
        new_area = float(new["selected_area_ratio"])
        if old_area <= 0.0:
            raise RuntimeError(f"non-positive control area for {image_id}")

        eval_row = paired.get(image_id)
        delta = None if eval_row is None else float(eval_row["delta_dice"])
        if eval_row is not None:
            if abs(float(control_eval[image_id]["dice"]) - float(eval_row["control_dice"])) > EPSILON:
                raise RuntimeError(f"control evaluation mismatch for {image_id}")
            if abs(float(primary_eval[image_id]["dice"]) - float(eval_row["primary_dice"])) > EPSILON:
                raise RuntimeError(f"primary evaluation mismatch for {image_id}")
        changed.append(
            {
                "image_id": image_id,
                "tumor": bool(int(old["tumor"])),
                "size_group": None if eval_row is None else eval_row["size_group"],
                "delta_dice": delta,
                "outcome": None if delta is None else classify_delta(delta),
                "control_complete_miss": None if eval_row is None else bool(int(eval_row["control_complete_miss"])),
                "primary_complete_miss": None if eval_row is None else bool(int(eval_row["primary_complete_miss"])),
                "control_candidate_index": old_index,
                "primary_candidate_index": new_index,
                "control_family": old_family,
                "primary_family": new_family,
                "same_family": old_family == new_family,
                "control_base_score": old_base,
                "primary_base_score": new_base,
                "base_score_delta": new_base - old_base,
                "permutation_p_value": permutation_p_value,
                "observed_improvement": observed_improvement,
                "control_area_ratio": old_area,
                "primary_area_ratio": new_area,
                "new_to_old_area_ratio": new_area / old_area,
                "mask_pair_dice": mask_pair_dice(
                    np.load(old_map_path, allow_pickle=False),
                    np.load(new_map_path, allow_pickle=False),
                ),
            }
        )

    evidence_switch_count = sum(int(row["switched"]) for row in evidence_manifest["rows"])
    if len(changed) != evidence_switch_count or len(changed) != 20:
        raise RuntimeError("switch count does not match frozen evidence")
    changed_tumor = [row for row in changed if row["tumor"]]
    changed_normal = [row for row in changed if not row["tumor"]]
    wins = [row for row in changed_tumor if row["outcome"] == "win"]
    losses = [row for row in changed_tumor if row["outcome"] == "loss"]
    ties = [row for row in changed_tumor if row["outcome"] == "tie"]
    tumor_delta_sum = float(sum(row["delta_dice"] for row in changed_tumor))
    largest = max(changed_tumor, key=lambda row: row["delta_dice"])

    input_paths = {
        "decision_audit.json": decision_audit_path,
        "gate_decision.json": decision_path,
        "paired_comparison.json": comparison_path,
        "paired_per_image.csv": paired_path,
        "prediction_pair_freeze.json": pair_freeze_path,
        "evidence_manifest.json": evidence_manifest_path,
        "control_prediction_manifest.csv": control_manifest_path,
        "primary_prediction_manifest.csv": primary_manifest_path,
        "control_evaluation_per_image.csv": control_eval_path,
        "primary_evaluation_per_image.csv": primary_eval_path,
    }
    dossier = {
        "analysis_id": "skelex_reconstruction_selector_s8_failure_analysis_v1",
        "analysis_scope": "read-only diagnosis of a frozen rejected arm; not model selection or rescue",
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "experiment_id": EXPERIMENT_ID,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()},
        "safety": {
            "consumer_trained": False,
            "predictions_modified": False,
            "post_hoc_rescue_or_sweep_performed": False,
            "raw_validation_gt_opened": False,
            "test_evaluated": False,
            "gt_derived_data_used_only_from_frozen_evaluation_tables": True,
        },
        "switch_incidence": {
            "all": {"switched": len(changed), "cohort": 371, "fraction": len(changed) / 371},
            "tumor": {"switched": len(changed_tumor), "cohort": 184, "fraction": len(changed_tumor) / 184},
            "normal": {"switched": len(changed_normal), "cohort": 187, "fraction": len(changed_normal) / 187},
            "tumor_to_normal_switch_rate_ratio": (len(changed_tumor) / 184) / (len(changed_normal) / 187),
        },
        "changed_outcomes": {
            "all": subset_summary(changed),
            "tumor": subset_summary(changed_tumor),
            "normal": subset_summary(changed_normal),
            "win": subset_summary(wins),
            "loss": subset_summary(losses),
            "tie": subset_summary(ties),
            "small": subset_summary([row for row in changed_tumor if row["size_group"] == "small"]),
            "medium": subset_summary([row for row in changed_tumor if row["size_group"] == "medium"]),
            "large": subset_summary([row for row in changed_tumor if row["size_group"] == "large"]),
        },
        "concentration": {
            "tumor_delta_sum": tumor_delta_sum,
            "largest_positive_contributor": {"image_id": largest["image_id"], "delta_dice": largest["delta_dice"]},
            "largest_contributor_fraction_of_net": largest["delta_dice"] / tumor_delta_sum,
            "tumor_delta_sum_without_largest_contributor": tumor_delta_sum - largest["delta_dice"],
        },
        "miss_transitions": {
            "recovered": sum(row["control_complete_miss"] and not row["primary_complete_miss"] for row in changed_tumor),
            "lost": sum(not row["control_complete_miss"] and row["primary_complete_miss"] for row in changed_tumor),
            "net": sum(row["control_complete_miss"] and not row["primary_complete_miss"] for row in changed_tumor)
            - sum(not row["control_complete_miss"] and row["primary_complete_miss"] for row in changed_tumor),
        },
        "diagnostic_association": {
            "changed_tumor_delta_vs_permutation_p_value_spearman": spearman(
                [row["delta_dice"] for row in changed_tumor],
                [row["permutation_p_value"] for row in changed_tumor],
            ),
            "changed_tumor_delta_vs_observed_improvement_spearman": spearman(
                [row["delta_dice"] for row in changed_tumor],
                [row["observed_improvement"] for row in changed_tumor],
            ),
            "changed_tumor_delta_vs_new_to_old_area_ratio_spearman": spearman(
                [row["delta_dice"] for row in changed_tumor],
                [row["new_to_old_area_ratio"] for row in changed_tumor],
            ),
            "changed_tumor_delta_vs_mask_pair_dice_spearman": spearman(
                [row["delta_dice"] for row in changed_tumor],
                [row["mask_pair_dice"] for row in changed_tumor],
            ),
        },
        "mechanism_interpretation": {
            "failure_type": "scientific_gate_failure_after_valid_evaluation",
            "supported": [
                "The spatial-null test detects non-random reconstruction evidence but does not establish tumor relevance or superiority to the accepted selector.",
                "Switching is not tumor-specific: tumor and normal switch rates are nearly equal.",
                "Accepted switches usually replace the baseline with a different-family, substantially smaller, weakly overlapping mask.",
                "The positive mean is fragile and dominated by one recovered medium lesion; without it the net switched-tumor delta is negative.",
                "Permutation significance does not separate beneficial from harmful tumor switches.",
            ],
            "rejected_or_unsupported": [
                "Promoting S8 as a selector improvement.",
                "Post-hoc threshold, p-value, weight, area, family, subgroup, or fusion sweeps on this frozen validation result.",
                "Treating orientation-consistent reconstruction winners as identity-consistent tumor candidates.",
                "A global area correction, because both beneficial and harmful switches can shrink support.",
            ],
            "transferable": [
                "Keep the frozen SKELEX reconstruction evidence as a diagnostic representation only.",
                "A successor must learn or predeclare an annotation-free identity/extent guard independently of this validation outcome and compare against the accepted control.",
                "Any successor must target cross-family identity ambiguity and extent collapse rather than merely increasing reconstruction significance.",
            ],
        },
        "changed_rows": changed,
    }
    write_json_exclusive(args.output, dossier)


if __name__ == "__main__":
    main()
