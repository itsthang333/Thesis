"""Read-only mechanism analysis for a frozen, rejected S9 comparison.

This utility never opens raw images, annotations, candidate archives, or BTXRD
test. It consumes only hash-frozen producer evidence and already-created
post-freeze evaluation/decision tables. It cannot create a rescue selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_s9_likelihood_equal_rank"
EXPECTED_VALIDATION = 371
EXPECTED_TUMOR = 184
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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _rankdata(values: Sequence[float]) -> np.ndarray:
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
    left = _rankdata(first)
    right = _rankdata(second)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _outcome(delta: float) -> str:
    if delta > EPSILON:
        return "win"
    if delta < -EPSILON:
        return "loss"
    return "tie"


def _subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta_dice"]) for row in rows if row.get("delta_dice") is not None]
    return {
        "n": len(rows),
        "wins": sum(_outcome(value) == "win" for value in deltas),
        "losses": sum(_outcome(value) == "loss" for value in deltas),
        "ties": sum(_outcome(value) == "tie" for value in deltas),
        "mean_delta_dice": _mean(deltas),
        "median_delta_dice": _median(deltas),
        "mean_likelihood_margin": _mean(
            [float(row["likelihood_margin_primary_minus_control"]) for row in rows]
        ),
        "median_primary_to_control_area_ratio": _median(
            [float(row["primary_to_control_area_ratio"]) for row in rows]
        ),
    }


def analyze(
    *,
    producer_root: Path,
    control_evaluation_root: Path,
    primary_evaluation_root: Path,
    decision_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    pair_path = producer_root / "prediction_pair_freeze.json"
    wrapper_audit_path = producer_root / "wrapper_output_audit.json"
    independent_path = producer_root / "independent_gt_blind_output_audit.json"
    evidence_manifest_path = producer_root / "s9_likelihood_evidence/evidence_manifest.csv"
    control_manifest_path = producer_root / CONTROL_ARM / "predictions/prediction_manifest.csv"
    primary_manifest_path = producer_root / PRIMARY_ARM / "predictions/prediction_manifest.csv"
    decision_path = decision_root / "gate_decision.json"
    decision_audit_path = decision_root / "decision_audit.json"
    paired_path = decision_root / "paired_per_image.csv"
    control_eval_path = control_evaluation_root / "per_image.csv"
    primary_eval_path = primary_evaluation_root / "per_image.csv"

    pair = _json(pair_path)
    wrapper_audit = _json(wrapper_audit_path)
    independent = _json(independent_path)
    decision = _json(decision_path)
    decision_audit = _json(decision_audit_path)
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or wrapper_audit.get("independent_gt_blind_output_audit_sha256")
        != sha256_file(independent_path)
        or independent.get("status")
        != "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_REPRODUCTION_PASS"
        or decision.get("status") not in {"FAIL", "MECHANISM_PASS"}
        or decision.get("operational_pass") is not False
        or decision.get("consumer_authorized") is not False
        or decision.get("test_evaluated") is not False
        or decision_audit.get("ground_truth_reopened_for_matched_comparison") is not False
    ):
        raise RuntimeError("S9 frozen failure/safety contract mismatch")

    control_rows = read_csv(control_manifest_path)
    primary_rows = read_csv(primary_manifest_path)
    evidence_rows = read_csv(evidence_manifest_path)
    if not (
        len(control_rows) == len(primary_rows) == len(evidence_rows) == EXPECTED_VALIDATION
    ):
        raise RuntimeError("S9 producer cohort is not exactly 371")
    control = {row["image_id"]: row for row in control_rows}
    primary = {row["image_id"]: row for row in primary_rows}
    evidence = {row["image_id"]: row for row in evidence_rows}
    if set(control) != set(primary) or set(control) != set(evidence):
        raise RuntimeError("S9 producer/evidence image sets differ")

    paired_rows = read_csv(paired_path)
    control_eval_rows = read_csv(control_eval_path)
    primary_eval_rows = read_csv(primary_eval_path)
    if not (
        len(paired_rows)
        == len(control_eval_rows)
        == len(primary_eval_rows)
        == EXPECTED_TUMOR
    ):
        raise RuntimeError("S9 evaluated tumor cohort is not exactly 184")
    paired = {row["image_id"]: row for row in paired_rows}
    control_eval = {row["image_id"]: row for row in control_eval_rows}
    primary_eval = {row["image_id"]: row for row in primary_eval_rows}
    if set(paired) != set(control_eval) or set(paired) != set(primary_eval):
        raise RuntimeError("S9 frozen evaluation image sets differ")

    per_image_mechanics: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for image_id in sorted(control):
        control_row = control[image_id]
        primary_row = primary[image_id]
        if control_row["tumor"] != primary_row["tumor"]:
            raise RuntimeError(f"S9 tumor flag mismatch: {image_id}")
        evidence_row = evidence[image_id]
        evidence_path = producer_root / "s9_likelihood_evidence" / evidence_row["evidence_path"]
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise RuntimeError(f"S9 evidence hash mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
            base_logits = np.asarray(payload["baseline_logits"], dtype=np.float32)
            upstream = np.asarray(payload["upstream_scores"], dtype=np.float32)
            likelihood = np.asarray(payload["candidate_likelihood"], dtype=np.float32)
            control_rank = np.asarray(payload["control_rank"], dtype=np.float32)
            primary_rank = np.asarray(payload["primary_rank"], dtype=np.float32)
            candidate_weights = np.asarray(payload["candidate_weights"], dtype=np.float32)
        if not (
            indices.ndim == 1
            and base_logits.shape == upstream.shape == likelihood.shape == control_rank.shape
            and control_rank.shape == primary_rank.shape == indices.shape
            and candidate_weights.shape[0] == len(indices)
            and np.isfinite(base_logits).all()
            and np.isfinite(upstream).all()
            and np.isfinite(likelihood).all()
            and np.isfinite(control_rank).all()
            and np.isfinite(primary_rank).all()
        ):
            raise RuntimeError(f"S9 evidence tensor contract mismatch: {image_id}")
        control_local = int(np.argmax(control_rank))
        primary_local = int(np.argmax(primary_rank))
        if (
            int(indices[control_local]) != int(control_row["selected_candidate_index"])
            or int(indices[primary_local]) != int(primary_row["selected_candidate_index"])
        ):
            raise RuntimeError(f"S9 selected candidate mismatch: {image_id}")
        area = candidate_weights.reshape(len(indices), -1).mean(axis=1)
        mechanics = {
            "image_id": image_id,
            "tumor": bool(int(control_row["tumor"])),
            "candidate_count": len(indices),
            "likelihood_range": float(likelihood.max() - likelihood.min()),
            "likelihood_std": float(likelihood.std()),
            "likelihood_vs_area_spearman": spearman(likelihood.tolist(), area.tolist()),
            "likelihood_vs_base_spearman": spearman(likelihood.tolist(), base_logits.tolist()),
            "likelihood_vs_upstream_spearman": spearman(likelihood.tolist(), upstream.tolist()),
            "control_primary_spearman": spearman(control_rank.tolist(), primary_rank.tolist()),
            "changed": control_local != primary_local,
        }
        per_image_mechanics.append(mechanics)
        if control_local == primary_local:
            continue
        control_area = float(control_row["selected_area_ratio"])
        primary_area = float(primary_row["selected_area_ratio"])
        if control_area <= 0.0:
            raise RuntimeError(f"S9 non-positive control area: {image_id}")
        eval_row = paired.get(image_id)
        delta = None if eval_row is None else float(eval_row["delta_dice"])
        if eval_row is not None:
            if (
                abs(float(control_eval[image_id]["dice"]) - float(eval_row["control_dice"]))
                > EPSILON
                or abs(
                    float(primary_eval[image_id]["dice"])
                    - float(eval_row["primary_dice"])
                )
                > EPSILON
            ):
                raise RuntimeError(f"S9 evaluation mismatch: {image_id}")
        changed.append(
            {
                "image_id": image_id,
                "tumor": bool(int(control_row["tumor"])),
                "size_group": None if eval_row is None else eval_row["size_group"],
                "delta_dice": delta,
                "outcome": None if delta is None else _outcome(delta),
                "control_candidate_index": int(indices[control_local]),
                "primary_candidate_index": int(indices[primary_local]),
                "likelihood_control": float(likelihood[control_local]),
                "likelihood_primary": float(likelihood[primary_local]),
                "likelihood_margin_primary_minus_control": float(
                    likelihood[primary_local] - likelihood[control_local]
                ),
                "control_area_ratio": control_area,
                "primary_area_ratio": primary_area,
                "primary_to_control_area_ratio": primary_area / control_area,
            }
        )

    changed_tumor = [row for row in changed if row["tumor"]]
    changed_normal = [row for row in changed if not row["tumor"]]
    subgroup_rows = {
        subgroup: [row for row in changed_tumor if row["size_group"] == subgroup]
        for subgroup in ("small", "medium", "large")
    }
    tumor_rate = len(changed_tumor) / EXPECTED_TUMOR
    normal_rate = len(changed_normal) / (EXPECTED_VALIDATION - EXPECTED_TUMOR)
    mechanics_tumor = [row for row in per_image_mechanics if row["tumor"]]
    mechanics_normal = [row for row in per_image_mechanics if not row["tumor"]]

    inputs = {
        "prediction_pair_freeze.json": pair_path,
        "wrapper_output_audit.json": wrapper_audit_path,
        "independent_gt_blind_output_audit.json": independent_path,
        "evidence_manifest.csv": evidence_manifest_path,
        "control_prediction_manifest.csv": control_manifest_path,
        "primary_prediction_manifest.csv": primary_manifest_path,
        "gate_decision.json": decision_path,
        "decision_audit.json": decision_audit_path,
        "paired_per_image.csv": paired_path,
        "control_per_image.csv": control_eval_path,
        "primary_per_image.csv": primary_eval_path,
    }
    result = {
        "analysis_id": "skelex_candidate_marginal_s9_failure_analysis_v1",
        "experiment_id": EXPERIMENT_ID,
        "analysis_scope": (
            "read-only diagnosis of frozen non-operational outputs; no rescue or selection"
        ),
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "switch_incidence": {
            "all": {"changed": len(changed), "cohort": EXPECTED_VALIDATION},
            "tumor": {
                "changed": len(changed_tumor),
                "cohort": EXPECTED_TUMOR,
                "fraction": tumor_rate,
            },
            "normal": {
                "changed": len(changed_normal),
                "cohort": EXPECTED_VALIDATION - EXPECTED_TUMOR,
                "fraction": normal_rate,
            },
            "tumor_to_normal_rate_ratio": None if normal_rate == 0.0 else tumor_rate / normal_rate,
        },
        "changed_outcomes": {
            "tumor": _subset(changed_tumor),
            **{subgroup: _subset(rows) for subgroup, rows in subgroup_rows.items()},
        },
        "likelihood_mechanics": {
            cohort: {
                "median_range": _median([float(row["likelihood_range"]) for row in rows]),
                "median_std": _median([float(row["likelihood_std"]) for row in rows]),
                "median_likelihood_vs_area_spearman": _median(
                    [
                        float(row["likelihood_vs_area_spearman"])
                        for row in rows
                        if row["likelihood_vs_area_spearman"] is not None
                    ]
                ),
                "median_likelihood_vs_base_spearman": _median(
                    [
                        float(row["likelihood_vs_base_spearman"])
                        for row in rows
                        if row["likelihood_vs_base_spearman"] is not None
                    ]
                ),
                "median_likelihood_vs_upstream_spearman": _median(
                    [
                        float(row["likelihood_vs_upstream_spearman"])
                        for row in rows
                        if row["likelihood_vs_upstream_spearman"] is not None
                    ]
                ),
                "median_control_primary_spearman": _median(
                    [
                        float(row["control_primary_spearman"])
                        for row in rows
                        if row["control_primary_spearman"] is not None
                    ]
                ),
            }
            for cohort, rows in (("tumor", mechanics_tumor), ("normal", mechanics_normal))
        },
        "safety": {
            "raw_validation_gt_opened": False,
            "gt_derived_data_used_only_from_frozen_evaluation_tables": True,
            "predictions_modified": False,
            "post_hoc_rescue_or_sweep_performed": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        "changed_rows": changed,
    }
    _write_json_exclusive(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--control-evaluation-root", type=Path, required=True)
    parser.add_argument("--primary-evaluation-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        producer_root=args.producer_root,
        control_evaluation_root=args.control_evaluation_root,
        primary_evaluation_root=args.primary_evaluation_root,
        decision_root=args.decision_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
