from __future__ import annotations

"""Decide a frozen S1 matched pair without reopening segmentation GT."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mae_reconstruction_io import sha256_file, validate_sha256


PROTOCOL_SHA256 = "62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413"
SUBGROUPS = ("overall", "small", "medium", "large")
TUMOR_SUBGROUPS = ("small", "medium", "large")
EXPECTED_COHORT = {
    "validation": 371,
    "tumor": 184,
    "normal": 187,
    "small": 94,
    "medium": 72,
    "large": 18,
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_safety(payload: Mapping[str, object], *, name: str) -> None:
    if payload.get("consumer_trained") is not False or payload.get("test_evaluated") is not False:
        raise ValueError(f"{name} safety mismatch")


def _verify_evaluation(
    root: Path,
    expected_audit_sha256: str,
    *,
    expected_arm_freeze_sha256: str,
) -> dict[str, Any]:
    audit_path = root / "evaluation_audit.json"
    if sha256_file(audit_path) != validate_sha256(
        expected_audit_sha256, name="evaluation audit SHA-256"
    ):
        raise ValueError("Evaluation-audit SHA-256 mismatch")
    audit = _json(audit_path)
    _require_safety(audit, name="evaluation audit")
    if (
        audit.get("arm_prediction_freeze_sha256") != expected_arm_freeze_sha256
        or audit.get("cohort") != EXPECTED_COHORT
        or audit.get("bootstrap_replicates") != 10000
        or audit.get("validation_gt_read_only_after_all_predictions_frozen_and_verified")
        is not True
    ):
        raise ValueError("Evaluation provenance/cohort mismatch")
    hashes = audit.get("output_hashes")
    expected_files = {
        "summary.json",
        "gate_decision.json",
        "paired_comparison.json",
        "per_image.csv",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise ValueError("Evaluation output inventory mismatch")
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Evaluation output hash mismatch: {name}")
    summary = _json(root / "summary.json")
    gate = _json(root / "gate_decision.json")
    paired = _json(root / "paired_comparison.json")
    for name, payload in (("summary", summary), ("gate", gate), ("paired", paired)):
        _require_safety(payload, name=name)
    if (
        summary.get("cohort") != EXPECTED_COHORT
        or summary.get("arm_protocol_sha256") != PROTOCOL_SHA256
        or paired.get("replicates") != 10000
        or paired.get("method") != "paired complete-group bootstrap"
        or gate.get("gate_id") != "mask_bag_selector_arm_gate_v1"
        or gate.get("status") not in {"FAIL", "MECHANISM_PASS", "OPERATIONAL_PASS"}
        or bool(gate.get("consumer_authorized"))
        != (gate.get("status") == "OPERATIONAL_PASS")
    ):
        raise ValueError("Evaluation scientific contract mismatch")
    subgroups = summary.get("subgroups")
    if not isinstance(subgroups, dict) or set(subgroups) != set(SUBGROUPS):
        raise ValueError("Evaluation subgroup schema mismatch")
    return {
        "audit": audit,
        "summary": summary,
        "gate": gate,
        "paired": paired,
        "audit_sha256": expected_audit_sha256,
        "per_image_sha256": hashes["per_image.csv"],
    }


def decide_s1_pair(
    pair_root: Path,
    pair_output_audit_path: Path,
    expected_pair_output_audit_sha256: str,
    standard_root: Path,
    expected_standard_audit_sha256: str,
    family_root: Path,
    expected_family_audit_sha256: str,
    matched_comparison_path: Path,
    expected_matched_comparison_sha256: str,
) -> dict[str, object]:
    expected_pair_audit = validate_sha256(
        expected_pair_output_audit_sha256, name="pair output audit SHA-256"
    )
    if sha256_file(pair_output_audit_path) != expected_pair_audit:
        raise ValueError("Pair output audit SHA-256 mismatch")
    pair_audit = _json(pair_output_audit_path)
    _require_safety(pair_audit, name="pair output audit")
    pair_freeze_path = pair_root / "pair_prediction_freeze.json"
    if (
        pair_audit.get("status")
        != "MATCHED_PAIR_PREDICTIONS_PHYSICALLY_VERIFIED_GT_BLIND"
        or pair_audit.get("protocol_sha256") != PROTOCOL_SHA256
        or pair_audit.get("validation_gt_read") is not False
        or sha256_file(pair_freeze_path)
        != pair_audit.get("pair_prediction_freeze_sha256")
    ):
        raise ValueError("Pair output audit/freeze contract mismatch")
    pair_freeze = _json(pair_freeze_path)
    _require_safety(pair_freeze, name="pair freeze")
    arms = pair_freeze.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"standard", "family_balanced"}:
        raise ValueError("Pair freeze arm inventory mismatch")
    standard = _verify_evaluation(
        standard_root,
        expected_standard_audit_sha256,
        expected_arm_freeze_sha256=arms["standard"]["prediction_freeze_sha256"],
    )
    family = _verify_evaluation(
        family_root,
        expected_family_audit_sha256,
        expected_arm_freeze_sha256=arms["family_balanced"]["prediction_freeze_sha256"],
    )
    if (
        standard["audit"]["baseline_prediction_freeze_sha256"]
        != family["audit"]["baseline_prediction_freeze_sha256"]
        or standard["audit"]["baseline_per_image_sha256"]
        != family["audit"]["baseline_per_image_sha256"]
    ):
        raise ValueError("Matched arms use different accepted baselines")

    expected_matched = validate_sha256(
        expected_matched_comparison_sha256, name="matched comparison SHA-256"
    )
    if sha256_file(matched_comparison_path) != expected_matched:
        raise ValueError("Matched comparison SHA-256 mismatch")
    matched = _json(matched_comparison_path)
    _require_safety(matched, name="matched comparison")
    if (
        matched.get("comparison") != "family_balanced minus standard"
        or matched.get("candidate_per_image_sha256") != family["per_image_sha256"]
        or matched.get("reference_per_image_sha256") != standard["per_image_sha256"]
        or matched.get("replicates") != 10000
        or matched.get("seed_family") != 20261101
        or matched.get("ground_truth_reopened") is not False
    ):
        raise ValueError("Matched comparison provenance mismatch")

    regret_improvements: dict[str, float] = {}
    for subgroup in SUBGROUPS:
        standard_row = standard["summary"]["subgroups"][subgroup]
        family_row = family["summary"]["subgroups"][subgroup]
        regret_improvements[subgroup] = _finite(
            standard_row["selected_to_oracle_regret"], name="standard regret"
        ) - _finite(family_row["selected_to_oracle_regret"], name="family regret")
        metric = matched["metrics"][subgroup]
        if (
            abs(
                _finite(metric["candidate_dice"], name="matched family Dice")
                - _finite(family_row["dice"], name="family Dice")
            )
            > 1.0e-12
            or abs(
                _finite(metric["reference_dice"], name="matched standard Dice")
                - _finite(standard_row["dice"], name="standard Dice")
            )
            > 1.0e-12
        ):
            raise ValueError(f"Matched comparison Dice differs: {subgroup}")
    improved = [
        subgroup for subgroup in TUMOR_SUBGROUPS if regret_improvements[subgroup] > 0.0
    ]
    overall_matched = matched["metrics"]["overall"]
    standard_association = standard["summary"]["subgroups"]["overall"][
        "candidate_count_vs_miss_spearman"
    ]
    family_association = family["summary"]["subgroups"]["overall"][
        "candidate_count_vs_miss_spearman"
    ]
    standard_magnitude = 0.0 if standard_association is None else abs(float(standard_association))
    family_magnitude = 0.0 if family_association is None else abs(float(family_association))
    causal_checks = {
        "regret_reduced_in_at_least_two_tumor_subgroups": {
            "improvements": regret_improvements,
            "observed_subgroups": improved,
            "minimum_count": 2,
            "pass": len(improved) >= 2,
        },
        "overall_selected_dice_no_regression": {
            "delta_family_minus_standard": overall_matched[
                "delta_candidate_minus_reference"
            ],
            "pass": float(overall_matched["delta_candidate_minus_reference"]) >= 0.0,
        },
        "absolute_count_miss_association_no_increase": {
            "family_balanced": family_magnitude,
            "standard": standard_magnitude,
            "pass": family_magnitude <= standard_magnitude + 1.0e-12,
        },
    }
    causal_pass = all(bool(check["pass"]) for check in causal_checks.values())
    family_operational_pass = family["gate"]["status"] == "OPERATIONAL_PASS"
    return {
        "decision_id": "mask_bag_family_balanced_s1_matched_decision_v1",
        "status": (
            "OPERATIONAL_PASS"
            if causal_pass and family_operational_pass
            else "CAUSAL_PROMOTION_PASS"
            if causal_pass
            else "FAIL"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "pair_output_audit_sha256": expected_pair_audit,
        "standard_evaluation_audit_sha256": expected_standard_audit_sha256,
        "family_balanced_evaluation_audit_sha256": expected_family_audit_sha256,
        "matched_comparison_sha256": expected_matched,
        "causal_checks": causal_checks,
        "matched_overall_ci95": overall_matched["ci95"],
        "standard_gate_status": standard["gate"]["status"],
        "family_balanced_gate_status": family["gate"]["status"],
        "family_balanced_operational_pass": family_operational_pass,
        "consumer_authorized": causal_pass and family_operational_pass,
        "result_adoption_allowed": causal_pass and family_operational_pass,
        "ground_truth_reopened": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--pair-output-audit", type=Path, required=True)
    parser.add_argument("--expected-pair-output-audit-sha256", required=True)
    parser.add_argument("--standard-evaluation-root", type=Path, required=True)
    parser.add_argument("--expected-standard-evaluation-audit-sha256", required=True)
    parser.add_argument("--family-evaluation-root", type=Path, required=True)
    parser.add_argument("--expected-family-evaluation-audit-sha256", required=True)
    parser.add_argument("--matched-comparison", type=Path, required=True)
    parser.add_argument("--expected-matched-comparison-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("S1 decision output already exists")
    result = decide_s1_pair(
        args.pair_root.resolve(),
        args.pair_output_audit.resolve(),
        args.expected_pair_output_audit_sha256,
        args.standard_evaluation_root.resolve(),
        args.expected_standard_evaluation_audit_sha256,
        args.family_evaluation_root.resolve(),
        args.expected_family_evaluation_audit_sha256,
        args.matched_comparison.resolve(),
        args.expected_matched_comparison_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
