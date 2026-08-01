from __future__ import annotations

"""Apply the exact S4 gate after the generic frozen-arm evaluator."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROTOCOL_SHA256 = "fb39234a03890d7201531066e3ca7a11f2379eaa120bd503fe4b92e6de30a2a6"
SOURCE_COMMIT = "95c4a3378eaf8463c57d57a0dd4e4cac6c69021f"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
OUTPUT_AUDIT_ID = "independent_mask_bag_proposal_cluster_s4_output_v1"
OUTPUT_AUDIT_PASS = "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS"
EVALUATOR_GATE_ID = "mask_bag_selector_arm_gate_v1"
REQUIRED_EVALUATION_FILES = (
    "gate_decision.json",
    "paired_comparison.json",
    "per_image.csv",
    "summary.json",
)
SUBGROUPS = ("overall", "small", "medium", "large")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _require_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_safety(payload: dict[str, Any], *, name: str) -> None:
    if payload.get("consumer_trained") is not False or payload.get("test_evaluated") is not False:
        raise ValueError(f"{name} safety lock mismatch")


def decide(
    output_audit_path: Path,
    evaluation_root: Path,
    evaluation_audit_path: Path,
    protocol_path: Path,
    *,
    expected_output_audit_sha256: str,
    expected_evaluation_audit_sha256: str,
) -> dict[str, Any]:
    _require_sha(expected_output_audit_sha256, name="output audit SHA-256")
    _require_sha(expected_evaluation_audit_sha256, name="evaluation audit SHA-256")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("S4 protocol SHA-256 mismatch")
    if sha256_file(output_audit_path) != expected_output_audit_sha256:
        raise ValueError("S4 independent output-audit SHA-256 mismatch")
    if sha256_file(evaluation_audit_path) != expected_evaluation_audit_sha256:
        raise ValueError("S4 evaluation-audit SHA-256 mismatch")

    output_audit = _json(output_audit_path)
    evaluation_audit = _json(evaluation_audit_path)
    if (
        output_audit.get("audit_id") != OUTPUT_AUDIT_ID
        or output_audit.get("status") != OUTPUT_AUDIT_PASS
        or output_audit.get("source_commit") != SOURCE_COMMIT
        or output_audit.get("protocol_sha256") != PROTOCOL_SHA256
        or output_audit.get("split_sha256") != SPLIT_SHA256
        or output_audit.get("cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or output_audit.get("training_labels") != "image_level_only"
        or output_audit.get("validation_gt_read") is not False
    ):
        raise ValueError("S4 independent output audit did not pass its GT-blind gate")
    _require_safety(output_audit, name="output audit")

    cohort = {
        "validation": 371,
        "tumor": 184,
        "normal": 187,
        "small": 94,
        "medium": 72,
        "large": 18,
    }
    if (
        evaluation_audit.get(
            "validation_gt_read_only_after_all_predictions_frozen_and_verified"
        )
        is not True
        or evaluation_audit.get("split_sha256") != SPLIT_SHA256
        or evaluation_audit.get("selector_cache_freeze_sha256")
        != CACHE_FREEZE_SHA256
        or evaluation_audit.get("arm_prediction_freeze_sha256")
        != output_audit.get("prediction_freeze_sha256")
        or evaluation_audit.get("bootstrap_replicates") != 10000
        or evaluation_audit.get("cohort") != cohort
        or evaluation_audit.get("complete_misses_included") is not True
    ):
        raise ValueError("S4 evaluation boundary/cohort contract mismatch")
    _require_safety(evaluation_audit, name="evaluation audit")
    output_hashes = evaluation_audit.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(
        REQUIRED_EVALUATION_FILES
    ):
        raise ValueError("S4 evaluation output inventory mismatch")
    for name in REQUIRED_EVALUATION_FILES:
        path = evaluation_root / name
        if not path.is_file() or sha256_file(path) != output_hashes[name]:
            raise ValueError(f"S4 evaluation output hash mismatch: {name}")

    gate = _json(evaluation_root / "gate_decision.json")
    paired = _json(evaluation_root / "paired_comparison.json")
    summary = _json(evaluation_root / "summary.json")
    for name, payload in (("gate", gate), ("paired", paired), ("summary", summary)):
        _require_safety(payload, name=name)
    if (
        gate.get("gate_id") != EVALUATOR_GATE_ID
        or paired.get("replicates") != 10000
        or paired.get("seed_family") != 20261013
        or summary.get("arm_source_commit") != SOURCE_COMMIT
        or summary.get("arm_protocol_sha256") != PROTOCOL_SHA256
        or summary.get("cohort") != cohort
        or summary.get(
            "validation_gt_read_only_after_all_predictions_frozen_and_verified"
        )
        is not True
    ):
        raise ValueError("S4 evaluator source/protocol/bootstrap contract mismatch")

    mechanism = gate.get("mechanism_checks")
    operational = gate.get("operational_goal_checks")
    oracle = gate.get("candidate_oracle_goal_checks")
    safety = gate.get("final_safety_checks")
    if (
        not isinstance(mechanism, dict)
        or set(mechanism)
        != {
            "regret_reduced_in_at_least_two_tumor_subgroups",
            "overall_selected_dice_no_regression",
            "absolute_count_miss_association_no_increase",
        }
        or not isinstance(operational, dict)
        or set(operational) != set(SUBGROUPS)
        or not isinstance(oracle, dict)
        or set(oracle) != set(SUBGROUPS)
        or not isinstance(safety, dict)
    ):
        raise ValueError("S4 gate schema mismatch")
    mechanism_pass = all(
        isinstance(check, dict) and check.get("pass") is True
        for check in mechanism.values()
    )
    full_adoption_pass = (
        mechanism_pass
        and all(
            isinstance(check, dict) and check.get("pass") is True
            for check in operational.values()
        )
        and all(
            isinstance(check, dict) and check.get("pass") is True
            for check in oracle.values()
        )
        and all(value is True for value in safety.values())
    )
    expected_status = (
        "OPERATIONAL_PASS"
        if full_adoption_pass
        else "MECHANISM_PASS"
        if mechanism_pass
        else "FAIL"
    )
    if (
        gate.get("status") != expected_status
        or gate.get("consumer_authorized") is not full_adoption_pass
    ):
        raise ValueError("S4 generic gate status/consumer authorization is inconsistent")
    return {
        "decision_id": "mask_bag_proposal_cluster_s4_decision_v1",
        "status": expected_status,
        "protocol_sha256": PROTOCOL_SHA256,
        "independent_output_audit_sha256": expected_output_audit_sha256,
        "evaluation_audit_sha256": expected_evaluation_audit_sha256,
        "evaluation_output_hashes": output_hashes,
        "mechanism_checks": mechanism,
        "operational_goal_checks": operational,
        "candidate_oracle_goal_checks": oracle,
        "final_safety_checks": safety,
        "mechanism_pass": mechanism_pass,
        "full_adoption_pass": full_adoption_pass,
        "consumer_authorized": full_adoption_pass,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-output-audit-sha256", required=True)
    parser.add_argument("--expected-evaluation-audit-sha256", required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    args = parser.parse_args()
    if args.decision_output.exists():
        raise FileExistsError("S4 decision output already exists")
    result = decide(
        args.output_audit.resolve(),
        args.evaluation_root.resolve(),
        args.evaluation_audit.resolve(),
        args.protocol.resolve(),
        expected_output_audit_sha256=args.expected_output_audit_sha256,
        expected_evaluation_audit_sha256=args.expected_evaluation_audit_sha256,
    )
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
