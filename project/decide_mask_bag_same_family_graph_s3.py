from __future__ import annotations

"""Apply the exact S3-only gate after the generic frozen-arm evaluator."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROTOCOL_SHA256 = "7d7636176fc05d407b51a913170ad780e2d43d328d9437b2d9d2656e191471ca"
OUTPUT_AUDIT_ID = "independent_mask_bag_same_family_graph_s3_output_v1"
OUTPUT_AUDIT_PASS = "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS"
EVALUATOR_GATE_ID = "mask_bag_selector_arm_gate_v1"
REQUIRED_EVALUATION_FILES = (
    "gate_decision.json",
    "paired_comparison.json",
    "per_image.csv",
    "summary.json",
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _safety_false(payload: dict[str, Any], *, name: str) -> None:
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
        raise ValueError("S3 protocol hash mismatch")
    if sha256_file(output_audit_path) != expected_output_audit_sha256:
        raise ValueError("S3 independent output-audit hash mismatch")
    if sha256_file(evaluation_audit_path) != expected_evaluation_audit_sha256:
        raise ValueError("S3 evaluation-audit hash mismatch")

    output_audit = _json(output_audit_path)
    evaluation_audit = _json(evaluation_audit_path)
    if (
        output_audit.get("audit_id") != OUTPUT_AUDIT_ID
        or output_audit.get("status") != OUTPUT_AUDIT_PASS
        or output_audit.get("protocol_sha256") != PROTOCOL_SHA256
        or output_audit.get("validation_gt_read") is not False
        or output_audit.get("training_labels") != "image_level_only"
    ):
        raise ValueError("S3 independent output audit did not pass the GT-blind gate")
    _safety_false(output_audit, name="output audit")

    if (
        evaluation_audit.get("validation_gt_read_only_after_all_predictions_frozen_and_verified")
        is not True
        or evaluation_audit.get("bootstrap_replicates") != 10000
        or evaluation_audit.get("cohort")
        != {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            "small": 94,
            "medium": 72,
            "large": 18,
        }
    ):
        raise ValueError("S3 evaluation boundary/cohort contract mismatch")
    _safety_false(evaluation_audit, name="evaluation audit")
    output_hashes = evaluation_audit.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(REQUIRED_EVALUATION_FILES):
        raise ValueError("S3 evaluation output inventory mismatch")
    for name in REQUIRED_EVALUATION_FILES:
        path = evaluation_root / name
        if not path.is_file() or sha256_file(path) != output_hashes[name]:
            raise ValueError(f"S3 evaluation output hash mismatch: {name}")

    gate = _json(evaluation_root / "gate_decision.json")
    paired = _json(evaluation_root / "paired_comparison.json")
    summary = _json(evaluation_root / "summary.json")
    for payload_name, payload in (("gate", gate), ("paired", paired), ("summary", summary)):
        _safety_false(payload, name=payload_name)
    if gate.get("gate_id") != EVALUATOR_GATE_ID:
        raise ValueError("S3 generic evaluator gate id mismatch")
    if paired.get("replicates") != 10000 or paired.get("seed_family") != 20261012:
        raise ValueError("S3 paired bootstrap contract mismatch")

    mechanism_checks = gate.get("mechanism_checks")
    if not isinstance(mechanism_checks, dict):
        raise ValueError("S3 generic mechanism checks missing")
    regret_check = mechanism_checks.get("regret_reduced_in_at_least_two_tumor_subgroups")
    if not isinstance(regret_check, dict) or not isinstance(regret_check.get("improvements"), dict):
        raise ValueError("S3 regret-improvement evidence missing")
    medium_improvement = float(regret_check["improvements"]["medium"])
    medium_regret_reduced = medium_improvement > 0.0
    evaluator_mechanism_pass = all(
        isinstance(check, dict) and check.get("pass") is True
        for check in mechanism_checks.values()
    )
    s3_mechanism_pass = evaluator_mechanism_pass and medium_regret_reduced

    operational_checks = gate.get("operational_goal_checks")
    oracle_checks = gate.get("candidate_oracle_goal_checks")
    final_safety_checks = gate.get("final_safety_checks")
    if not all(isinstance(value, dict) for value in (operational_checks, oracle_checks)):
        raise ValueError("S3 operational/oracle checks missing")
    if not isinstance(final_safety_checks, dict):
        raise ValueError("S3 final safety checks missing")
    full_adoption_pass = (
        s3_mechanism_pass
        and all(check.get("pass") is True for check in operational_checks.values())
        and all(check.get("pass") is True for check in oracle_checks.values())
        and all(value is True for value in final_safety_checks.values())
        and gate.get("consumer_authorized") is True
    )
    status = (
        "OPERATIONAL_PASS"
        if full_adoption_pass
        else "MECHANISM_PASS"
        if s3_mechanism_pass
        else "FAIL"
    )
    return {
        "decision_id": "mask_bag_same_family_graph_s3_decision_v1",
        "status": status,
        "protocol_sha256": PROTOCOL_SHA256,
        "independent_output_audit_sha256": expected_output_audit_sha256,
        "evaluation_audit_sha256": expected_evaluation_audit_sha256,
        "evaluation_output_hashes": output_hashes,
        "generic_evaluator_status": gate.get("status"),
        "s3_mechanism_checks": {
            "generic_mechanism_gate_pass": evaluator_mechanism_pass,
            "medium_selected_to_oracle_regret_reduced": {
                "improvement": medium_improvement,
                "minimum_exclusive": 0.0,
                "pass": medium_regret_reduced,
            },
        },
        "s3_mechanism_pass": s3_mechanism_pass,
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
        raise FileExistsError("S3 decision output already exists")
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
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

