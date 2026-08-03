"""Freeze terminal S9 hashes after independent audit and before validation GT."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
PROTOCOL_SHA256 = "0a303c9c86c3c43c750c85a50087e792bf0942a0b43fc9a1cf9e143c4832ee3d"
SOURCE_COMMIT = "7dcd6c6f055c69f3f048a005ed2fea6177dc7ed8"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_s9_likelihood_equal_rank"
ARMS = (CONTROL_ARM, PRIMARY_ARM)
PRE_GT_STATUS = "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_REPRODUCTION_PASS"
READINESS_STATUS = "FROZEN_AFTER_INDEPENDENT_GT_BLIND_AUDIT_BEFORE_VALIDATION_GT"


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


def _require_safety(payload: Mapping[str, Any], *, name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"S9 {name} safety boundary mismatch")


def freeze_readiness(
    output_root: Path,
    pre_gt_audit_path: Path,
    protocol_path: Path,
    evaluation_addendum_path: Path,
    readiness_output: Path,
    *,
    expected_pre_gt_audit_sha256: str,
    expected_evaluation_addendum_sha256: str,
) -> dict[str, Any]:
    if readiness_output.exists():
        raise FileExistsError("S9 postfreeze readiness output already exists")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("S9 protocol SHA-256 mismatch")
    protocol = _json(protocol_path)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("S9 protocol experiment mismatch")
    if sha256_file(evaluation_addendum_path) != expected_evaluation_addendum_sha256:
        raise ValueError("S9 evaluation addendum SHA-256 mismatch")
    addendum = _json(evaluation_addendum_path)
    _require_safety(addendum, name="evaluation addendum")
    if (
        addendum.get("status") != "FROZEN_BEFORE_VALIDATION_GT"
        or addendum.get("experiment_id") != EXPERIMENT_ID
        or addendum.get("scientific_protocol_sha256") != PROTOCOL_SHA256
        or addendum.get("arm_order") != list(ARMS)
        or addendum.get("bootstrap_replicates") != 10_000
        or addendum.get("bootstrap_seed") != 20261205
    ):
        raise ValueError("S9 evaluation addendum contract mismatch")
    if sha256_file(pre_gt_audit_path) != expected_pre_gt_audit_sha256:
        raise ValueError("S9 independent pre-GT audit SHA-256 mismatch")
    pre_gt = _json(pre_gt_audit_path)
    _require_safety(pre_gt, name="independent pre-GT audit")
    if (
        pre_gt.get("status") != PRE_GT_STATUS
        or pre_gt.get("protocol_sha256") != PROTOCOL_SHA256
        or pre_gt.get("source_commit") != SOURCE_COMMIT
        or pre_gt.get("validation_predictions_per_arm") != 371
        or pre_gt.get("feature_hashes_verified") != 3352
        or pre_gt.get("physical_likelihood_evidence_verified") != 371
        or pre_gt.get("physical_prediction_maps_verified") != 742
        or pre_gt.get("physical_candidate_scores_verified") != 742
        or pre_gt.get("training_reexecuted") is not False
        or pre_gt.get("checkpoint_selection")
        != "final_epoch_only_no_validation_selection"
    ):
        raise ValueError("S9 independent pre-GT audit contract mismatch")

    pair_path = output_root / "prediction_pair_freeze.json"
    pair_sha = sha256_file(pair_path)
    pair = _json(pair_path)
    _require_safety(pair, name="prediction pair freeze")
    if (
        pair_sha != pre_gt.get("pair_freeze_sha256")
        or pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("source_commit") != SOURCE_COMMIT
        or pair.get("protocol_sha256") != PROTOCOL_SHA256
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S9 prediction pair freeze mismatch")

    arm_contracts: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        arm_root = output_root / arm
        freeze_path = arm_root / "prediction_freeze.json"
        freeze_sha = sha256_file(freeze_path)
        freeze = _json(freeze_path)
        _require_safety(freeze, name=f"{arm} freeze")
        score_manifest = arm_root / "candidate_scores/candidate_score_manifest.csv"
        prediction_manifest = arm_root / "predictions/prediction_manifest.csv"
        if (
            pair["arms"].get(arm) != freeze_sha
            or freeze.get("experiment_id") != EXPERIMENT_ID
            or freeze.get("arm") != arm
            or freeze.get("source_commit") != SOURCE_COMMIT
            or freeze.get("protocol_sha256") != PROTOCOL_SHA256
            or freeze.get("split_sha256") != SPLIT_SHA256
            or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
            or freeze.get("baseline_prediction_freeze_sha256")
            != BASELINE_FREEZE_SHA256
            or freeze.get("validation_predictions") != 371
            or freeze.get("training_labels") != "image_level_normal_tumor_only"
            or sha256_file(score_manifest)
            != freeze.get("candidate_score_manifest_sha256")
            or sha256_file(prediction_manifest)
            != freeze.get("prediction_manifest_sha256")
        ):
            raise ValueError(f"S9 {arm} dynamic hash contract mismatch")
        for key in (
            "s9_checkpoint_sha256",
            "training_history_sha256",
            "feature_cache_operational_gate_sha256",
            "feature_cache_manifest_sha256",
            "s9_likelihood_evidence_manifest_sha256",
        ):
            value = freeze.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"S9 {arm} lacks frozen {key}")
        arm_contracts[arm] = {
            "prediction_freeze_sha256": freeze_sha,
            "prediction_manifest_sha256": str(freeze["prediction_manifest_sha256"]),
            "candidate_score_manifest_sha256": str(
                freeze["candidate_score_manifest_sha256"]
            ),
        }

    result = {
        "schema_version": 1,
        "status": READINESS_STATUS,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "evaluation_addendum_sha256": expected_evaluation_addendum_sha256,
        "terminal_pre_gt_audit_sha256": expected_pre_gt_audit_sha256,
        "prediction_pair_freeze_sha256": pair_sha,
        "arms": arm_contracts,
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 20261205,
        "evaluation_authorized_after_commit_and_central_visibility": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "collaborator_output_accessed": False,
    }
    readiness_output.parent.mkdir(parents=True, exist_ok=True)
    with readiness_output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if _json(readiness_output) != result:
        raise RuntimeError("S9 postfreeze readiness write/read mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pre-gt-audit", type=Path, required=True)
    parser.add_argument("--expected-pre-gt-audit-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation-addendum", type=Path, required=True)
    parser.add_argument("--expected-evaluation-addendum-sha256", required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_readiness(
        args.output_root.resolve(),
        args.pre_gt_audit.resolve(),
        args.protocol.resolve(),
        args.evaluation_addendum.resolve(),
        args.readiness_output.resolve(),
        expected_pre_gt_audit_sha256=args.expected_pre_gt_audit_sha256,
        expected_evaluation_addendum_sha256=args.expected_evaluation_addendum_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

