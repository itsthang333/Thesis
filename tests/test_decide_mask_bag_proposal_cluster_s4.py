from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from decide_mask_bag_proposal_cluster_s4 import decide


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "artifacts"
    / "research_protocols"
    / "rad_dino_mask_bag_proposal_cluster_s4_v1.json"
)
PROTOCOL_SHA256 = "fb39234a03890d7201531066e3ca7a11f2379eaa120bd503fe4b92e6de30a2a6"
SOURCE_COMMIT = "95c4a3378eaf8463c57d57a0dd4e4cac6c69021f"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
COHORT = {
    "validation": 371,
    "tumor": 184,
    "normal": 187,
    "small": 94,
    "medium": 72,
    "large": 18,
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    mechanism_pass: bool,
    operational_pass: bool,
) -> tuple[Path, Path, Path]:
    output_audit = tmp_path / "output_audit.json"
    _write_json(
        output_audit,
        {
            "audit_id": "independent_mask_bag_proposal_cluster_s4_output_v1",
            "status": "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS",
            "source_commit": SOURCE_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "split_sha256": SPLIT_SHA256,
            "cache_freeze_sha256": CACHE_FREEZE_SHA256,
            "prediction_freeze_sha256": "a" * 64,
            "training_labels": "image_level_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    evaluation = tmp_path / "evaluation"
    common_safety = {"consumer_trained": False, "test_evaluated": False}
    mechanism = {
        "regret_reduced_in_at_least_two_tumor_subgroups": {
            "pass": mechanism_pass,
            "improvements": {"small": 0.1, "medium": 0.1, "large": 0.0},
        },
        "overall_selected_dice_no_regression": {"pass": mechanism_pass},
        "absolute_count_miss_association_no_increase": {"pass": mechanism_pass},
    }
    goals = {
        name: {"pass": operational_pass}
        for name in ("overall", "small", "medium", "large")
    }
    safety = {
        "overall_ci95_low_above_zero": operational_pass,
        "no_tumor_subgroup_mean_decrease": operational_pass,
        "no_complete_miss_increase": operational_pass,
        "image_auroc_at_least_0_75": operational_pass,
    }
    full = mechanism_pass and operational_pass
    status = "OPERATIONAL_PASS" if full else "MECHANISM_PASS" if mechanism_pass else "FAIL"
    _write_json(
        evaluation / "gate_decision.json",
        {
            "gate_id": "mask_bag_selector_arm_gate_v1",
            "status": status,
            "mechanism_checks": mechanism,
            "operational_goal_checks": goals,
            "candidate_oracle_goal_checks": goals,
            "final_safety_checks": safety,
            "consumer_authorized": full,
            **common_safety,
        },
    )
    _write_json(
        evaluation / "paired_comparison.json",
        {"replicates": 10000, "seed_family": 20261013, **common_safety},
    )
    _write_json(
        evaluation / "summary.json",
        {
            "arm_source_commit": SOURCE_COMMIT,
            "arm_protocol_sha256": PROTOCOL_SHA256,
            "cohort": COHORT,
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            **common_safety,
        },
    )
    (evaluation / "per_image.csv").write_text("image_id\nIMG1\n", encoding="utf-8")
    output_hashes = {
        name: _sha(evaluation / name)
        for name in (
            "gate_decision.json",
            "paired_comparison.json",
            "per_image.csv",
            "summary.json",
        )
    }
    evaluation_audit = evaluation / "evaluation_audit.json"
    _write_json(
        evaluation_audit,
        {
            "split_sha256": SPLIT_SHA256,
            "selector_cache_freeze_sha256": CACHE_FREEZE_SHA256,
            "arm_prediction_freeze_sha256": "a" * 64,
            "bootstrap_replicates": 10000,
            "cohort": COHORT,
            "complete_misses_included": True,
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            "output_hashes": output_hashes,
            **common_safety,
        },
    )
    return output_audit, evaluation, evaluation_audit


def _decide(paths: tuple[Path, Path, Path]) -> dict[str, object]:
    output_audit, evaluation, evaluation_audit = paths
    return decide(
        output_audit,
        evaluation,
        evaluation_audit,
        PROTOCOL,
        expected_output_audit_sha256=_sha(output_audit),
        expected_evaluation_audit_sha256=_sha(evaluation_audit),
    )


def test_s4_decision_preserves_mechanism_only_consumer_lock(tmp_path: Path) -> None:
    result = _decide(
        _fixture(tmp_path, mechanism_pass=True, operational_pass=False)
    )
    assert result["status"] == "MECHANISM_PASS"
    assert result["consumer_authorized"] is False


def test_s4_decision_authorizes_consumer_only_after_every_gate(tmp_path: Path) -> None:
    result = _decide(
        _fixture(tmp_path, mechanism_pass=True, operational_pass=True)
    )
    assert result["status"] == "OPERATIONAL_PASS"
    assert result["consumer_authorized"] is True


def test_s4_decision_rejects_inconsistent_generic_authorization(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, mechanism_pass=True, operational_pass=False)
    gate_path = paths[1] / "gate_decision.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["consumer_authorized"] = True
    _write_json(gate_path, gate)
    audit = json.loads(paths[2].read_text(encoding="utf-8"))
    audit["output_hashes"]["gate_decision.json"] = _sha(gate_path)
    _write_json(paths[2], audit)
    with pytest.raises(ValueError, match="inconsistent"):
        _decide(paths)


def test_s4_decision_rejects_mutated_evaluation_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, mechanism_pass=True, operational_pass=True)
    (paths[1] / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        _decide(paths)
