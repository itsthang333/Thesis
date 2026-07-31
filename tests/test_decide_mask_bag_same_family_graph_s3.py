from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from decide_mask_bag_same_family_graph_s3 import decide


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1.json"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, medium: float, generic_pass: bool = True) -> tuple[Path, Path, Path]:
    output_audit = tmp_path / "output_audit.json"
    _write_json(
        output_audit,
        {
            "audit_id": "independent_mask_bag_same_family_graph_s3_output_v1",
            "status": "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS",
            "protocol_sha256": "7d7636176fc05d407b51a913170ad780e2d43d328d9437b2d9d2656e191471ca",
            "training_labels": "image_level_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    evaluation = tmp_path / "evaluation"
    common_safety = {"consumer_trained": False, "test_evaluated": False}
    gate = {
        "gate_id": "mask_bag_selector_arm_gate_v1",
        "status": "OPERATIONAL_PASS" if generic_pass else "FAIL",
        "mechanism_checks": {
            "regret_reduced_in_at_least_two_tumor_subgroups": {
                "pass": generic_pass,
                "improvements": {"small": 0.1, "medium": medium, "large": 0.1},
            },
            "overall_selected_dice_no_regression": {"pass": generic_pass},
            "absolute_count_miss_association_no_increase": {"pass": generic_pass},
        },
        "operational_goal_checks": {name: {"pass": True} for name in ("overall", "small", "medium", "large")},
        "candidate_oracle_goal_checks": {name: {"pass": True} for name in ("overall", "small", "medium", "large")},
        "final_safety_checks": {"ci": True, "subgroup": True, "miss": True, "auroc": True},
        "consumer_authorized": generic_pass,
        **common_safety,
    }
    _write_json(evaluation / "gate_decision.json", gate)
    _write_json(evaluation / "paired_comparison.json", {"replicates": 10000, "seed_family": 20261012, **common_safety})
    _write_json(evaluation / "summary.json", common_safety)
    (evaluation / "per_image.csv").write_text("image_id\nIMG1\n", encoding="utf-8")
    output_hashes = {name: _sha(evaluation / name) for name in ("gate_decision.json", "paired_comparison.json", "per_image.csv", "summary.json")}
    evaluation_audit = evaluation / "evaluation_audit.json"
    _write_json(
        evaluation_audit,
        {
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            "bootstrap_replicates": 10000,
            "cohort": {"validation": 371, "tumor": 184, "normal": 187, "small": 94, "medium": 72, "large": 18},
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


def test_s3_specific_medium_gate_blocks_generic_operational_pass(tmp_path: Path) -> None:
    result = _decide(_fixture(tmp_path, medium=0.0))
    assert result["generic_evaluator_status"] == "OPERATIONAL_PASS"
    assert result["status"] == "FAIL"
    assert result["consumer_authorized"] is False


def test_s3_operational_pass_requires_generic_and_medium_gates(tmp_path: Path) -> None:
    result = _decide(_fixture(tmp_path, medium=0.01))
    assert result["status"] == "OPERATIONAL_PASS"
    assert result["consumer_authorized"] is True


def test_s3_decision_rejects_mutated_evaluation_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, medium=0.01)
    (paths[1] / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        _decide(paths)

