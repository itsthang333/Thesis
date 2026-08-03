from __future__ import annotations

import ast
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from decide_skelex_candidate_marginal_s9 import (  # noqa: E402
    ARMS,
    BASELINE_FREEZE_SHA256,
    BASELINE_PER_IMAGE_SHA256,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CACHE_FREEZE_SHA256,
    COHORT,
    CONTROL_ARM,
    EXPERIMENT_ID,
    PRE_GT_STATUS,
    PRIMARY_ARM,
    PROTOCOL_SHA256,
    READINESS_STATUS,
    SOURCE_COMMIT,
    SPLIT_SHA256,
    decide,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/research_protocols/skelex_candidate_marginal_s9_v1.json"
DECISION_SOURCE = ROOT / "project/decide_skelex_candidate_marginal_s9.py"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_s9_decision_cannot_reopen_segmentation_data() -> None:
    source = DECISION_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "datasets.btxrd" not in imported
    assert "PIL" not in imported
    assert "Annotations" not in source
    assert 'BOOTSTRAP_SEED = 20261205' in source
    assert '"overall": 0.34024039' in source
    assert '"ground_truth_reopened": False' in source
    assert '"post_hoc_rescue_or_sweep_authorized": False' in source


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _rows(dice: float, misses: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for subgroup, count, area in (
        ("small", 94, 0.005),
        ("medium", 72, 0.02),
        ("large", 18, 0.08),
    ):
        for _ in range(count):
            rows.append(
                {
                    "image_id": f"IMG{index:06d}.jpeg",
                    "group_id": f"group-{index // 2:04d}",
                    "gt_area_ratio": area,
                    "size_group": subgroup,
                    "dice": dice,
                    "oracle_best_single_dice": 0.7,
                    "complete_miss": int(index < misses),
                    "baseline_dice": 0.2,
                    "baseline_oracle_best_single_dice": 0.7,
                    "baseline_complete_miss": int(index < 10),
                }
            )
            index += 1
    return rows


def _evaluation(
    root: Path,
    *,
    dice: float,
    misses: int,
    arm_freeze_sha: str,
    score_manifest_sha: str,
) -> tuple[Path, str]:
    root.mkdir()
    per_image = root / "per_image.csv"
    rows = _rows(dice, misses)
    with per_image.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payloads = {
        "summary.json": {
            "arm_source_commit": SOURCE_COMMIT,
            "arm_protocol_sha256": PROTOCOL_SHA256,
            "cohort": COHORT,
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        "paired_comparison.json": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_family": BOOTSTRAP_SEED,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        "gate_decision.json": {
            "consumer_trained": False,
            "test_evaluated": False,
        },
    }
    for name, payload in payloads.items():
        _write_json(root / name, payload)
    output_hashes = {
        name: _sha(root / name)
        for name in (
            "per_image.csv",
            "summary.json",
            "paired_comparison.json",
            "gate_decision.json",
        )
    }
    audit = root / "evaluation_audit.json"
    _write_json(
        audit,
        {
            "split_sha256": SPLIT_SHA256,
            "selector_cache_freeze_sha256": CACHE_FREEZE_SHA256,
            "arm_prediction_freeze_sha256": arm_freeze_sha,
            "candidate_score_manifest_sha256": score_manifest_sha,
            "baseline_prediction_freeze_sha256": BASELINE_FREEZE_SHA256,
            "baseline_per_image_sha256": BASELINE_PER_IMAGE_SHA256,
            "cohort": COHORT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            "output_hashes": output_hashes,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    return audit, _sha(audit)


def _fixture(tmp_path: Path) -> dict[str, object]:
    pre_gt = tmp_path / "pre_gt.json"
    _write_json(
        pre_gt,
        {
            "status": PRE_GT_STATUS,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    addendum = tmp_path / "addendum.json"
    _write_json(
        addendum,
        {
            "experiment_id": EXPERIMENT_ID,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "arm_order": list(ARMS),
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    arm_contracts = {
        CONTROL_ARM: {
            "prediction_freeze_sha256": "1" * 64,
            "candidate_score_manifest_sha256": "2" * 64,
        },
        PRIMARY_ARM: {
            "prediction_freeze_sha256": "3" * 64,
            "candidate_score_manifest_sha256": "4" * 64,
        },
    }
    readiness = tmp_path / "readiness.json"
    _write_json(
        readiness,
        {
            "status": READINESS_STATUS,
            "experiment_id": EXPERIMENT_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "evaluation_addendum_sha256": _sha(addendum),
            "terminal_pre_gt_audit_sha256": _sha(pre_gt),
            "prediction_pair_freeze_sha256": "5" * 64,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "arms": arm_contracts,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    control_root = tmp_path / "control"
    control_audit, control_audit_sha = _evaluation(
        control_root,
        dice=0.2,
        misses=10,
        arm_freeze_sha=arm_contracts[CONTROL_ARM]["prediction_freeze_sha256"],
        score_manifest_sha=arm_contracts[CONTROL_ARM][
            "candidate_score_manifest_sha256"
        ],
    )
    primary_root = tmp_path / "primary"
    primary_audit, primary_audit_sha = _evaluation(
        primary_root,
        dice=0.6,
        misses=2,
        arm_freeze_sha=arm_contracts[PRIMARY_ARM]["prediction_freeze_sha256"],
        score_manifest_sha=arm_contracts[PRIMARY_ARM][
            "candidate_score_manifest_sha256"
        ],
    )
    return {
        "pre_gt": pre_gt,
        "addendum": addendum,
        "readiness": readiness,
        "control_root": control_root,
        "control_audit": control_audit,
        "control_audit_sha": control_audit_sha,
        "primary_root": primary_root,
        "primary_audit": primary_audit,
        "primary_audit_sha": primary_audit_sha,
    }


def _decide(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return decide(
        fixture["pre_gt"],
        PROTOCOL,
        fixture["addendum"],
        fixture["readiness"],
        fixture["control_root"],
        fixture["control_audit"],
        fixture["primary_root"],
        fixture["primary_audit"],
        output,
        expected_pre_gt_audit_sha256=_sha(fixture["pre_gt"]),
        expected_evaluation_addendum_sha256=_sha(fixture["addendum"]),
        expected_readiness_sha256=_sha(fixture["readiness"]),
        expected_control_evaluation_audit_sha256=fixture["control_audit_sha"],
        expected_primary_evaluation_audit_sha256=fixture["primary_audit_sha"],
    )


def test_s9_decision_applies_all_predeclared_operational_gates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _decide(fixture, tmp_path / "decision")
    gate = result["gate"]
    assert gate["status"] == "OPERATIONAL_PASS"
    assert gate["consumer_authorized"] is True
    assert gate["post_hoc_rescue_or_sweep_authorized"] is False
    comparison = result["comparison"]
    assert comparison["seed_family"] == BOOTSTRAP_SEED
    for subgroup in ("overall", "small", "medium", "large"):
        assert comparison["metrics"][subgroup][
            "delta_primary_minus_control"
        ] == pytest.approx(0.4)
    assert comparison["metrics"]["overall"]["misses_recovered"] == 8
    assert comparison["ground_truth_reopened"] is False


def test_s9_decision_rejects_evaluation_hash_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["primary_root"] / "per_image.csv").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="output hash"):
        _decide(fixture, tmp_path / "decision")


def test_s9_decision_rejects_readiness_safety_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    readiness = fixture["readiness"]
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["validation_gt_read"] = True
    _write_json(readiness, payload)
    with pytest.raises(ValueError, match="readiness"):
        decide(
            fixture["pre_gt"],
            PROTOCOL,
            fixture["addendum"],
            readiness,
            fixture["control_root"],
            fixture["control_audit"],
            fixture["primary_root"],
            fixture["primary_audit"],
            tmp_path / "decision",
            expected_pre_gt_audit_sha256=_sha(fixture["pre_gt"]),
            expected_evaluation_addendum_sha256=_sha(fixture["addendum"]),
            expected_readiness_sha256=_sha(readiness),
            expected_control_evaluation_audit_sha256=fixture["control_audit_sha"],
            expected_primary_evaluation_audit_sha256=fixture["primary_audit_sha"],
        )
