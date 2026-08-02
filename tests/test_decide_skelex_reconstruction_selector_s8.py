from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from decide_skelex_reconstruction_selector_s8 import (  # noqa: E402
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
PROTOCOL = ROOT / "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json"
COUNTS = {"small": 94, "medium": 72, "large": 18}
CONTROL_FREEZE = "1" * 64
PRIMARY_FREEZE = "2" * 64
CONTROL_SCORES = "3" * 64
PRIMARY_SCORES = "4" * 64
PAIR_FREEZE = "5" * 64


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_evaluation(
    root: Path,
    *,
    arm: str,
    primary_values: dict[str, float],
    control_values: dict[str, float],
) -> Path:
    root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    image_index = 0
    for subgroup, count in COUNTS.items():
        for _ in range(count):
            image_index += 1
            control = control_values[subgroup]
            dice = control if arm == CONTROL_ARM else primary_values[subgroup]
            rows.append(
                {
                    "image_id": f"IMG{image_index:06d}.jpeg",
                    "group_id": f"group-{image_index:06d}",
                    "gt_area_ratio": 0.005 if subgroup == "small" else 0.02 if subgroup == "medium" else 0.08,
                    "size_group": subgroup,
                    "dice": dice,
                    "oracle_best_single_dice": 0.8,
                    "complete_miss": 0,
                    "baseline_dice": control,
                    "baseline_oracle_best_single_dice": 0.8,
                    "baseline_complete_miss": 0,
                }
            )
    per_image = root / "per_image.csv"
    with per_image.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    safety = {"consumer_trained": False, "test_evaluated": False}
    _write_json(
        root / "summary.json",
        {
            "arm_source_commit": SOURCE_COMMIT,
            "arm_protocol_sha256": PROTOCOL_SHA256,
            "cohort": COHORT,
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            **safety,
        },
    )
    _write_json(
        root / "paired_comparison.json",
        {"replicates": BOOTSTRAP_REPLICATES, "seed_family": BOOTSTRAP_SEED, **safety},
    )
    _write_json(root / "gate_decision.json", {"gate_id": "mask_bag_selector_arm_gate_v1", **safety})
    output_hashes = {
        name: _sha(root / name)
        for name in ("gate_decision.json", "paired_comparison.json", "per_image.csv", "summary.json")
    }
    is_control = arm == CONTROL_ARM
    audit = root / "evaluation_audit.json"
    _write_json(
        audit,
        {
            "split_sha256": SPLIT_SHA256,
            "selector_cache_freeze_sha256": CACHE_FREEZE_SHA256,
            "arm_prediction_freeze_sha256": CONTROL_FREEZE if is_control else PRIMARY_FREEZE,
            "candidate_score_manifest_sha256": CONTROL_SCORES if is_control else PRIMARY_SCORES,
            "baseline_prediction_freeze_sha256": BASELINE_FREEZE_SHA256,
            "baseline_per_image_sha256": BASELINE_PER_IMAGE_SHA256,
            "output_hashes": output_hashes,
            "cohort": COHORT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
            **safety,
        },
    )
    return audit


def _fixture(
    tmp_path: Path,
    *,
    primary_values: dict[str, float] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    control_values = {"small": 0.18, "medium": 0.52, "large": 0.50}
    if primary_values is None:
        primary_values = {"small": 0.20, "medium": 0.54, "large": 0.55}
    addendum = tmp_path / "evaluation_addendum.json"
    _write_json(
        addendum,
        {
            "experiment_id": EXPERIMENT_ID,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "validation_gt_read": False,
        },
    )
    pre_gt = tmp_path / "pre_gt.json"
    _write_json(
        pre_gt,
        {
            "status": PRE_GT_STATUS,
            "experiment_id": EXPERIMENT_ID,
            "pair_freeze_sha256": PAIR_FREEZE,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    readiness = tmp_path / "readiness.json"
    _write_json(
        readiness,
        {
            "status": READINESS_STATUS,
            "experiment_id": EXPERIMENT_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "evaluation_addendum_sha256": _sha(addendum),
            "terminal_pre_gt_audit_sha256": _sha(pre_gt),
            "prediction_pair_freeze_sha256": PAIR_FREEZE,
            "arms": {
                CONTROL_ARM: {
                    "prediction_freeze_sha256": CONTROL_FREEZE,
                    "candidate_score_manifest_sha256": CONTROL_SCORES,
                },
                PRIMARY_ARM: {
                    "prediction_freeze_sha256": PRIMARY_FREEZE,
                    "candidate_score_manifest_sha256": PRIMARY_SCORES,
                },
            },
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    control_root = tmp_path / "control"
    primary_root = tmp_path / "primary"
    control_audit = _write_evaluation(
        control_root,
        arm=CONTROL_ARM,
        primary_values=primary_values,
        control_values=control_values,
    )
    primary_audit = _write_evaluation(
        primary_root,
        arm=PRIMARY_ARM,
        primary_values=primary_values,
        control_values=control_values,
    )
    return addendum, pre_gt, readiness, control_root, control_audit, primary_root, primary_audit


def _decide(paths: tuple[Path, Path, Path, Path, Path, Path, Path], output: Path) -> dict[str, object]:
    addendum, pre_gt, readiness, control_root, control_audit, primary_root, primary_audit = paths
    return decide(
        pre_gt,
        PROTOCOL,
        addendum,
        readiness,
        control_root,
        control_audit,
        primary_root,
        primary_audit,
        output,
        expected_pre_gt_audit_sha256=_sha(pre_gt),
        expected_evaluation_addendum_sha256=_sha(addendum),
        expected_readiness_sha256=_sha(readiness),
        expected_control_evaluation_audit_sha256=_sha(control_audit),
        expected_primary_evaluation_audit_sha256=_sha(primary_audit),
    )


def test_s8_decision_authorizes_only_full_operational_pass(tmp_path: Path) -> None:
    result = _decide(_fixture(tmp_path), tmp_path / "decision")
    assert result["gate"]["status"] == "OPERATIONAL_PASS"
    assert result["gate"]["mechanism_pass"] is True
    assert result["gate"]["consumer_authorized"] is True
    assert result["comparison"]["seed_family"] == BOOTSTRAP_SEED
    assert result["comparison"]["ground_truth_reopened"] is False


def test_s8_decision_rejects_medium_regression(tmp_path: Path) -> None:
    result = _decide(
        _fixture(tmp_path, primary_values={"small": 0.20, "medium": 0.50, "large": 0.55}),
        tmp_path / "decision",
    )
    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["mechanism_checks"]["medium_mean_no_regression"] is False
    assert result["gate"]["consumer_authorized"] is False


def test_s8_decision_rejects_evaluator_seed_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paired_path = paths[5] / "paired_comparison.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    paired["seed_family"] = 20261101
    _write_json(paired_path, paired)
    audit = json.loads(paths[6].read_text(encoding="utf-8"))
    audit["output_hashes"]["paired_comparison.json"] = _sha(paired_path)
    _write_json(paths[6], audit)
    with pytest.raises(ValueError, match="bootstrap"):
        _decide(paths, tmp_path / "decision")


def test_s8_decision_rejects_pre_gt_boundary_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pre_gt = json.loads(paths[1].read_text(encoding="utf-8"))
    pre_gt["validation_gt_read"] = True
    _write_json(paths[1], pre_gt)
    with pytest.raises(ValueError, match="boundary mismatch"):
        _decide(paths, tmp_path / "decision")
