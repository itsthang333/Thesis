from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from freeze_skelex_reconstruction_selector_s8_postfreeze_readiness import (  # noqa: E402
    ARMS,
    BASELINE_FREEZE_SHA256,
    CACHE_FREEZE_SHA256,
    EXPERIMENT_ID,
    PRE_GT_STATUS,
    PROTOCOL_SHA256,
    READINESS_STATUS,
    SOURCE_COMMIT,
    SPLIT_SHA256,
    freeze_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json"
ADDENDUM = ROOT / (
    "artifacts/research_protocols/"
    "skelex_reconstruction_selector_s8_v1_postfreeze_evaluation_addendum.json"
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "producer"
    arm_hashes: dict[str, str] = {}
    for index, arm in enumerate(ARMS):
        arm_root = output_root / arm
        score_manifest = arm_root / "candidate_scores/candidate_score_manifest.csv"
        prediction_manifest = arm_root / "predictions/prediction_manifest.csv"
        score_manifest.parent.mkdir(parents=True, exist_ok=True)
        prediction_manifest.parent.mkdir(parents=True, exist_ok=True)
        score_manifest.write_text(f"arm,index\n{arm},{index}\n", encoding="utf-8")
        prediction_manifest.write_text(f"arm,index\n{arm},{index}\n", encoding="utf-8")
        freeze = {
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "source_commit": SOURCE_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "split_sha256": SPLIT_SHA256,
            "selector_cache_freeze_sha256": CACHE_FREEZE_SHA256,
            "baseline_prediction_freeze_sha256": BASELINE_FREEZE_SHA256,
            "validation_predictions": 371,
            "candidate_score_manifest_sha256": _sha(score_manifest),
            "prediction_manifest_sha256": _sha(prediction_manifest),
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        _write_json(freeze_path, freeze)
        arm_hashes[arm] = _sha(freeze_path)
    pair_path = output_root / "prediction_pair_freeze.json"
    _write_json(
        pair_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "source_commit": SOURCE_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "arms": arm_hashes,
            "pair_physically_frozen_before_validation_gt": True,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    pre_gt = tmp_path / "pre_gt.json"
    _write_json(
        pre_gt,
        {
            "status": PRE_GT_STATUS,
            "experiment_id": EXPERIMENT_ID,
            "validation_predictions_per_arm": 371,
            "physical_prediction_maps_verified": 742,
            "candidate_projections_reproduced": 371,
            "spatial_null_distributions_reproduced": 371,
            "nonconstant_reconstruction_banks": 371,
            "pair_freeze_sha256": _sha(pair_path),
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    return output_root, pre_gt


def _freeze(output_root: Path, pre_gt: Path, destination: Path) -> dict[str, object]:
    return freeze_readiness(
        output_root,
        pre_gt,
        PROTOCOL,
        ADDENDUM,
        destination,
        expected_pre_gt_audit_sha256=_sha(pre_gt),
        expected_evaluation_addendum_sha256=_sha(ADDENDUM),
    )


def test_s8_postfreeze_readiness_binds_all_dynamic_hashes(tmp_path: Path) -> None:
    output_root, pre_gt = _fixture(tmp_path)
    destination = tmp_path / "readiness.json"
    result = _freeze(output_root, pre_gt, destination)
    assert result["status"] == READINESS_STATUS
    assert result["prediction_pair_freeze_sha256"] == _sha(
        output_root / "prediction_pair_freeze.json"
    )
    assert set(result["arms"]) == set(ARMS)
    assert result["validation_gt_read"] is False
    with pytest.raises(FileExistsError):
        _freeze(output_root, pre_gt, destination)


def test_s8_postfreeze_readiness_rejects_score_manifest_tamper(tmp_path: Path) -> None:
    output_root, pre_gt = _fixture(tmp_path)
    score_manifest = output_root / ARMS[1] / "candidate_scores/candidate_score_manifest.csv"
    score_manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dynamic hash contract"):
        _freeze(output_root, pre_gt, tmp_path / "readiness.json")


def test_s8_postfreeze_readiness_rejects_pre_gt_safety_drift(tmp_path: Path) -> None:
    output_root, pre_gt = _fixture(tmp_path)
    payload = json.loads(pre_gt.read_text(encoding="utf-8"))
    payload["validation_gt_read"] = True
    _write_json(pre_gt, payload)
    with pytest.raises(ValueError, match="safety boundary"):
        _freeze(output_root, pre_gt, tmp_path / "readiness.json")
