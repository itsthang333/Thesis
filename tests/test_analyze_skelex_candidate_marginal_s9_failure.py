from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analyze_skelex_candidate_marginal_s9_failure import (
    CONTROL_ARM,
    EXPECTED_TUMOR,
    EXPECTED_VALIDATION,
    EXPERIMENT_ID,
    PRIMARY_ARM,
    analyze,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    producer = tmp_path / "producer"
    control_eval = tmp_path / "control_eval"
    primary_eval = tmp_path / "primary_eval"
    decision = tmp_path / "decision"
    controls: list[dict[str, object]] = []
    primaries: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    control_eval_rows: list[dict[str, object]] = []
    primary_eval_rows: list[dict[str, object]] = []
    for index in range(EXPECTED_VALIDATION):
        image_id = f"IMG{index:06d}.jpeg"
        tumor = int(index < EXPECTED_TUMOR)
        control_local = 0
        primary_local = 1 if index % 2 == 0 else 0
        indices = np.asarray([4, 9, 13], dtype=np.int32)
        likelihood = np.asarray([-0.7, -0.2, -0.5], dtype=np.float32)
        control_rank = np.asarray([0.9, 0.3, 0.1], dtype=np.float32)
        primary_rank = (
            np.asarray([0.4, 0.8, 0.1], dtype=np.float32)
            if primary_local == 1
            else control_rank.copy()
        )
        relative = Path(f"{index:04d}_{index}.npz")
        evidence_path = producer / "s9_likelihood_evidence" / relative
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            evidence_path,
            candidate_indices=indices,
            baseline_logits=np.asarray([0.8, 0.2, -0.1], dtype=np.float32),
            upstream_scores=np.asarray([0.7, 0.1, 0.0], dtype=np.float32),
            candidate_likelihood=likelihood,
            control_rank=control_rank,
            primary_rank=primary_rank,
            candidate_weights=np.stack(
                [
                    np.full((2, 2), 0.1, dtype=np.float16),
                    np.full((2, 2), 0.5, dtype=np.float16),
                    np.full((2, 2), 0.9, dtype=np.float16),
                ]
            ),
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "evidence_path": str(relative),
                "evidence_sha256": _sha(evidence_path),
            }
        )
        common = {"image_id": image_id, "tumor": tumor}
        controls.append(
            {
                **common,
                "selected_candidate_index": int(indices[control_local]),
                "selected_area_ratio": 0.1,
            }
        )
        primaries.append(
            {
                **common,
                "selected_candidate_index": int(indices[primary_local]),
                "selected_area_ratio": 0.2 if primary_local == 1 else 0.1,
            }
        )
        if tumor:
            control_dice = 0.2
            primary_dice = 0.3 if primary_local == 1 else 0.2
            size = ("small", "medium", "large")[index % 3]
            paired_rows.append(
                {
                    "image_id": image_id,
                    "size_group": size,
                    "control_dice": control_dice,
                    "primary_dice": primary_dice,
                    "delta_dice": primary_dice - control_dice,
                }
            )
            control_eval_rows.append({"image_id": image_id, "dice": control_dice})
            primary_eval_rows.append({"image_id": image_id, "dice": primary_dice})

    _csv(producer / CONTROL_ARM / "predictions/prediction_manifest.csv", controls)
    _csv(producer / PRIMARY_ARM / "predictions/prediction_manifest.csv", primaries)
    _csv(producer / "s9_likelihood_evidence/evidence_manifest.csv", evidence_rows)
    _csv(control_eval / "per_image.csv", control_eval_rows)
    _csv(primary_eval / "per_image.csv", primary_eval_rows)
    _csv(decision / "paired_per_image.csv", paired_rows)
    independent_path = producer / "independent_gt_blind_output_audit.json"
    _json(
        independent_path,
        {"status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_REPRODUCTION_PASS"},
    )
    _json(
        producer / "prediction_pair_freeze.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pair_physically_frozen_before_validation_gt": True,
        },
    )
    _json(
        producer / "wrapper_output_audit.json",
        {"independent_gt_blind_output_audit_sha256": _sha(independent_path)},
    )
    _json(
        decision / "gate_decision.json",
        {"status": "FAIL", "consumer_authorized": False, "test_evaluated": False},
    )
    _json(
        decision / "decision_audit.json",
        {"ground_truth_reopened_for_matched_comparison": False},
    )
    return producer, control_eval, primary_eval, decision


def test_failure_analyzer_reports_switch_mechanics_without_raw_gt(tmp_path: Path) -> None:
    producer, control_eval, primary_eval, decision = _fixture(tmp_path)
    output = tmp_path / "analysis.json"
    result = analyze(
        producer_root=producer,
        control_evaluation_root=control_eval,
        primary_evaluation_root=primary_eval,
        decision_root=decision,
        output=output,
    )
    assert output.is_file()
    assert result["switch_incidence"]["all"]["changed"] == 186
    assert result["switch_incidence"]["tumor"]["changed"] == 92
    assert result["changed_outcomes"]["tumor"]["wins"] == 92
    assert result["safety"]["raw_validation_gt_opened"] is False
    assert result["safety"]["post_hoc_rescue_or_sweep_performed"] is False


def test_failure_analyzer_rejects_nonfailed_decision(tmp_path: Path) -> None:
    producer, control_eval, primary_eval, decision = _fixture(tmp_path)
    _json(
        decision / "gate_decision.json",
        {"status": "PASS", "consumer_authorized": True, "test_evaluated": False},
    )
    with pytest.raises(RuntimeError, match="failure/safety"):
        analyze(
            producer_root=producer,
            control_evaluation_root=control_eval,
            primary_evaluation_root=primary_eval,
            decision_root=decision,
            output=tmp_path / "analysis.json",
        )


def test_failure_analyzer_rejects_tampered_evidence(tmp_path: Path) -> None:
    producer, control_eval, primary_eval, decision = _fixture(tmp_path)
    first = next((producer / "s9_likelihood_evidence").glob("*.npz"))
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="evidence hash"):
        analyze(
            producer_root=producer,
            control_evaluation_root=control_eval,
            primary_evaluation_root=primary_eval,
            decision_root=decision,
            output=tmp_path / "analysis.json",
        )
