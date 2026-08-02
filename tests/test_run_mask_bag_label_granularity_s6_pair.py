from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from project.run_mask_bag_label_granularity_s6_pair import (
    EXPECTED_SUBTYPE_COUNTS,
    _binary_metrics,
    _diagnostic_summary,
    _rank_auc,
    _subtype_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_label_granularity_s6_pair.py"


def test_runner_has_no_evaluator_or_annotation_import() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("evaluate" in name for name in imported)
    assert not any("annotation" in name for name in imported)
    source = SOURCE.read_text(encoding="utf-8")
    assert "validation_subtype_label_used_for_routing\": False" in source
    assert "pair_physically_frozen_before_validation_gt\": True" in source
    assert EXPECTED_SUBTYPE_COUNTS == (598, 211, 164, 74, 35, 41, 92, 237, 36)


def test_rank_auc_and_binary_metrics_handle_ties() -> None:
    labels = np.asarray([0, 0, 1, 1])
    perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
    tied = np.asarray([0.5, 0.5, 0.5, 0.5])
    assert _rank_auc(labels, perfect) == 1.0
    assert _rank_auc(labels, tied) == 0.5
    assert _binary_metrics(labels, perfect)["auroc"] == 1.0


def _diagnostic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subtype in range(1, 10):
        rows.append(
            {
                "image_id": f"P{subtype}",
                "candidate_count": 10 + subtype,
                "tumor": 1,
                "tumor_type": subtype,
                "control_bag_probability": 0.6 + subtype * 0.01,
                "hierarchy_bag_probability": 0.7 + subtype * 0.01,
                "predicted_tumor_type": subtype,
                "predicted_subtype_probability": 0.8,
                "entropy_route_strength": 0.5,
                "control_selected_local_index": 0,
                "hierarchy_selected_local_index": subtype % 2,
                "control_original_flip_agreement": 1,
                "hierarchy_original_flip_agreement": 1,
            }
        )
    for index in range(9):
        rows.append(
            {
                "image_id": f"N{index}",
                "candidate_count": 30 + index,
                "tumor": 0,
                "tumor_type": 0,
                "control_bag_probability": 0.1 + index * 0.01,
                "hierarchy_bag_probability": 0.2 + index * 0.01,
                "predicted_tumor_type": 1,
                "predicted_subtype_probability": 0.2,
                "entropy_route_strength": 0.1,
                "control_selected_local_index": 0,
                "hierarchy_selected_local_index": 0,
                "control_original_flip_agreement": 1,
                "hierarchy_original_flip_agreement": 0,
            }
        )
    return rows


def test_diagnostics_report_subtype_and_nonblocking_contract() -> None:
    rows = _diagnostic_rows()
    subtype = _subtype_metrics(rows)
    assert subtype["accuracy"] == 1.0
    assert subtype["macro_recall"] == 1.0
    summary = _diagnostic_summary(rows)
    assert summary["records"] == 18
    assert summary["diagnostics_block_prediction_freeze"] is False
    assert summary["validation_gt_read"] is False
    assert summary["test_evaluated"] is False
    assert 0.0 < summary["changed_selection_fraction"] < 1.0


def test_subtype_metrics_fail_when_a_class_is_missing() -> None:
    rows = _diagnostic_rows()
    rows = [row for row in rows if int(row["tumor_type"]) != 9]
    with pytest.raises(ValueError, match="omits tumor subtype 9"):
        _subtype_metrics(rows)
