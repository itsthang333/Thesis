from __future__ import annotations

import ast
from pathlib import Path

import pytest

from audit_rich_gallery_bas_semantic_b2_evaluation import (
    _assert_nested_close,
    _verify_independence,
    paired_group_bootstrap,
)


def test_independent_evaluation_auditor_imports_neither_producer_nor_evaluator() -> None:
    source = Path(
        "project/audit_rich_gallery_bas_semantic_b2_evaluation.py"
    ).resolve()
    _verify_independence(source)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "evaluate_rich_gallery_bas_semantic_b2" not in imported
    assert "run_rich_gallery_bas_semantic_b2" not in imported


def test_independent_bootstrap_is_deterministic() -> None:
    expected = paired_group_bootstrap(
        [0.4, 0.7, 0.3],
        [0.2, 0.6, 0.4],
        ["a", "a", "b"],
        replicates=1000,
        seed=42,
    )
    assert expected["delta_semantic_minus_control"] == pytest.approx(0.0666666667)
    assert expected == paired_group_bootstrap(
        [0.4, 0.7, 0.3],
        [0.2, 0.6, 0.4],
        ["a", "a", "b"],
        replicates=1000,
        seed=42,
    )


def test_nested_comparator_rejects_metric_drift() -> None:
    _assert_nested_close({"x": [1.0, True]}, {"x": [1.0, True]}, "same")
    with pytest.raises(ValueError):
        _assert_nested_close({"x": [1.1, True]}, {"x": [1.0, True]}, "drift")
