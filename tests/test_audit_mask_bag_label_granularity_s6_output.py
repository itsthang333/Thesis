from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from project.audit_mask_bag_label_granularity_s6_output import (
    REPRODUCTION_ATOL,
    _absolute_spearman,
    _entropy_route,
    _safe_child,
    _sigmoid,
    _smooth_pool,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_label_granularity_s6_output.py"


def test_auditor_is_independent_of_runner_and_evaluator() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("run_mask_bag_label_granularity" in name for name in imported)
    assert not any("training" in name for name in imported)
    assert not any("evaluate" in name for name in imported)
    assert REPRODUCTION_ATOL == 5.0e-5


def test_independent_numeric_primitives() -> None:
    assert _smooth_pool(np.asarray([2.0, 2.0, 2.0])) == pytest.approx(2.0)
    assert _sigmoid(0.0) == 0.5
    predicted, route, probabilities = _entropy_route(np.zeros(9))
    assert predicted == 0
    assert route == pytest.approx(0.0, abs=1.0e-12)
    assert probabilities.sum() == pytest.approx(1.0)
    confident = np.full(9, -100.0)
    confident[6] = 100.0
    predicted, route, _ = _entropy_route(confident)
    assert predicted == 6
    assert route == pytest.approx(1.0)


def test_spearman_and_safe_child(tmp_path: Path) -> None:
    assert _absolute_spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)
    assert _safe_child(tmp_path, "scores/a.npz") == (
        tmp_path / "scores" / "a.npz"
    ).resolve()
    with pytest.raises(ValueError, match="escapes"):
        _safe_child(tmp_path, "../escape.npy")
