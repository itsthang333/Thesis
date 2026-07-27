from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.audit_rad_dino_multilayer_soft_region_probe import (
    paired_group_bootstrap,
)


AUDITOR = (
    ROOT / "project/tools/audit_rad_dino_multilayer_soft_region_probe.py"
)


def test_auditor_defers_gt_until_after_no_gt_physical_audit() -> None:
    source = AUDITOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evaluate_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_from_gt"
    )
    gt_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "datasets.btxrd"
    ]
    assert len(gt_imports) == 1
    assert gt_imports[0] in set(ast.walk(evaluate_node))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main)
    assert main_source.index("audit_no_gt_artifacts") < main_source.index(
        "evaluate_from_gt"
    )
    assert not any(
        keyword.arg == "split"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "test"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )


def test_independent_bootstrap_uses_physical_ci95_list_schema() -> None:
    result = paired_group_bootstrap(
        [("a", 0.1), ("b", 0.2)], replicates=100, seed=42
    )
    assert result["delta_candidate_minus_affinity"] == pytest.approx(0.15)
    assert isinstance(result["ci95"], list)
    assert len(result["ci95"]) == 2
    assert "ci95_low" not in result
