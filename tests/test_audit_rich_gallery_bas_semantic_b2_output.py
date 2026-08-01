from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from audit_rich_gallery_bas_semantic_b2_output import (
    _rank_correlation,
    _require_close,
    _verify_no_runner_import,
    unpack_prediction_payload,
)


def test_independent_prediction_unpack_reconstructs_exact_mask(tmp_path: Path) -> None:
    mask = np.zeros((7, 13), dtype=bool)
    mask[1:6, 4:10] = True
    path = tmp_path / "prediction.npz"
    np.savez_compressed(
        path,
        packed_mask=np.packbits(mask.reshape(-1), bitorder="little"),
        shape=np.asarray(mask.shape, dtype=np.int32),
    )
    np.testing.assert_array_equal(unpack_prediction_payload(path), mask)


def test_independent_prediction_unpack_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    np.savez_compressed(path, wrong=np.asarray([1], dtype=np.uint8))
    with pytest.raises(ValueError):
        unpack_prediction_payload(path)


def test_independent_evidence_tolerance_is_tight() -> None:
    _require_close(np.asarray([1.0]), np.asarray([1.0 + 1.0e-6]), name="ok")
    with pytest.raises(ValueError):
        _require_close(np.asarray([1.0]), np.asarray([1.0 + 3.0e-6]), name="bad")


def test_independent_constant_rank_is_conservatively_redundant() -> None:
    assert _rank_correlation(np.ones(3), np.arange(3)) == 1.0


def test_auditor_has_no_scientific_producer_import() -> None:
    auditor = Path(__file__).parents[1] / "project" / "audit_rich_gallery_bas_semantic_b2_output.py"
    _verify_no_runner_import(auditor)
    tree = ast.parse(auditor.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "run_rich_gallery_bas_semantic_b2" not in imported
