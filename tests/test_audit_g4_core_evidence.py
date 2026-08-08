from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_g4_core_evidence import E0_ARMS, OFFLINE_ARMS, _require_matrix  # noqa: E402


def test_declared_core_arm_sets_are_complete() -> None:
    assert len(E0_ARMS) == 6
    assert len(OFFLINE_ARMS) == 27
    assert {f"E8__R{index}" for index in range(9)} <= OFFLINE_ARMS
    assert {"E6__random", "E6__sam_only", "E6__upstream_only", "E6__g1_only"} <= OFFLINE_ARMS


def test_core_matrix_rejects_missing_arm() -> None:
    arms = {"a", "b"}
    rows = [
        {
            "image_id": f"IMG{index:06d}.jpeg",
            "arm": arm,
            "tumor": "1" if index < 184 else "0",
        }
        for index in range(371)
        for arm in sorted(arms)
    ]
    assert _require_matrix(rows, arms, tumor_field="tumor") == (371, 184)
    with pytest.raises(ValueError):
        _require_matrix(rows[:-1], arms, tumor_field="tumor")
