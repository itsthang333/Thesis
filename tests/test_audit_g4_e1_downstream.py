from __future__ import annotations

import ast
from pathlib import Path

from project.audit_g4_e1_downstream import RUNNER_SHA256, SOURCE_COMMIT, no_test


ROOT = Path(__file__).resolve().parents[1]


def test_e1_downstream_auditor_is_exact_and_validation_only() -> None:
    source = (ROOT / "project" / "audit_g4_e1_downstream.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert SOURCE_COMMIT == {
        "binary": "b119a1dbd470f3802c60669e364db4912d5e755a",
        "ten_class": "c3cb2eb1c59466ecc6a455da666e925ea14d4718",
    }
    assert RUNNER_SHA256 == (
        "c2d0b60b13b73f0379168e83b1130aeb92a92bdafa81d6c52f69999a1bdfb4e5"
    )
    assert 'len(rows) != 184' in source
    assert 'len(selections) != 371' in source
    assert '"small": 94, "medium": 72, "large": 18' in source
    assert 'evaluation.get("validation_ablation") is not True' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source


def test_e1_no_test_contract_is_fail_closed() -> None:
    no_test({"test_evaluated": False, "test_images_read": 0}, name="valid")
    for invalid in (
        {"test_evaluated": True, "test_images_read": 0},
        {"test_evaluated": False, "test_images_read": 1},
        {"test_evaluated": False},
    ):
        try:
            no_test(invalid, name="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid no-test artifact: {invalid}")
