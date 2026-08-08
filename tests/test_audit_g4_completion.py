from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_g4_completion import _zero_test  # noqa: E402


def test_zero_test_accepts_only_explicit_zero_false() -> None:
    _zero_test({"test_images_read": 0, "test_evaluated": False}, "good")
    with pytest.raises(ValueError):
        _zero_test({"test_images_read": 1, "test_evaluated": False}, "read")
    with pytest.raises(ValueError):
        _zero_test({"test_images_read": 0, "test_evaluated": True}, "evaluated")
    with pytest.raises(ValueError):
        _zero_test({}, "missing")
