from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluate_x4_student_predictions import size_group  # noqa: E402


def test_x4_size_groups_follow_native_area_contract() -> None:
    assert size_group(0.0001) == "small_lt_1pct"
    assert size_group(0.009999) == "small_lt_1pct"
    assert size_group(0.01) == "medium_1_to_5pct"
    assert size_group(0.04999) == "medium_1_to_5pct"
    assert size_group(0.05) == "large_ge_5pct"
