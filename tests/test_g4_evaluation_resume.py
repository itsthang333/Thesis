from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from resume_g4_offline_evaluation import coerce_csv_value  # noqa: E402


def test_resume_csv_value_coercion_preserves_identifiers_and_metrics() -> None:
    assert coerce_csv_value("True") is True
    assert coerce_csv_value("False") is False
    assert coerce_csv_value("17") == 17
    assert coerce_csv_value("-2") == -2
    assert coerce_csv_value("0.288729") == 0.288729
    assert coerce_csv_value("nan") != coerce_csv_value("nan")
    assert coerce_csv_value("external_saliency") == "external_saliency"
