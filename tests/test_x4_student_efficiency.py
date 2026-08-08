from __future__ import annotations

import pytest

from freeze_x4_student_predictions import latency_summary


def test_latency_summary_reports_median_and_iqr():
    result = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert result["images"] == 4
    assert result["median_seconds_per_image"] == 2.5
    assert result["iqr_low_seconds_per_image"] == 1.75
    assert result["iqr_high_seconds_per_image"] == 3.25
    assert result["mean_seconds_per_image"] == 2.5


@pytest.mark.parametrize("values", [[], [float("nan")], [-1.0]])
def test_latency_summary_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        latency_summary(values)
