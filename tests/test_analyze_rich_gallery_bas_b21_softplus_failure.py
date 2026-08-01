from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analyze_rich_gallery_bas_b21_softplus_failure import (  # noqa: E402
    activation_collapse_features,
    summarize,
)


def test_activation_features_detect_sparse_border_spike() -> None:
    activation = np.full((10, 10), 1.0e-12, dtype=np.float32)
    activation[0, 0] = 2.0e-7
    result = activation_collapse_features(activation)

    assert result["argmax_border"] == 1
    assert result["activation_max"] == pytest.approx(2.0e-7)
    assert result["sigmoid_logit_max"] < -15.0
    assert result["sigmoid_gradient_max"] == pytest.approx(2.0e-7, rel=1.0e-5)
    assert result["effective_support_fraction"] < 0.02
    assert result["top_1_percent_mass_fraction"] > 0.99


def test_activation_features_reject_invalid_map() -> None:
    with pytest.raises(ValueError, match="finite nonnegative 2-D"):
        activation_collapse_features(np.asarray([[0.0, -1.0]], dtype=np.float32))


def test_summarize_stratifies_without_changing_values() -> None:
    row = activation_collapse_features(np.full((4, 4), 0.25, dtype=np.float32))
    summary = summarize([row, row])

    assert summary["n"] == 2
    assert summary["activation_max_mean"] == pytest.approx(0.25)
    assert summary["argmax_border_mean"] == pytest.approx(1.0)
