from __future__ import annotations

import numpy as np

from project.analyze_rich_gallery_bas_b22_foreground_control_failure import (
    _spearman,
    spatial_shape_features,
)
from project.analyze_rich_gallery_bas_b22_spatial_failure import (
    _auc,
    _stable_fold,
)


def test_spatial_shape_features_separates_border_and_interior() -> None:
    activation = np.ones((10, 10), dtype=np.float32)
    activation[1:-1, 1:-1] = 3.0
    features = spatial_shape_features(activation)
    assert features["border_mean"] == 1.0
    assert features["interior_mean"] == 3.0
    assert features["border_minus_interior"] == -2.0


def test_spatial_shape_features_reports_broad_saturation() -> None:
    activation = np.concatenate(
        (np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32))
    ).reshape(10, 10)
    features = spatial_shape_features(activation)
    assert features["fraction_ge_0_90"] == 0.5
    assert features["fraction_ge_0_99"] == 0.5
    assert features["fraction_le_0_10"] == 0.5
    assert features["bimodal_extreme_fraction"] == 1.0


def test_spearman_is_tie_aware() -> None:
    assert np.isclose(_spearman([1.0, 1.0, 2.0], [3.0, 3.0, 4.0]), 1.0)
    assert np.isclose(_spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)


def test_auc_is_tie_aware_and_directional() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    assert np.isclose(_auc(labels, np.asarray([0.0, 1.0, 2.0, 3.0])), 1.0)
    assert np.isclose(_auc(labels, np.asarray([3.0, 2.0, 1.0, 0.0])), 0.0)
    assert np.isclose(_auc(labels, np.ones(4)), 0.5)


def test_stable_fold_is_deterministic_and_bounded() -> None:
    observed = [_stable_fold(f"group-{index}") for index in range(100)]
    repeated = [_stable_fold(f"group-{index}") for index in range(100)]
    assert observed == repeated
    assert set(observed).issubset(set(range(5)))
    assert len(set(observed)) == 5
