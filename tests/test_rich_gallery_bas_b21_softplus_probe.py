from __future__ import annotations

import numpy as np

from project.run_rich_gallery_bas_b21_softplus_probe import (
    _mechanics_gate,
    _rank_correlation,
)


def test_softplus_probe_area_rank_correlation_is_tie_aware() -> None:
    assert np.isclose(
        _rank_correlation(
            np.asarray([1.0, 2.0, 2.0, 4.0]),
            np.asarray([10.0, 20.0, 20.0, 40.0]),
        ),
        1.0,
    )


def test_softplus_probe_gate_passes_only_noncollapsed_mechanics() -> None:
    history = [
        {"full_ce": 0.65, "accuracy": 0.65},
    ]
    validation = {
        "auroc": 0.70,
        "activation_range_mean": 0.10,
        "tumor_nondegenerate_activation_fraction": 0.90,
    }
    area = {"tumor_bas_area_spearman_mean": 0.50}
    result = _mechanics_gate(history, validation, area)
    assert result["pass"] is True
    assert all(result["checks"].values())

    collapsed = _mechanics_gate(
        [{"full_ce": 0.693359375, "accuracy": 1493 / 2981}],
        {
            "auroc": 0.5,
            "activation_range_mean": 1.0e-6,
            "tumor_nondegenerate_activation_fraction": 0.0,
        },
        {"tumor_bas_area_spearman_mean": 0.9999},
    )
    assert collapsed["pass"] is False
    assert not any(collapsed["checks"].values())
