from __future__ import annotations

import numpy as np

from project.run_rich_gallery_bas_b22_foreground_control_probe import (
    _mechanics_gate,
    _spatial_mechanics,
)


def _passing_gate() -> dict[str, object]:
    history = [
        {
            "full_ce": 0.40,
            "foreground_ce": 0.60,
            "accuracy": 0.80,
        }
    ]
    validation = {
        "auroc": 0.75,
        "activation_range_mean": 0.20,
        "tumor_nondegenerate_activation_fraction": 1.0,
    }
    spatial = {
        "tumor_argmax_border_fraction": 0.20,
        "tumor_top_1_percent_mass_median": 0.40,
        "tumor_effective_support_median": 0.10,
    }
    area = {"tumor_bas_area_spearman_mean": 0.50}
    return _mechanics_gate(history, validation, spatial, area)


def test_all_foreground_control_mechanics_gates_must_pass() -> None:
    result = _passing_gate()
    assert result["pass"] is True
    assert all(result["checks"].values())


def test_border_shortcut_alone_blocks_continuation() -> None:
    history = [{"full_ce": 0.40, "foreground_ce": 0.60, "accuracy": 0.80}]
    validation = {
        "auroc": 0.75,
        "activation_range_mean": 0.20,
        "tumor_nondegenerate_activation_fraction": 1.0,
    }
    spatial = {
        "tumor_argmax_border_fraction": 1.0,
        "tumor_top_1_percent_mass_median": 0.40,
        "tumor_effective_support_median": 0.10,
    }
    area = {"tumor_bas_area_spearman_mean": 0.50}
    result = _mechanics_gate(history, validation, spatial, area)
    assert result["pass"] is False
    assert result["checks"]["tumor_argmax_border_fraction"] is False


def test_spatial_mechanics_uses_exact_tumor_cohort() -> None:
    activations: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for index in range(184):
        image_id = f"tumor-{index}"
        activation = np.full((10, 10), 0.1, dtype=np.float32)
        activation[5, 5] = 0.9
        activations[image_id] = activation
        rows.append({"image_id": image_id, "tumor": 1})
    result = _spatial_mechanics(activations, rows)
    assert result["tumor_argmax_border_fraction"] == 0.0
    assert result["tumor_effective_support_median"] > 0.1
