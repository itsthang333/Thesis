from __future__ import annotations

import numpy as np

from project.analyze_rich_gallery_g1_conditional_information import (
    derive_matched_minus_random,
    matched_minus_random_name,
    partial_rank_correlation,
)


def test_partial_rank_correlation_removes_shared_area_confounder() -> None:
    area = np.arange(1.0, 9.0)
    target = area + np.asarray([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4])
    signal = area.copy()
    controls = area[:, None]
    assert abs(partial_rank_correlation(target, signal, controls)) < 1.0e-12


def test_partial_rank_correlation_preserves_independent_ordering() -> None:
    area = np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.float64)
    identity = np.asarray([1, 3, 2, 4, 5, 7, 6, 8], dtype=np.float64)
    target = identity.copy()
    signal = identity.copy()
    controls = area[:, None]
    assert partial_rank_correlation(target, signal, controls) > 0.99


def test_matched_minus_random_derivation_handles_top_level_and_stage_names() -> None:
    names = [
        "matched_logit_delta",
        "random_logit_delta",
        "transition2_matched_ring_mass",
        "transition2_random_ring_mass",
    ]
    row = {
        "matched_logit_delta": 3.0,
        "random_logit_delta": 1.0,
        "transition2_matched_ring_mass": 5.0,
        "transition2_random_ring_mass": 2.0,
    }
    derived = derive_matched_minus_random(row, names)
    assert matched_minus_random_name("matched_logit_delta") == (
        "matched_minus_random_logit_delta"
    )
    assert matched_minus_random_name("transition2_matched_ring_mass") == (
        "transition2_matched_minus_random_ring_mass"
    )
    assert derived == {
        "matched_minus_random_logit_delta": 2.0,
        "transition2_matched_minus_random_ring_mass": 3.0,
    }
