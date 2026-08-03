from __future__ import annotations

import numpy as np

from project.analyze_skelex_reconstruction_selector_s8_failure import (
    classify_delta,
    mask_pair_dice,
    spearman,
    subset_summary,
    write_json_exclusive,
)


def test_delta_classification_is_fail_closed_around_epsilon() -> None:
    assert classify_delta(2.0e-12) == "win"
    assert classify_delta(-2.0e-12) == "loss"
    assert classify_delta(1.0e-13) == "tie"


def test_mask_pair_dice_handles_overlap_and_empty_masks() -> None:
    first = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    second = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    assert mask_pair_dice(first, second) == 0.5
    assert mask_pair_dice(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0


def test_spearman_handles_ties_and_constant_inputs() -> None:
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_subset_summary_counts_scientific_outcomes() -> None:
    rows = [
        {
            "delta_dice": 0.2,
            "permutation_p_value": 0.04,
            "observed_improvement": 0.1,
            "new_to_old_area_ratio": 0.25,
            "mask_pair_dice": 0.0,
            "same_family": False,
        },
        {
            "delta_dice": -0.1,
            "permutation_p_value": 0.01,
            "observed_improvement": 0.2,
            "new_to_old_area_ratio": 1.5,
            "mask_pair_dice": 0.5,
            "same_family": True,
        },
        {
            "delta_dice": 0.0,
            "permutation_p_value": 0.02,
            "observed_improvement": 0.3,
            "new_to_old_area_ratio": 0.75,
            "mask_pair_dice": 0.2,
            "same_family": False,
        },
    ]
    summary = subset_summary(rows)
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    assert summary["same_family_count"] == 1
    assert summary["area_shrink_count"] == 2
    assert summary["near_disjoint_count"] == 1


def test_write_json_exclusive_is_lf_and_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "nested" / "audit.json"
    write_json_exclusive(output, {"status": "PASS"})
    assert output.read_bytes() == b'{\n  "status": "PASS"\n}\n'
    try:
        write_json_exclusive(output, {"status": "REPLACED"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("exclusive writer unexpectedly overwrote an artifact")
