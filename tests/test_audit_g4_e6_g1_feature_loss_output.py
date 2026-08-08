from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_g4_e6_g1_feature_loss_output import (  # noqa: E402
    BASELINE_ARM,
    SEEDS,
    _aggregate,
    _require_matrix,
    expected_reported_arms,
    expected_unique_models,
)


def test_expected_e6_arm_and_checkpoint_sets_are_exact() -> None:
    assert len(expected_unique_models()) == 21
    assert len(expected_reported_arms()) == 24
    assert BASELINE_ARM not in expected_reported_arms()


def test_require_matrix_accepts_exact_population_and_rejects_duplicate() -> None:
    arms = {BASELINE_ARM, "learned"}
    rows = []
    for index in range(371):
        for arm in sorted(arms):
            rows.append(
                {
                    "image_id": f"IMG{index:06d}.jpeg",
                    "tumor": "1" if index < 184 else "0",
                    "arm": arm,
                }
            )
    image_ids, tumor_images = _require_matrix(rows, arms, name="synthetic")
    assert len(image_ids) == 371
    assert tumor_images == 184
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError):
        _require_matrix(rows, arms, name="tampered")


def test_aggregate_reports_sample_sd_and_full_alias_identity() -> None:
    summary = {"summaries": {}}
    label_metrics = {}
    prefixes = (
        "E6F__inside_only",
        "E6F__inside_ring",
        "E6F__inside_ring_contrast",
        "E6F__full",
        "E6L__bag_only",
        "E6L__bag_negative",
        "E6L__bag_selfguided",
        "E6L__full",
    )
    for prefix in prefixes:
        for offset, seed in enumerate(SEEDS):
            arm = f"{prefix}__seed{seed}"
            value = 0.2 + 0.1 * offset
            summary["summaries"][arm] = {
                "mean_tumor_dice": value,
                "selected_dice_common320": value + 0.01,
                "native_subgroups": {
                    "small_lt_1pct": {"mean_tumor_dice": value - 0.02},
                    "medium_1_to_5pct": {"mean_tumor_dice": value},
                    "large_ge_5pct": {"mean_tumor_dice": value + 0.02},
                },
            }
            label_metrics[arm] = {"auroc": value + 0.5}
    result = _aggregate(summary, label_metrics)
    assert result["feature_full"] == result["loss_full"]
    assert abs(result["loss_bag_negative"]["native_dice"]["mean"] - 0.3) < 1e-12
    assert abs(result["loss_bag_negative"]["native_dice"]["sample_sd"] - 0.1) < 1e-12
