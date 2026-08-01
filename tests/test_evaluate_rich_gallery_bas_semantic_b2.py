from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluate_rich_gallery_bas_semantic_b2 import (
    CONTROL_ARM,
    SEMANTIC_ARM,
    _summarize,
    average_percentile_rank,
    dice,
    iou,
    mechanism_diagnostics,
    paired_group_bootstrap,
    rank_correlation,
    safe_path,
    size_group,
    unpack_prediction_payload,
)


def test_binary_metrics_and_size_groups() -> None:
    target = np.zeros((10, 10), dtype=bool)
    target[:2, :2] = True
    prediction = target.copy()
    assert dice(prediction, target) == 1.0
    assert iou(prediction, target) == 1.0
    assert size_group(0.009) == "small"
    assert size_group(0.01) == "medium"
    assert size_group(0.05) == "large"


def test_candidate_rank_diagnostics_are_tie_aware_and_finite() -> None:
    np.testing.assert_allclose(
        average_percentile_rank(np.asarray([2.0, 1.0, 1.0])),
        [1.0, 0.25, 0.25],
    )
    assert rank_correlation(
        np.asarray([0.0, 1.0, 2.0]), np.asarray([2.0, 1.0, 0.0])
    ) == pytest.approx(-1.0)
    assert rank_correlation(np.ones(3), np.arange(3)) == 0.0


def test_group_bootstrap_is_deterministic_and_paired() -> None:
    result = paired_group_bootstrap(
        [0.5, 0.6, 0.8],
        [0.4, 0.4, 0.7],
        ["a", "a", "b"],
        replicates=1000,
        seed=42,
    )
    assert result["delta_semantic_minus_control"] == pytest.approx(0.1333333333)
    assert result == paired_group_bootstrap(
        [0.5, 0.6, 0.8],
        [0.4, 0.4, 0.7],
        ["a", "a", "b"],
        replicates=1000,
        seed=42,
    )


def test_failure_summary_preserves_regret_and_rank_depth() -> None:
    records: list[dict[str, object]] = []
    for index, subgroup in enumerate(("small", "medium", "large"), start=1):
        records.append(
            {
                "arm": CONTROL_ARM,
                "dice": 0.1 * index,
                "iou": 0.05 * index,
                "complete_miss": int(index == 1),
                "precision": 0.2,
                "recall": 0.3,
                "selected_area_ratio": 0.1,
                "selected_to_gt_area_ratio": 2.0,
                "gallery_oracle_dice": 0.8,
                "eligible_oracle_dice": 0.7,
                "selector_regret": 0.8 - 0.1 * index,
                "wrong_source_regret": 0.2,
                "within_selected_source_regret": 0.5 - 0.1 * index,
                "candidate_truncation_regret": 0.1,
                "eligible_oracle_rank": 10 * index,
                "score_quality_rank_correlation": 0.5,
                "selected_source_matches_eligible_oracle": int(index == 2),
                "size_group": subgroup,
                **{
                    f"top{depth}_oracle_dice": min(0.7, 0.1 * depth)
                    for depth in (1, 3, 5, 10, 20, 50)
                },
            }
        )
    summary = _summarize(records, CONTROL_ARM)
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["gallery_oracle_dice"] == pytest.approx(0.8)
    assert summary["overall"]["candidate_truncation_regret"] == pytest.approx(0.1)
    assert summary["overall"]["selector_regret"] == pytest.approx(
        summary["overall"]["candidate_truncation_regret"]
        + summary["overall"]["wrong_source_regret"]
        + summary["overall"]["within_selected_source_regret"]
    )
    assert summary["overall"]["eligible_oracle_rank_median"] == 20.0
    assert summary["overall"]["topk_restricted_oracle_dice"]["10"] == pytest.approx(0.7)
    assert summary["small"]["recoverable_complete_misses"] == {
        "0.1": 1,
        "0.3": 1,
        "0.5": 1,
    }


def test_prediction_unpack_and_safe_path(tmp_path: Path) -> None:
    arm_root = tmp_path / "arm"
    prediction_root = arm_root / "predictions"
    prediction_root.mkdir(parents=True)
    mask = np.eye(7, dtype=bool)
    path = prediction_root / "image.npz"
    np.savez_compressed(
        path,
        packed_mask=np.packbits(mask.reshape(-1), bitorder="little"),
        shape=np.asarray(mask.shape, dtype=np.int32),
    )
    resolved = safe_path(arm_root, "predictions/image.npz")
    assert resolved == path.resolve()
    np.testing.assert_array_equal(unpack_prediction_payload(resolved), mask)
    with pytest.raises(ValueError):
        safe_path(arm_root, "../escape.npz")


def test_mechanism_diagnostics_reports_only_paired_new_arm_changes() -> None:
    records: list[dict[str, object]] = []
    examples = [
        ("a.png", "g1", "small", 0.0, 0.4, 1, 0, 3, 4, "x", "y", 0.02),
        ("b.png", "g2", "medium", 0.7, 0.5, 0, 0, 2, 5, "x", "x", 0.10),
    ]
    for (
        image_id,
        group_id,
        subgroup,
        control_dice,
        semantic_dice,
        control_miss,
        semantic_miss,
        control_index,
        semantic_index,
        control_source,
        semantic_source,
        gt_area,
    ) in examples:
        for arm, score, miss, candidate, source, area in (
            (CONTROL_ARM, control_dice, control_miss, control_index, control_source, 0.1),
            (SEMANTIC_ARM, semantic_dice, semantic_miss, semantic_index, semantic_source, 0.2),
        ):
            records.append(
                {
                    "arm": arm,
                    "image_id": image_id,
                    "group_id": group_id,
                    "size_group": subgroup,
                    "gt_area_ratio": gt_area,
                    "dice": score,
                    "complete_miss": miss,
                    "selected_area_ratio": area,
                    "selected_source": source,
                    "selected_candidate_index": candidate,
                }
            )
    # The production cohort guard is intentional; extend the synthetic set
    # without changing its two non-trivial paired cases.
    for index in range(2, 184):
        for arm in (CONTROL_ARM, SEMANTIC_ARM):
            records.append(
                {
                    "arm": arm,
                    "image_id": f"same-{index}.png",
                    "group_id": f"g{index}",
                    "size_group": "large",
                    "gt_area_ratio": 0.1,
                    "dice": 0.5,
                    "complete_miss": 0,
                    "selected_area_ratio": 0.1,
                    "selected_source": "x",
                    "selected_candidate_index": 1,
                }
            )
    result = mechanism_diagnostics(records)
    assert result["changed_positive_choices"] == 2
    assert result["paired_images_improved"] == 1
    assert result["paired_images_worsened"] == 1
    assert result["paired_images_tied"] == 182
    assert result["complete_misses_recovered"] == 1
    assert result["complete_hits_lost"] == 0
    assert result["positive_dice_mass"] == pytest.approx(0.4)
    assert result["negative_dice_mass"] == pytest.approx(-0.2)
    assert result["changed_choice_source_transitions"] == {"x->x": 1, "x->y": 1}
