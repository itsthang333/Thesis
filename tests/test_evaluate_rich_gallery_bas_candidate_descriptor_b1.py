from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from project.evaluate_rich_gallery_bas_candidate_descriptor_b1 import (
    _failure_decomposition,
    dice,
    iou,
    rank_correlation,
    size_group,
)


def test_overlap_metrics_and_groups() -> None:
    target = np.asarray([[1, 1], [0, 0]], dtype=bool)
    prediction = np.asarray([[1, 0], [1, 0]], dtype=bool)
    assert dice(prediction, target) == 0.5
    assert iou(prediction, target) == 1.0 / 3.0
    assert size_group(0.009) == "small"
    assert size_group(0.01) == "medium"
    assert size_group(0.05) == "large"


def test_rank_correlation_is_tie_aware_and_finite() -> None:
    assert np.isclose(rank_correlation(np.asarray([0.0, 1.0, 2.0]), np.asarray([2.0, 1.0, 0.0])), -1.0)
    assert rank_correlation(np.ones(3), np.arange(3)) == 0.0


def test_evaluator_is_the_only_file_that_imports_segmentation_after_verification() -> None:
    path = Path(__file__).resolve().parents[1] / "project" / "evaluate_rich_gallery_bas_candidate_descriptor_b1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(getattr(node.func, "id", "") == "verify_stage_a" for node in calls)
    assert "build_segmentation_dataset" in source
    assert "split=\"test\"" not in source


def test_failure_decomposition_detects_fusion_dilution() -> None:
    groups = ("overall", "small", "medium", "large")
    def metrics(value: float, misses: int = 1) -> dict[str, dict[str, float]]:
        return {
            group: {
                "dice": value,
                "complete_misses": misses,
                "selected_gt_area_ratio_median": 1.0,
                "precision": value,
                "recall": value,
                "oracle_dice": 0.8,
                "selector_regret": 0.8 - value,
                "candidate_truncation_regret": 0.001,
                "cross_source_regret": 0.1,
                "within_selected_source_regret": 0.8 - value - 0.101,
                "top3_oracle_dice": 0.5,
                "top5_oracle_dice": 0.6,
                "top10_oracle_dice": 0.7,
                "top20_oracle_dice": 0.75,
                "top50_oracle_dice": 0.78,
            }
            for group in groups
        }
    summary = {
        "g1_upstream_baseline": metrics(0.3),
        "g1_upstream_bas_three_way": metrics(0.29),
        "bas_only": metrics(0.31),
        "g1_bas_two_way": metrics(0.32),
        "upstream_bas_two_way": metrics(0.28),
    }
    per_image = []
    for variant, value in (("g1_upstream_baseline", 0.3), ("g1_upstream_bas_three_way", 0.29)):
        per_image.append({
            "variant": variant,
            "image_id": "x",
            "group_id": "g",
            "size_group": "small",
            "dice": value,
            "complete_miss": 0,
            "selected_source": "classifier448",
        })
    result = _failure_decomposition(per_image, summary, {"mean_bas_upstream_rank_correlation": 0.2})
    assert "equal_three_way_fusion_dilutes_complementary_bas_signal" in result["identified_failure_branches"]
    assert result["no_next_gpu_run_before_manual_dossier_review"] is True
