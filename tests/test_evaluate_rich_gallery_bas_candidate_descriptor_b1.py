from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from project.evaluate_rich_gallery_bas_candidate_descriptor_b1 import (
    _failure_decomposition,
    _render_mechanism_dossier,
    dice,
    iou,
    rank_correlation,
    reproduce_frozen_selection,
    size_group,
)
from project.run_rich_gallery_bas_candidate_descriptor_b1 import build_variant_scores


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


def test_frozen_selection_rebuilds_float64_rank_fusion_before_tie_break() -> None:
    count = 155
    g1 = np.arange(count, dtype=np.float32)
    g1[[48, 152]] = g1[[152, 48]]
    g1[[60, 151]] = g1[[151, 60]]
    # Pair every non-target G1 rank with its reverse BAS rank, then reserve
    # BAS ranks 152/153 for the two target candidates.  They are therefore the
    # only global maxima of the two-way fusion.
    bas = (count - 1 - g1).astype(np.float32)
    bas_rank_152 = int(np.flatnonzero(bas == 152)[0])
    bas[[48, bas_rank_152]] = bas[[bas_rank_152, 48]]
    bas_rank_153 = int(np.flatnonzero(bas == 153)[0])
    bas[[60, bas_rank_153]] = bas[[bas_rank_153, 60]]
    upstream = np.arange(count, dtype=np.float32)
    rebuilt = build_variant_scores(g1, upstream, bas)
    payload = {
        "g1_logits": g1,
        "upstream_scores": upstream,
        "bas_scores": bas,
        **{name: values.astype(np.float32) for name, values in rebuilt.items()},
    }

    # The transported fusion ties at float32, where a direct tie break would
    # choose index 48.  The immutable Stage-A float64 decision is index 60.
    assert payload["g1_bas_two_way"][48] == payload["g1_bas_two_way"][60]
    assert reproduce_frozen_selection(payload, "g1_bas_two_way") == 60


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

    dossier = _render_mechanism_dossier(
        summary,
        {
            "overall": {"ci95_low": -0.02, "ci95_high": 0.01},
            "small": {"ci95_low": -0.03, "ci95_high": 0.01},
            "medium": {"ci95_low": -0.02, "ci95_high": 0.02},
            "large": {"ci95_low": -0.05, "ci95_high": 0.03},
        },
        result,
        {"pass": False},
    )
    assert "Exact selector-regret decomposition" in dossier
    assert "within_selected_source" in dossier
    assert "equal_three_way_fusion_dilutes_complementary_bas_signal" in dossier
    assert "Promotion pass: `false`" in dossier
    assert "BTXRD test remains locked" in dossier
