from __future__ import annotations

from pathlib import Path

from project.evaluate_rich_gallery_matched_normal_transplant_stage_b import (
    _layer_bottleneck,
    recipient_pair_sign_agreement,
)
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_source_has_no_annotation_or_segmentation_reader() -> None:
    source = (
        ROOT / "project/freeze_rich_gallery_matched_normal_transplant_stage_a.py"
    ).read_text(encoding="utf-8")
    assert "build_segmentation_dataset" not in source
    assert "Annotations" not in source
    assert "annotation_path" not in source
    assert '"test"' not in source


def test_layer_bottleneck_identifies_early_signal_absence() -> None:
    rows = []
    for _ in range(5):
        row = {
            "class_inside_oracle_percentile": 0.8,
            "logit_oracle_percentile": 0.8,
        }
        for stage in ("pool0", "transition1", "transition2", "transition3", "norm5"):
            row[f"{stage}_oracle_percentile"] = 0.4
            row[f"{stage}_quality_rank_correlation"] = 0.0
            row[f"{stage}_area_rank_correlation"] = 0.0
        rows.append(row)
    result = _layer_bottleneck(rows)
    assert (
        result["identified_first_failure_branch"]
        == "candidate_content_not_discriminative_at_stem_after_sham_cancellation"
    )


def test_layer_bottleneck_identifies_pooling_dilution() -> None:
    rows = []
    for _ in range(5):
        row = {
            "class_inside_oracle_percentile": 0.8,
            "logit_oracle_percentile": 0.55,
        }
        for stage in ("pool0", "transition1", "transition2", "transition3", "norm5"):
            row[f"{stage}_oracle_percentile"] = 0.8
            row[f"{stage}_quality_rank_correlation"] = 0.1
            row[f"{stage}_area_rank_correlation"] = 0.0
        rows.append(row)
    result = _layer_bottleneck(rows)
    assert (
        result["identified_first_failure_branch"]
        == "global_pooling_dilutes_spatial_tumor_response"
    )


def test_two_recipient_sign_agreement_is_recovered_from_mean_and_population_std() -> None:
    # [2, 4] agree positive; [-1, 1] straddles zero; [-4, -2] agree negative.
    mean = np.asarray([3.0, 0.0, -3.0])
    std = np.asarray([1.0, 1.0, 1.0])
    assert recipient_pair_sign_agreement(mean, std).tolist() == [1.0, 0.0, 1.0]


def test_deep_layer_bottleneck_uses_matched_random_specificity() -> None:
    rows = []
    for group in ("small", "medium", "large"):
        row = {
            "size_group": group,
            "baseline_complete_miss": 0,
            "baseline_wrong_source": 0,
            "matched_class_inside_oracle_percentile": 0.8,
            "random_class_inside_oracle_percentile": 0.79,
            "matched_logit_oracle_percentile": 0.8,
            "random_logit_oracle_percentile": 0.79,
            "matched_random_score_rank_correlation": 0.95,
            "matched_recipient_sign_agreement_fraction": 1.0,
        }
        for stage in ("pool0", "transition1", "transition2", "transition3", "norm5"):
            for arm, percentile in (("matched", 0.8), ("random", 0.79)):
                row[f"{stage}_{arm}_oracle_percentile"] = percentile
                row[f"{stage}_{arm}_quality_rank_correlation"] = 0.1
                row[f"{stage}_{arm}_area_rank_correlation"] = 0.0
                row[f"{stage}_{arm}_oracle_relative_l2_contrast"] = 0.2
                row[f"{stage}_{arm}_oracle_recipient_cv"] = 0.1
            row[f"{stage}_oracle_percentile_gain"] = 0.01
            row[f"{stage}_quality_rank_correlation_gain"] = 0.0
        rows.append(row)
    result = _layer_bottleneck(rows)
    assert (
        result["identified_first_failure_branch"]
        == "matched_recipients_do_not_add_tumor_specific_candidate_identity"
    )
