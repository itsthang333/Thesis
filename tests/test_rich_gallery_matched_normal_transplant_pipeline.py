from __future__ import annotations

from pathlib import Path

from project.evaluate_rich_gallery_matched_normal_transplant_stage_b import (
    _layer_bottleneck,
)


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
