from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "evaluate_rich_gallery_g1_ranking.py"


def test_annotations_open_only_after_all_frozen_inputs_are_verified() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    evidence = source.index("freeze, evidence_rows = _load_frozen_evidence(")
    candidates = source.index("validate_candidate_diagnostics_manifest(")
    variant_freeze = source.index("variant_selections, variant_freeze_sha256 =")
    annotation_import = source.index("from datasets.factory import")
    assert evidence < candidates < variant_freeze < annotation_import
    assert '"candidate_scores_frozen_before_gt": True' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source


def test_ranking_evaluator_decomposes_source_and_within_source_regret() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "source_choice_regret =" in source
    assert "within_source_regret =" in source
    assert "source_choice_regret" in source
    assert "within_source_regret" in source
    assert "regret decomposition identity failed" in source
    assert "oracle_best_rank" in source
    assert "mil_oracle_weight" in source
    assert "hard_positive_is_low_quality" in source


def test_ranking_evaluator_reports_actual_candidate_quality_by_subgroup() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "candidate_ranking_diagnostics(logits, dice" in source
    assert 'subgroup_counts != {"small": 94, "medium": 72, "large": 18}' in source
    assert '"selected_dice"' in source
    assert '"oracle_dice"' in source
    assert '"dice_vs_log_area_error_spearman"' in source
    assert '"pooled_score_dice_spearman"' in source


def test_mechanism_ablation_choices_are_gt_blind_and_report_actual_dice() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        '"upstream_selection"',
        '"source_zscore_g1"',
        '"without_inside_discrimination"',
        '"without_context_discrimination"',
        '"without_contrast_discrimination"',
        '"without_metadata_discrimination"',
    ):
        assert required in source
    assert 'path = args.output_dir / "selector_variant_freeze.json"' in source
    assert '"validation_gt_read": False' in source
    assert '"selector_variant_actual_dice"' in source
