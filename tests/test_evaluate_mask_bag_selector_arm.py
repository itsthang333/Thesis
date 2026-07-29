from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "evaluate_mask_bag_selector_arm.py"


def test_evaluator_has_one_explicit_post_freeze_gt_boundary() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    gt_loader = source.index("from datasets.factory import build_segmentation_dataset")
    for required in (
        "cache_freeze, cache_rows = _verify_selector_cache(args, val_rows)",
        "arm_freeze, arm_predictions, score_rows = _verify_arm(",
        "baseline_freeze, baseline_predictions = _verify_baseline(args, val_rows)",
    ):
        assert source.index(required) < gt_loader
    assert source.index("args.baseline_per_image.open(") > gt_loader
    assert 'split="test"' not in source


def test_all_candidate_scores_and_cache_indices_are_bound_before_gt() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    gt_loader = source.index("from datasets.factory import build_segmentation_dataset")
    for required in (
        "validate_candidate_score_manifest(",
        "load_selector_cache_record(",
        "np.array_equal(",
        'prediction["selected_candidate_index"]',
    ):
        assert source.index(required) < gt_loader
    assert "selected_original_index = int(candidate_indices[local_selected])" in source
    assert "oracle_original_index = int(candidate_indices[local_oracle])" in source
    assert (
        'prediction["candidate_logit_tta"]\n'
        '            != "mean_original_aligned_horizontal_flip"'
        in source
    )


def test_complete_ranking_diagnostics_are_reported() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ranking_source = (
        ROOT / "project" / "models" / "mask_bag_ranking_diagnostics.py"
    ).read_text(encoding="utf-8")
    for required in (
        "candidate_ranking_diagnostics(",
        "summarize_ranking_diagnostics(",
        '"oracle_best_rank"',
        '"selected_to_oracle_regret"',
        '"score_quality_spearman"',
    ):
        assert required in source
    assert '"recovered_baseline_misses"' in ranking_source
    assert '"lost_baseline_hits"' in ranking_source
    for k in (1, 3, 5, 10):
        assert f"top_{{k}}_oracle_reach" in source


def test_mechanism_and_operational_pass_are_distinct() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '"OPERATIONAL_PASS"' in source
    assert '"MECHANISM_PASS"' in source
    assert '"FAIL"' in source
    assert '"consumer_authorized": operational_pass' in source
    assert "len(improved_subgroups) >= 2" in source
    assert "arm_count_magnitude <= baseline_count_magnitude" in source
    assert "paired[\"overall\"][\"delta_arm_minus_baseline\"] >= 0.0" in source


def test_operational_goals_and_complete_cohort_are_fixed() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for value in ("0.34024039", "0.17895493", "0.51244178", "0.49370336"):
        assert value in source
    assert "args.bootstrap_replicates != 10000" in source
    assert "len(per_image) != 184" in source
    assert '{"small": 94, "medium": 72, "large": 18}' in source
    assert '"complete_misses_included": True' in source
    assert '"candidate_oracle_goal_checks": oracle_checks' in source
    assert "all(check[\"pass\"] for check in oracle_checks.values())" in source


def test_consumer_and_test_stay_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
    assert "train_consumer" not in source
    assert "test_loader" not in source
