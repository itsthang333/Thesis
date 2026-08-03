from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path

import pytest


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


def test_paired_bootstrap_is_fail_closed_and_python39_compatible() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("def _paired_group_bootstrap(")
    stop = source.index("\ndef _verify_prediction_manifest(", start)
    bootstrap = source[start:stop]
    assert 'raise ValueError("paired bootstrap vectors must have equal length")' in bootstrap
    assert "len(arm) != len(baseline)" in bootstrap
    assert "len(arm) != len(groups)" in bootstrap
    assert "zip(delta, groups)" in bootstrap
    assert "strict=True" not in bootstrap


def test_consumer_and_test_stay_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
    assert "train_consumer" not in source
    assert "test_loader" not in source


def test_validation_loader_cannot_verify_locked_test_bytes() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _write_validation_projection(" in source
    assert 'row.get("split") != "val"' in source
    assert 'row.get("eligible") != "1"' in source
    assert "split_manifest=validation_projection" in source
    assert "split_manifest=args.split_manifest" not in source
    assert '"locked_test_bytes_read": False' in source


def test_validation_projection_is_val_only_and_deterministic(tmp_path: Path) -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_write_validation_projection"
    )
    namespace = {
        "Path": Path,
        "csv": csv,
        "sha256_file": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    write_projection = namespace["_write_validation_projection"]

    rows = [
        {
            "image_id": f"val_{index:03d}.png",
            "group_id": f"group_{index:03d}",
            "split": "val",
            "eligible": "1",
            "tumor": "0",
            "image_sha256": "",
        }
        for index in range(371)
    ]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first_sha = write_projection(first, rows)
    second_sha = write_projection(second, rows)
    assert first_sha == second_sha == hashlib.sha256(first.read_bytes()).hexdigest()
    with first.open("r", encoding="utf-8", newline="") as handle:
        projected = list(csv.DictReader(handle))
    assert len(projected) == 371
    assert {row["split"] for row in projected} == {"val"}

    invalid = [dict(row) for row in rows]
    invalid[0]["split"] = "test"
    with pytest.raises(ValueError, match="cohort mismatch"):
        write_projection(tmp_path / "invalid.csv", invalid)
