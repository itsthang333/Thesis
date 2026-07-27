from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "compare_mask_bag_evaluated_arms.py"


def test_evaluated_arm_comparator_never_reopens_segmentation_data() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "datasets.btxrd" not in imported
    assert "PIL" not in imported
    assert "Annotations" not in text
    assert '"ground_truth_reopened": False' in text
    assert '"consumer_trained": False' in text
    assert '"test_evaluated": False' in text


def test_evaluated_arm_comparator_freezes_cohort_metric_and_bootstrap() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "len(rows) != 184" in text
    assert 'EXPECTED_COUNTS = {"small": 94, "medium": 72, "large": 18}' in text
    assert "args.bootstrap_replicates != 10000" in text
    assert "args.bootstrap_seed != 20261101" in text
    assert "Frozen paired field" in text
    assert '"oracle_best_single_dice"' in text
    assert '"complete_misses_included": True' in text
    assert '"misses_recovered"' in text
    assert '"overlaps_lost"' in text
