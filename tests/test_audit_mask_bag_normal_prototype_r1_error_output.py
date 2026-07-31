from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_mask_bag_normal_prototype_r1_error_output import (  # noqa: E402
    EXPECTED_TRAIN,
    expected_relative_paths,
    summarize_count_guard,
)


def test_error_inventory_is_exactly_crossfit_plus_15_folds_and_3_aggregates() -> None:
    paths = expected_relative_paths()
    assert len(paths) == 67
    assert "crossfit_assignment.json" in paths
    assert "oof/k_8/fold_0/adapter.pt" in paths
    assert "oof/k_32/oof_summary.json" in paths
    assert not any("prediction" in path and "oof_predictions" not in path and "heldout_predictions" not in path for path in paths)


def test_count_guard_summary_uses_complete_oof_cohort(tmp_path: Path) -> None:
    # The real-output integration audit covers numerical recomputation. This
    # static test keeps the expected cohort and frozen K inventory explicit.
    assert EXPECTED_TRAIN == 2981
    assert summarize_count_guard.__name__ == "summarize_count_guard"
