from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import run_mask_bag_normal_only_direct_anomaly_n1 as runner


def test_n1_runner_has_no_segmentation_gt_evaluator_or_consumer_import() -> None:
    source = (PROJECT / "run_mask_bag_normal_only_direct_anomaly_n1.py").read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("from ", "import "))
    )
    assert "datasets.factory" not in import_lines
    assert "evaluate_mask_bag_selector_arm" not in import_lines
    assert "consumer" not in import_lines
    assert "test" not in import_lines
    assert "image_level_normal_only" in source
    assert "direct_normal_anomaly_distance_not_classification_logit" in source


def test_n1_spearman_matches_exact_monotonic_and_rejects_constant() -> None:
    assert runner._absolute_spearman([1, 2, 3], [9, 7, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="nonconstant"):
        runner._absolute_spearman([1, 1, 1], [1, 2, 3])


def test_n1_controls_and_safety_are_fail_closed_in_source() -> None:
    source = (PROJECT / "run_mask_bag_normal_only_direct_anomaly_n1.py").read_text(encoding="utf-8")
    main_source = source[source.index("def main() -> None:") :]
    assert "args.prototype_count != 32 or args.seed != 42" in source
    assert 'torch.cuda.device_count() != 2' in source
    assert 'all("T4" in name for name in device_names)' in source
    assert 'if args.output_dir.exists()' in source
    assert main_source.index("_verify_cache_freeze(args)") < main_source.index(
        "fit_direct_normal_anomaly_bank"
    )
    assert main_source.index("_verify_baseline(args") < main_source.index(
        "fit_direct_normal_anomaly_bank"
    )
