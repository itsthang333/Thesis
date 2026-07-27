from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluate_rad_dino_multilayer_soft_region_probe import apply_gate


RUNNER = ROOT / "project" / "run_rad_dino_multilayer_soft_region_probe.py"
EVALUATOR = (
    ROOT / "project" / "evaluate_rad_dino_multilayer_soft_region_probe.py"
)


def _summary() -> dict[str, object]:
    return {
        "image_level_auroc_from_raw_p99": 0.80,
        "tumor_localization": {
            "overall": {"pixel_auroc": 0.80, "dice_p90": 0.11},
            "small": {"pixel_auroc": 0.82, "dice_p97": 0.04},
            "medium": {"dice_p90": 0.13},
            "large": {"dice_p90": 0.36},
        },
    }


def _comparison() -> dict[str, object]:
    return {
        "metrics": {
            "dice_p90": {
                "overall": {
                    "ci95_low": 0.001,
                    "delta_candidate_minus_affinity": 0.01,
                },
                "small": {
                    "delta_candidate_minus_affinity": 0.01,
                },
                "medium": {
                    "delta_candidate_minus_affinity": 0.01,
                },
                "large": {
                    "delta_candidate_minus_affinity": 0.01,
                },
            }
        }
    }


def test_runner_has_no_segmentation_dataset_or_test_access() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.startswith("datasets") for name in imported)
    assert 'split="test"' not in source
    assert "BTXRDSegmentationDataset" not in source


def test_evaluator_imports_gt_only_inside_post_freeze_function() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    freeze_offset = source.index("def verify_prediction_freeze")
    evaluator_offset = source.index("def evaluate_frozen_predictions")
    gt_import_offset = source.index(
        "from datasets.btxrd import BTXRDSegmentationDataset"
    )
    assert freeze_offset < evaluator_offset < gt_import_offset
    main_offset = source.index("def main()")
    verify_call_offset = source.index(
        "manifest, freeze = verify_prediction_freeze(args)",
        main_offset,
    )
    evaluate_call_offset = source.index(
        "evaluated, summary = evaluate_frozen_predictions(args, manifest)",
        main_offset,
    )
    assert verify_call_offset < evaluate_call_offset
    assert 'split="test"' not in source


def test_gate_requires_absolute_and_relative_checks() -> None:
    gate = apply_gate(_summary(), _comparison())
    assert gate["status"] == "PASS"
    failed_absolute = _summary()
    failed_absolute["tumor_localization"]["medium"]["dice_p90"] = 0.119
    assert apply_gate(failed_absolute, _comparison())["status"] == "FAIL"
    failed_relative = _comparison()
    failed_relative["metrics"]["dice_p90"]["small"][
        "delta_candidate_minus_affinity"
    ] = -1.0e-6
    assert apply_gate(_summary(), failed_relative)["status"] == "FAIL"
