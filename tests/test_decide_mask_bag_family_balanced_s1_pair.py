from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "decide_mask_bag_family_balanced_s1_pair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("s1_pair_decision", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s1_decision_never_reopens_segmentation_gt_or_test() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "PIL" not in imported
    assert "Annotations" not in source
    assert '"ground_truth_reopened": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s1_decision_requires_causal_and_operational_pass_for_adoption() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '"regret_reduced_in_at_least_two_tumor_subgroups"' in source
    assert '"overall_selected_dice_no_regression"' in source
    assert '"absolute_count_miss_association_no_increase"' in source
    assert '"consumer_authorized": causal_pass and family_operational_pass' in source
    assert '"result_adoption_allowed": causal_pass and family_operational_pass' in source


def test_verify_evaluation_requires_physical_output_hashes_and_arm_freeze(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root = tmp_path / "evaluation"
    root.mkdir()
    summary = {
        "cohort": module.EXPECTED_COHORT,
        "arm_protocol_sha256": module.PROTOCOL_SHA256,
        "subgroups": {name: {} for name in module.SUBGROUPS},
        "consumer_trained": False,
        "test_evaluated": False,
    }
    gate = {
        "gate_id": "mask_bag_selector_arm_gate_v1",
        "status": "FAIL",
        "consumer_authorized": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    paired = {
        "replicates": 10000,
        "method": "paired complete-group bootstrap",
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "gate_decision.json").write_text(json.dumps(gate), encoding="utf-8")
    (root / "paired_comparison.json").write_text(json.dumps(paired), encoding="utf-8")
    (root / "per_image.csv").write_text("header\n", encoding="utf-8")
    hashes = {
        name: module.sha256_file(root / name)
        for name in (
            "summary.json",
            "gate_decision.json",
            "paired_comparison.json",
            "per_image.csv",
        )
    }
    audit = {
        "arm_prediction_freeze_sha256": "a" * 64,
        "cohort": module.EXPECTED_COHORT,
        "bootstrap_replicates": 10000,
        "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
        "output_hashes": hashes,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    audit_path = root / "evaluation_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    result = module._verify_evaluation(
        root,
        module.sha256_file(audit_path),
        expected_arm_freeze_sha256="a" * 64,
    )
    assert result["gate"]["status"] == "FAIL"
    (root / "per_image.csv").write_text("changed\n", encoding="utf-8")
    try:
        module._verify_evaluation(
            root,
            module.sha256_file(audit_path),
            expected_arm_freeze_sha256="a" * 64,
        )
    except ValueError as error:
        assert "output hash mismatch" in str(error)
    else:
        raise AssertionError("mutated evaluation output was accepted")
