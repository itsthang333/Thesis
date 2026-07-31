from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_critical_relation_r3_output.py"


def test_r3_auditor_is_gt_blind_and_pins_physical_helper() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "candidate_quality",
        "size_group",
        "oracle_best",
        "ground_truth",
    ):
        assert forbidden not in lowered
    assert "PHYSICAL_HELPER_SHA256" in source
    assert "_verify_validation_evidence" in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_r3_auditor_recomputes_every_predeclared_gt_blind_gate() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "gt_blind_diagnostics.csv",
        "absolute_candidate_count_probability_spearman",
        "COUNT_SPEARMAN_CEILING = 0.5013777759365411",
        "base_flip_critical_agreement",
        "final_flip_selected_agreement",
        "minimum = base_agreement - 0.01",
        "gt_blind_gate_pass",
    ):
        assert required in source


def test_r3_auditor_requires_exact_fit_runtime_and_launch_binding() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        '"epochs": 16',
        '"batch_size": 16',
        '"instance_loss_weight": 0.25',
        '"instance_warmup_epochs": 2',
        'binding.get("status") != "FROZEN_PRELAUNCH"',
        'runtime.get("validation_shards") != [186, 185]',
        'not all("T4" in name for name in runtime["cuda_device_names"])',
        "physical_cache_records_verified",
    ):
        assert required in source
