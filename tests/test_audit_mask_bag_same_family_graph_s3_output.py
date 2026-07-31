from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_same_family_graph_s3_output.py"


def test_s3_auditor_is_gt_blind_and_pins_physical_helper() -> None:
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


def test_s3_auditor_recomputes_every_gt_blind_graph_gate() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "pregraph_identity_audit.csv",
        "gt_blind_diagnostics.csv",
        "absolute_candidate_count_probability_spearman",
        "COUNT_SPEARMAN_CEILING = 0.5013777759365411",
        'binary_sums["view_swap_exact"] == expected_validation',
        'binary_sums["alpha_zero_identity_exact"] == expected_validation',
        'binary_sums["graph_symmetric"] == expected_validation',
        "cross_family_edges == 0",
        "non_self_edges > 0",
        'binary_sums["isolated_logits_exact"] == expected_validation',
        "gt_blind_gate_pass",
    ):
        assert required in source


def test_s3_auditor_requires_exact_operator_runtime_and_binding() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        '"minimum_iou": 0.25',
        '"minimum_containment": 0.5',
        '"alpha": 0.5',
        '"iterations": 10',
        'binding.get("status") != "FROZEN_PRELAUNCH"',
        'runtime.get("validation_shards") != [186, 185]',
        'not all("T4" in name for name in runtime["cuda_device_names"])',
        "physical_cache_records_verified",
        "physical_pregraph_identity_rows_verified",
    ):
        assert required in source
