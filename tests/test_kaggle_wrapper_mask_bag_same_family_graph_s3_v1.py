from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "project"
    / "kaggle_wrappers"
    / "run_mask_bag_same_family_graph_s3_v1.py"
)


def test_s3_wrapper_is_fail_closed_and_gt_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert "if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1" in source
    lowered = source.lower()
    for forbidden in (
        "segmentation_dataset",
        "candidate_quality",
        "size_group",
        "oracle_best",
    ):
        assert forbidden not in lowered
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s3_wrapper_binds_t4x2_transport_and_all_outputs() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "square_corrected_baseline.zip.bin",
        "selector_cache_freeze.json",
        "torch.cuda.device_count() != 2",
        'all("T4" in name for name in names)',
        "pregraph_identity_audit.csv",
        "gt_blind_diagnostics.csv",
        '"physical_prediction_maps_verified": 371',
        '"physical_candidate_score_payloads_verified": 371',
        '"physical_pregraph_identity_rows_verified": 371',
        '"physical_gt_blind_diagnostic_rows_verified": 371',
    ):
        assert required in source

