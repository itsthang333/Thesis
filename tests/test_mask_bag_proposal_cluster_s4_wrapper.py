from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "project"
    / "kaggle_wrappers"
    / "run_mask_bag_proposal_cluster_s4_v1.py"
)


def test_s4_wrapper_is_unbound_fail_closed_and_gt_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    assert "kernel_version = 0" in lowered
    assert "launch_binding_ready = false" in lowered
    assert 'checkout_commit = "unbound"' in lowered
    assert "if not launch_binding_ready or kernel_version < 1" in lowered
    for forbidden in (
        "datasets.factory",
        "build_segmentation_dataset",
        "mask_tensor",
        "size_group",
        'split="test"',
    ):
        assert forbidden not in lowered
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s4_wrapper_binds_cache_baseline_t4x2_and_physical_evidence() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "CACHE_FREEZE_SHA256" in source
    assert "CACHE_WRAPPER_AUDIT_SHA256" in source
    assert "BASELINE_ARCHIVE_SHA256" in source
    assert "TRANSPORT_AUDIT_SHA256" in source
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in names)' in source
    for expected in (
        '"physical_oof_score_payloads_verified": 2981',
        '"physical_train_cluster_payloads_verified": 2981',
        '"physical_validation_teacher_score_payloads_verified": 371',
        '"physical_validation_cluster_payloads_verified": 371',
        '"physical_validation_residual_payloads_verified": 371',
        '"physical_candidate_score_payloads_verified": 371',
        '"physical_prediction_maps_verified": 371',
    ):
        assert expected in source


def test_s4_wrapper_runs_tests_before_scientific_runner_and_cleans_exact_paths() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    main = source[source.index("def main()") :]
    focused = main.index('"tests/test_mask_bag_proposal_clusters.py"')
    full = main.index('run([sys.executable, "-m", "pytest", "-q"]')
    runner = main.index('"project/run_mask_bag_proposal_cluster_s4_arm.py"')
    audit = main.index("audit_output(source_hashes, cache, baseline, t4)")
    cleanup = main.index("for cleanup_path in (SOURCE, RUNTIME)")
    assert focused < full < runner < audit < cleanup
    assert "cleanup_path.resolve().parent == WORK.resolve()" in source
