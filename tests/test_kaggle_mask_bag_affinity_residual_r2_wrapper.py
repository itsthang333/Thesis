from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "project/kaggle_wrappers/run_mask_bag_affinity_residual_r2_v1.py"


def test_r2_wrapper_is_fail_closed_until_prelaunch_binding() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert source.index("if not LAUNCH_BINDING_READY") < source.index("git\", \"clone")
    assert '"bound_wrapper_sha256": canonical_hash(Path(__file__))' in source
    assert "BOUND_WRAPPER_SHA256" not in source


def test_r2_wrapper_orders_all_gates_before_fit_and_freeze() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    main_source = source[source.index("def main() -> None:") :]
    positions = [
        main_source.index("source_hashes = clone_and_verify()"),
        main_source.index("install_runtime()"),
        main_source.index("t4 = verify_t4x2()"),
        main_source.index("split = prepare_split()"),
        main_source.index("baseline_root, baseline = prepare_baseline()"),
        main_source.index("cache_root, cache = find_cache()"),
        main_source.index('"tests/test_run_mask_bag_affinity_residual_arm.py"'),
        main_source.index('"project/run_mask_bag_affinity_residual_arm.py"'),
        main_source.index("audit_output(source_hashes, cache, baseline, t4)"),
    ]
    assert positions == sorted(positions)


def test_r2_wrapper_keeps_gt_consumer_and_test_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "evaluate_mask_bag_selector_arm" not in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
    assert "square_corrected_baseline.zip.bin" in source
