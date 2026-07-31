from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project/kaggle_wrappers/run_mask_bag_family_balanced_s1_pair_v1.py"


def test_s1_wrapper_is_fail_closed_until_prelaunch_binding() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert source.index("if not LAUNCH_BINDING_READY") < source.index('"git",\n            "clone"')
    assert '"bound_wrapper_sha256": canonical_hash(Path(__file__))' in source
    assert "BOUND_WRAPPER_SHA256" not in source


def test_s1_wrapper_orders_all_gates_before_pair_fit() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    main = source[source.index("def main() -> None:") :]
    positions = [
        main.index("source_hashes = clone_and_verify()"),
        main.index("install_runtime()"),
        main.index("t4 = verify_t4x2()"),
        main.index("split = prepare_split()"),
        main.index("baseline_root, baseline = prepare_baseline()"),
        main.index("cache_root, cache = find_cache()"),
        main.index('"tests/test_run_mask_bag_family_balanced_pair.py"'),
        main.index('"project/run_mask_bag_family_balanced_pair.py"'),
        main.index("audit_output(source_hashes, cache, baseline, t4)"),
    ]
    assert positions == sorted(positions)


def test_s1_wrapper_freezes_and_audits_matched_pair_without_gt() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "evaluate_mask_bag_selector_arm" not in source
    assert 'POOL_MODES = ("standard", "family_balanced")' in source
    assert '"physical_prediction_maps_verified": 742' in source
    assert '"physical_candidate_score_payloads_verified": 742' in source
    assert '"physical_candidate_family_payloads_verified": 371' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s1_wrapper_removes_redundant_checkout_after_output_audit() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    audit = source.index("audit_output(source_hashes, cache, baseline, t4)")
    cleanup_source = source.index("shutil.rmtree(SOURCE)")
    cleanup_runtime = source.index("shutil.rmtree(RUNTIME)")
    assert audit < cleanup_source < cleanup_runtime
