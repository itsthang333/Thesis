from __future__ import annotations

from pathlib import Path


WRAPPER = Path("project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_audit_v1.py")


def test_s8_audit_only_wrapper_is_unbound_and_preserves_prediction() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert 'AUDITOR_CORRECTION_COMMIT = "969327c4fbbd635fff2e3a00d34d533af8a3c340"' in source
    assert 'CORRECTION_SHA256 = "94e5881f763cc2cb3bd0a3f49cb563f2449140a7c576211252a45579597fc8a2"' in source
    assert 'PAIR_FREEZE_SHA256 = "b2cfd59fb01046f445d098790efa5a0fdc649bbc80f565439ba51c5cd453fa00"' in source
    main = source[source.index("def main() -> None:") :]
    assert main.index("clone_and_verify()") < main.index("verify_t4x2()")
    assert main.index("verify_t4x2()") < main.index("run_static_tests()")
    assert main.index("run_static_tests()") < main.index("find_and_verify_producer_output()")
    assert main.index("find_and_verify_producer_output()") < main.index('"--output-root"')
    assert '"prediction_changed": False' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
