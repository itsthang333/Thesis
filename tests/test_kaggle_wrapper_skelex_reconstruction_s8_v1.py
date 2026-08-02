from __future__ import annotations

from pathlib import Path


WRAPPER = Path("project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_v1.py")


def test_s8_wrapper_is_hash_bound_and_fail_closed_before_launch_binding() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert 'SOURCE_COMMIT = "b4543aeb9430345c9b789384943bd218816a85dd"' in source
    assert 'PROTOCOL_SHA256 = "7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07"' in source
    assert 'EXPECTED_TRANSFORMERS_VERSION = "4.50.2"' in source
    assert '"model.safetensors": "81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8"' in source
    assert '"--maximum-candidates", "81"' in source
    assert '"--seed", "42"' in source
    main = source[source.index("def main() -> None:") :]
    assert main.index("clone_and_verify()") < main.index("verify_t4x2()")
    assert main.index("verify_t4x2()") < main.index("install_runtime()")
    assert main.index("install_runtime()") < main.index("run_static_tests()")
    assert main.index("run_static_tests()") < main.index("download_skelex()")
    assert main.index("download_skelex()") < main.index("find_dataset_root()")
    assert main.index("find_dataset_root()") < main.index("write_binding(source_hashes)")


def test_s8_wrapper_runs_independent_audit_before_wrapper_audit() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    main = source[source.index("def main() -> None:") :]
    assert main.index('"--audit-output"') < main.index("audit_wrapper_output(")
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
