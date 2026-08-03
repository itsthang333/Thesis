from __future__ import annotations

from pathlib import Path


WRAPPER = Path(
    "project/kaggle_wrappers/run_skelex_candidate_marginal_s9_v1.py"
)


def test_s9_wrapper_is_hash_bound_and_fail_closed_before_launch_binding() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert 'SOURCE_COMMIT = "7dcd6c6f055c69f3f048a005ed2fea6177dc7ed8"' in source
    assert (
        'CORRECTION_SOURCE_COMMIT = "cb608cd8ca501e840d4ae7c73cc7592187683a27"'
        in source
    )
    assert (
        'PROTOCOL_SHA256 = "0a303c9c86c3c43c750c85a50087e792bf0942a0b43fc9a1cf9e143c4832ee3d"'
        in source
    )
    assert (
        'CORRECTION_SHA256 = "0ddf17d73c9ddcf24799827a075f41a32e671e15894ae3d6d0780a278edb11a9"'
        in source
    )
    assert (
        'AUDITOR_SHA256 = "9e665bdbf2dee5f487642f3844c656d4ff9814a2f9e89a51ba627f73cd55b30c"'
        in source
    )
    assert "tests/test_skelex_candidate_marginal_s9_rank_exactness_correction.py" in source
    assert 'EXPECTED_TRANSFORMERS_VERSION = "4.50.2"' in source
    assert (
        '"model.safetensors": '
        '"81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8"'
        in source
    )
    assert '"--encoder-batch-size", "2"' in source
    assert '"--train-batch-size", "8"' in source
    assert '"--epochs", "32"' in source
    assert '"--learning-rate", "0.001"' in source
    assert '"--weight-decay", "0.0001"' in source
    assert '"--maximum-candidates", "81"' in source
    assert '"--seed", "42"' in source


def test_s9_wrapper_checks_code_before_model_and_scientific_inputs() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    main = source[source.index("def main() -> None:") :]
    assert main.index("clone_and_verify()") < main.index("verify_t4x2()")
    assert main.index("verify_t4x2()") < main.index("install_runtime()")
    assert main.index("install_runtime()") < main.index("run_static_tests()")
    assert main.index("run_static_tests()") < main.index("download_skelex()")
    assert main.index("download_skelex()") < main.index("prepare_split()")
    assert main.index("prepare_split()") < main.index("find_dataset_root()")
    assert main.index("find_dataset_root()") < main.index("write_binding(source_hashes)")


def test_s9_wrapper_runs_independent_gt_blind_audit_before_wrapper_audit() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    main = source[source.index("def main() -> None:") :]
    assert main.index('"--audit-output"') < main.index("audit_wrapper_output(")
    assert "evaluate" not in main.lower()
    assert '"collaborator_output_accessed": False' in source
    assert '"annotation_paths_resolved": False' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s9_wrapper_cleans_transient_source_runtime_and_public_weights() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    main = source[source.index("def main() -> None:") :]
    assert "finally:" in main
    assert "for path in (SOURCE, RUNTIME):" in main
    assert "path.resolve().parent == WORK.resolve()" in main
    assert "shutil.rmtree(path)" in main
    assert "skelex_root" in main
