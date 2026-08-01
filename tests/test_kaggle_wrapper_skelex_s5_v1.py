from __future__ import annotations

from pathlib import Path
import re


WRAPPER = Path("project/kaggle_wrappers/run_skelex_mask_bag_selector_s5_v1.py")


def test_s5_v4_pins_correction_chain_and_tests_before_large_download() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert 'EXPECTED_TRANSFORMERS_VERSION = "4.50.2"' in source
    assert (
        'CORRECTION_SOURCE_COMMIT = "664578758225501dc163a6fc35d9ecdb9a1947d7"'
        in source
    )
    assert (
        'NUMERIC_ADDENDUM_SHA256 = '
        '"ded254883a13da9ec0b961970ebacbd2b61badd04c644b7b9c64747a6abd2f72"'
        in source
    )
    assert (
        'TEST_CONTRACT_CORRECTION_SOURCE_COMMIT = '
        '"f9e56111ddf98b474c3ea1532c2da77b68e90232"'
        in source
    )
    assert (
        'TEST_CONTRACT_ADDENDUM_SHA256 = '
        '"591858f1e5bfaefad55b9583f3904ae6447bc98986a77d7a52471a49586b74a8"'
        in source
    )
    assert (
        'AUDITOR_SHA256 = '
        '"dbf84451ae32b5fd819af53c48f3357da0c236defdaa5eda2d1b787640e01049"'
        in source
    )
    for relative in (
        "project/models/skelex_mask_bag_descriptor.py",
        "project/run_skelex_mask_bag_selector_s5.py",
        "project/audit_skelex_mask_bag_selector_s5_output.py",
        "tests/test_skelex_mask_bag_descriptor.py",
    ):
        assert f'"{relative}"' in source
    assert 'f"transformers=={EXPECTED_TRANSFORMERS_VERSION}"' in source
    assert '"TOKENIZERS_PARALLELISM": "false"' in source

    main = source[source.index("def main() -> None:") :]
    assert main.index("verify_t4x2()") < main.index("install_runtime()")
    assert main.index("install_runtime()") < main.index("run_static_tests()")
    assert main.index("run_static_tests()") < main.index("download_skelex()")
    assert main.index("download_skelex()") < main.index("prepare_split()")

    clone = source[source.index("def clone_and_verify()") : source.index("def verify_t4x2()")]
    assert clone.index("numeric correction addendum hash mismatch") < clone.index(
        "numeric correction addendum provenance mismatch"
    )
    assert clone.index("numeric correction addendum provenance mismatch") < clone.index(
        "test-contract correction addendum provenance mismatch"
    )
    assert clone.index("test-contract correction addendum provenance mismatch") < clone.index(
        "verified_hashes.update(TEST_CONTRACT_SOURCE_OVERRIDE)"
    )
    assert clone.index("verified_hashes.update(TEST_CONTRACT_SOURCE_OVERRIDE)") < clone.index(
        "for relative, expected in verified_hashes.items()"
    )


def test_s5_packaged_wrapper_matches_canonical_source() -> None:
    packaged = Path(
        "tmp/kaggle/skelex_mask_bag_selector_s5_v1/"
        "btxrd-skelex-mask-bag-selector-s5-v1.py"
    )
    if packaged.is_file():
        def normalize_binding(text: str) -> str:
            text = re.sub(r"^KERNEL_VERSION = .*$", "KERNEL_VERSION = <BOUND>", text, flags=re.M)
            text = re.sub(
                r"^LAUNCH_BINDING_READY = .*$",
                "LAUNCH_BINDING_READY = <BOUND>",
                text,
                flags=re.M,
            )
            return re.sub(
                r'^CHECKOUT_COMMIT = ".*"$',
                'CHECKOUT_COMMIT = "<BOUND>"',
                text,
                flags=re.M,
            )

        canonical = normalize_binding(WRAPPER.read_text(encoding="utf-8"))
        packaged_source = normalize_binding(packaged.read_text(encoding="utf-8"))
        assert packaged_source == canonical
