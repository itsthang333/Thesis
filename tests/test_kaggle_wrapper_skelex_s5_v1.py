from __future__ import annotations

from pathlib import Path
import re


WRAPPER = Path("project/kaggle_wrappers/run_skelex_mask_bag_selector_s5_v1.py")


def test_s5_v2_pins_transformers_before_tests_and_large_download() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert 'EXPECTED_TRANSFORMERS_VERSION = "4.50.2"' in source
    assert 'f"transformers=={EXPECTED_TRANSFORMERS_VERSION}"' in source
    assert '"TOKENIZERS_PARALLELISM": "false"' in source

    main = source[source.index("def main() -> None:") :]
    assert main.index("verify_t4x2()") < main.index("install_runtime()")
    assert main.index("install_runtime()") < main.index("run_static_tests()")
    assert main.index("run_static_tests()") < main.index("download_skelex()")
    assert main.index("download_skelex()") < main.index("prepare_split()")


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
