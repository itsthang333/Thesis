from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "project/kaggle_wrappers/run_same_gallery_class_contrast_bas_b4_v1.py"
BINDER = ROOT / "project/bind_same_gallery_class_contrast_bas_b4_wrapper.py"


def _load_binder():
    spec = importlib.util.spec_from_file_location("b4_binder", BINDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b4_wrapper_is_unbound_and_annotation_free() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    assert "BTXRD test" not in source
    assert "Annotations" not in source
    assert "annotation_sha256" not in source
    assert 'str(SOURCE / "project/evaluate_mask_bag_selector_arm.py")' not in source
    assert 'str(SOURCE / "project/compare_mask_bag_evaluated_arms.py")' not in source


def test_b4_wrapper_orders_hardware_before_input_and_audit_after_runner() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    main = source[source.index("def main()") :]
    assert main.index("verify_t4x2()") < main.index("prepare_split()")
    assert main.index("find_dataset_root()") < main.index("run(\n            [")
    assert main.index("run_same_gallery_bas_semantic_b4.py") < main.index(
        "str(SOURCE / AUDITOR_RELATIVE)"
    )
    assert main.index("str(SOURCE / AUDITOR_RELATIVE)") < main.index("audit_output(")


def test_b4_binder_rejects_nonexistent_checkout(tmp_path: Path) -> None:
    binder = _load_binder()
    with pytest.raises(Exception):
        binder.bind(
            WRAPPER,
            tmp_path / "bound.py",
            tmp_path / "binding.json",
            repository_root=ROOT,
            checkout_commit="0" * 40,
            kernel_version=1,
        )
