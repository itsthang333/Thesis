from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_affinity_residual_r2_output.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r2_output_auditor", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r2_auditor_is_gt_blind_and_has_no_evaluator_import() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "evaluate_mask_bag_selector_arm" not in source
    assert "BTXRD" not in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_r2_auditor_pins_protocol_cache_and_physical_helper() -> None:
    module = _load_module()
    assert module.PROTOCOL_SHA256 == "3f28cc7187ad64f3755ae4c7a10bb380a0085d1733807dcf667c44d92d9f593d"
    assert module.CACHE_FREEZE_SHA256 == "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
    module._verify_helper_source()


def test_r2_history_requires_exact_ordered_fixed_epochs(tmp_path: Path) -> None:
    module = _load_module()
    history = tmp_path / "history.json"
    history.write_text(
        __import__("json").dumps(
            [{"epoch": epoch, "loss": 1.0 / epoch} for epoch in range(1, 17)]
        ),
        encoding="utf-8",
    )
    module._verify_history(history)
    history.write_text('[{"epoch": 1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 16"):
        module._verify_history(history)


def test_r2_launch_binding_must_be_prelaunch_and_protocol_frozen() -> None:
    module = _load_module()
    protocol = {
        "canonical_lf_source_hashes": {
            path: "a" * 64 for path in module.REQUIRED_RUNTIME_SOURCES
        }
    }
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "protocol_sha256": module.PROTOCOL_SHA256,
        "scientific_source_commit": module.SOURCE_COMMIT,
        "kernel": "owner/kernel",
        "kernel_version": 1,
        "checkout_commit": "b" * 40,
        "bound_wrapper_sha256": "c" * 64,
        "runtime_source_hashes": {
            path: "a" * 64 for path in module.REQUIRED_RUNTIME_SOURCES
        },
    }
    module._verify_launch_binding(binding, protocol)
    binding["status"] = "DRAFT"
    with pytest.raises(ValueError, match="binding contract"):
        module._verify_launch_binding(binding, protocol)
