from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "project/kaggle_wrappers/run_mask_bag_global_local_instance_s7_v1.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("s7_wrapper_template", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s7_wrapper_template_is_unbound_and_exactly_scoped() -> None:
    module = _module()
    assert module.KERNEL == "itsthang333/btxrd-rad-dino-mask-bag-instance-s7-v1"
    assert module.KERNEL_VERSION == 0
    assert module.LAUNCH_BINDING_READY is False
    assert module.CHECKOUT_COMMIT == "UNBOUND"
    assert module.SOURCE_COMMIT == "0e524807937e6fb6effde1649993825f3923c43f"
    assert module.PROTOCOL_SHA256 == (
        "81fbb2f40af3a49e4653a15d298858c973e88524dea06fc42c9095cec55579a1"
    )
    assert module.CACHE_FREEZE_SHA256 == (
        "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
    )
    assert module.BASELINE["checkpoint"] == (
        "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
    )


def test_s7_wrapper_runs_gt_blind_producer_and_independent_auditor_only() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "run_mask_bag_global_local_instance_s7_pair.py" in source
    assert "audit_mask_bag_global_local_instance_s7_output.py" in source
    assert "evaluate_mask_bag_selector_arm" not in source
    assert "compare_mask_bag_evaluated_arms" not in source
    assert "validation_gt_read\": False" in source
    assert "accepted_bag_probability_preserved\": True" in source
    assert "target_snapshot_manifest.json" in source
    assert "physical_candidate_score_payloads_verified" in source
    assert "physical_prediction_maps_verified" in source
