from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "project"
    / "kaggle_wrappers"
    / "run_mask_bag_label_granularity_s6_v1.py"
)


def test_s6_wrapper_is_unbound_and_prediction_first() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"KERNEL_VERSION", "LAUNCH_BINDING_READY", "CHECKOUT_COMMIT"}
    }
    assert assignments == {
        "KERNEL_VERSION": 0,
        "LAUNCH_BINDING_READY": False,
        "CHECKOUT_COMMIT": "UNBOUND",
    }
    source = SOURCE.read_text(encoding="utf-8")
    assert source.rindex("run_mask_bag_label_granularity_s6_pair.py") < source.rindex(
        "audit_mask_bag_label_granularity_s6_output.py"
    )
    assert "evaluate_mask_bag_selector_arm.py" not in source
    assert "compare_mask_bag_evaluated_arms.py" not in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s6_wrapper_requires_exact_inputs_and_two_t4s() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for value in (
        "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c",
        "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c",
        "8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e",
        "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069",
        "b0dca40bf4f8bd933a902facb7bfdf5ec393c429672b0beb0b0594f2d15dfc63",
    ):
        assert value in source
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in names)' in source
    assert "physical_candidate_score_payloads_verified" in source
    assert "physical_prediction_maps_verified" in source
