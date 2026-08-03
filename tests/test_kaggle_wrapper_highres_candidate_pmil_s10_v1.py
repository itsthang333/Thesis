from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import project.kaggle_wrappers.run_highres_candidate_pmil_s10_v1 as wrapper


def test_wrapper_is_unbound_and_targets_private_s10_kernel() -> None:
    assert wrapper.KERNEL == "itsthang333/btxrd-highres-candidate-pmil-s10-v1"
    assert wrapper.KERNEL_VERSION == 0
    assert wrapper.LAUNCH_BINDING_READY is False
    assert wrapper.CHECKOUT_COMMIT == "UNBOUND"
    assert len(wrapper.PROTOCOL_SHA256) == 64
    assert len(wrapper.AUDITOR_SHA256) == 64


def test_wrapper_source_has_no_gt_or_test_dataset_argument() -> None:
    source = Path(wrapper.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "--gt-root" not in string_literals
    assert "--annotation-root" not in string_literals
    assert "--test-root" not in string_literals


def test_clone_refuses_unbound_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrapper, "SOURCE", tmp_path / "source")
    with pytest.raises(RuntimeError, match="launch binding"):
        wrapper.clone_and_verify()


def test_protocol_contract_matches_wrapper_constants() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / wrapper.PROTOCOL_RELATIVE
    assert wrapper.hash_file(protocol_path) == wrapper.PROTOCOL_SHA256
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["scientific_source"]["commit"] == wrapper.SOURCE_COMMIT
    assert protocol["representation"]["input_size"] == 640
    assert protocol["training"]["epochs"] == 32
    assert protocol["finite_arms"] == {
        "control": "geometry_v3_plus_upstream_control",
        "capacity": "control_plus_s10_identity_capacity",
        "primary": "s10_pareto_identity_capture_purity",
        "control_recipe": "equal tie-aware within-image ranks of Geometry-v3 and upstream selection score",
        "capacity_recipe": "equal tie-aware within-image ranks of Geometry-v3, upstream, and S10 identity",
        "primary_recipe": "relative to control, switch only to a candidate weakly no worse in identity/capture/purity ranks and strictly better in at least one; maximize minimum component rank, then identity rank, then smallest immutable index",
        "no_dominator": "retain exact control winner",
        "bag_probability": "exact accepted Geometry-v3 bag probability for all arms",
        "dense_evidence_as_final_mask": False,
        "weight_threshold_extent_subgroup_alternatives": False,
    }


def test_resnet_public_weight_identity_is_frozen() -> None:
    assert wrapper.RESNET_URL.startswith("https://download.pytorch.org/models/")
    assert wrapper.RESNET_SHA256 == (
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    )
    assert wrapper.RESNET_BYTES == 102_540_417
