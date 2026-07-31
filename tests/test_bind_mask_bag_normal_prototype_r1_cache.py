from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "bind_mask_bag_normal_prototype_r1_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r1_cache_binder", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_contract(tmp_path: Path):
    module = _load_module()
    template = tmp_path / "template.py"
    template.write_text(
        "\n".join(
            [
                f'SCIENTIFIC_SOURCE_COMMIT = "{module.R1_SOURCE_COMMIT}"',
                f'PROTOCOL_SHA256 = "{module.R1_PROTOCOL_SHA256}"',
                f'CACHE_PROTOCOL_SHA256 = "{module.CORRECTED_CACHE_PROTOCOL_SHA256}"',
                f'CACHE_SCIENTIFIC_SOURCE_COMMIT = "{module.CORRECTED_CACHE_SOURCE_COMMIT}"',
                "CACHE_BINDING_READY = False",
                f'CACHE_FREEZE_SHA256 = "{module.PENDING}"',
                f'CACHE_WRAPPER_AUDIT_SHA256 = "{module.PENDING}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    freeze = {
        "source_commit": module.CORRECTED_CACHE_SOURCE_COMMIT,
        "protocol_sha256": module.CORRECTED_CACHE_PROTOCOL_SHA256,
        "cohort": {"train": 2981, "validation": 371},
        "validation_selected_indices_reproduced": 371,
        "validation_map_hashes_reproduced": 371,
        "train_masks_discarded": True,
        "validation_masks_bitpacked": True,
        "affinity_features_cached": True,
        "affinity_feature_dim": 24,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = cache_root / "selector_cache_freeze.json"
    _write_json(freeze_path, freeze)
    wrapper_audit = {
        "scientific_source_commit": module.CORRECTED_CACHE_SOURCE_COMMIT,
        "protocol_sha256": module.CORRECTED_CACHE_PROTOCOL_SHA256,
        "selector_cache_freeze_sha256": _sha256(freeze_path),
        "physical_cache_records_verified": 3352,
        "cohort": {"train": 2981, "val": 371},
        "validation_selected_indices_reproduced": 371,
        "validation_map_hashes_reproduced": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    wrapper_audit_path = cache_root / "wrapper_output_audit.json"
    _write_json(wrapper_audit_path, wrapper_audit)
    independent = {
        "audit_id": "independent_mask_bag_selector_cache_output_v1",
        "cache_freeze_sha256": _sha256(freeze_path),
        "wrapper_output_audit_sha256": _sha256(wrapper_audit_path),
        "source_commit": module.CORRECTED_CACHE_SOURCE_COMMIT,
        "protocol_sha256": module.CORRECTED_CACHE_PROTOCOL_SHA256,
        "physical_cache_records_verified": 3352,
        "validation_packed_mask_geometry_records_verified": 371,
        "cohort": {"train": 2981, "val": 371},
        "baseline_reproduction": {
            "validation_images": 371,
            "selected_indices_exact": 371,
            "map_hashes_exact": 371,
        },
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    independent_path = tmp_path / "independent_audit.json"
    _write_json(independent_path, independent)
    return module, template, cache_root, independent_path


def test_binder_surface_is_gt_training_and_test_dataset_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "annotation_name",
        'split="test"',
        "optimizer",
        ".backward(",
    ):
        assert forbidden not in lowered


def test_binding_changes_exactly_three_constants(tmp_path: Path) -> None:
    module, template, cache_root, independent = _synthetic_contract(tmp_path)
    output_wrapper = tmp_path / "bound" / "wrapper.py"
    output_audit = tmp_path / "bound" / "binding_audit.json"
    audit = module.bind_r1_cache(
        template_path=template,
        cache_root=cache_root,
        independent_audit_path=independent,
        expected_independent_audit_sha256=_sha256(independent),
        output_wrapper_path=output_wrapper,
        output_audit_path=output_audit,
        expected_template_sha256=_sha256(template),
    )
    assert audit["status"] == "BOUND_NOT_LAUNCHED"
    assert audit["binder_source_sha256"] == _sha256(SOURCE)
    assert audit["exact_wrapper_replacements"] == 3
    assert audit["inverse_byte_reconstruction_sha256"] == _sha256(template)
    assert audit["kernel_launched"] is False
    assignments = module._literal_assignments(output_wrapper.read_text(encoding="utf-8"))
    assert assignments["CACHE_BINDING_READY"] is True
    assert assignments["CACHE_FREEZE_SHA256"] == _sha256(
        cache_root / "selector_cache_freeze.json"
    )
    assert assignments["CACHE_WRAPPER_AUDIT_SHA256"] == _sha256(
        cache_root / "wrapper_output_audit.json"
    )


def test_binding_rejects_same_wrapper_and_audit_output(tmp_path: Path) -> None:
    module, template, cache_root, independent = _synthetic_contract(tmp_path)
    output = tmp_path / "bound.py"
    with pytest.raises(ValueError, match="outputs must differ"):
        module.bind_r1_cache(
            template_path=template,
            cache_root=cache_root,
            independent_audit_path=independent,
            expected_independent_audit_sha256=_sha256(independent),
            output_wrapper_path=output,
            output_audit_path=output,
            expected_template_sha256=_sha256(template),
        )


def test_binding_rejects_tampered_independent_audit(tmp_path: Path) -> None:
    module, template, cache_root, independent = _synthetic_contract(tmp_path)
    accepted_hash = _sha256(independent)
    payload = json.loads(independent.read_text(encoding="utf-8"))
    payload["physical_cache_records_verified"] = 3351
    _write_json(independent, payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.bind_r1_cache(
            template_path=template,
            cache_root=cache_root,
            independent_audit_path=independent,
            expected_independent_audit_sha256=accepted_hash,
            output_wrapper_path=tmp_path / "bound.py",
            output_audit_path=tmp_path / "audit.json",
            expected_template_sha256=_sha256(template),
        )


def test_binding_rejects_unsafe_cache_freeze(tmp_path: Path) -> None:
    module, template, cache_root, independent = _synthetic_contract(tmp_path)
    freeze_path = cache_root / "selector_cache_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["validation_gt_read"] = True
    _write_json(freeze_path, freeze)
    with pytest.raises(ValueError, match="audit contract mismatch"):
        module.bind_r1_cache(
            template_path=template,
            cache_root=cache_root,
            independent_audit_path=independent,
            expected_independent_audit_sha256=_sha256(independent),
            output_wrapper_path=tmp_path / "bound.py",
            output_audit_path=tmp_path / "audit.json",
            expected_template_sha256=_sha256(template),
        )
