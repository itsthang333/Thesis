from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import audit_mask_bag_normal_only_direct_anomaly_n1_output as auditor


def _cache_record(path: Path, families: list[int], offset: int) -> None:
    count = len(families)
    original = np.zeros((count, auditor.EXPECTED_DESCRIPTOR_DIM), dtype=np.float16)
    flipped = np.zeros_like(original)
    for index in range(count):
        original[index, (offset + index) % original.shape[1]] = 1.0
        flipped[index, (offset + index + 7) % original.shape[1]] = 1.0
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int32),
        descriptors=original,
        flipped_descriptors=flipped,
        candidate_indices=np.arange(count, dtype=np.int32),
        family_ids=np.asarray(families, dtype=np.int32),
        fallback_flags=np.zeros(count, dtype=np.uint8),
        packed_masks_included=np.asarray(0, dtype=np.uint8),
    )


def test_independent_hierarchical_weights_match_image_family_candidate_view_contract(
    tmp_path: Path,
) -> None:
    _cache_record(tmp_path / "one.npz", [0, 0, 1], 0)
    _cache_record(tmp_path / "two.npz", [0], 20)
    rows = [{"image_id": "one"}, {"image_id": "two"}]
    inventory = {
        "one": {"cache_path": "one.npz"},
        "two": {"cache_path": "two.npz"},
    }
    values, weights, audit = auditor._normal_training_arrays(
        tmp_path, rows, inventory
    )
    assert values.shape == (8, auditor.EXPECTED_DESCRIPTOR_DIM)
    assert np.array_equal(
        weights,
        np.asarray([0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125, 0.25, 0.25]),
    )
    assert audit == {
        "normal_images": 2,
        "normal_candidates": 4,
        "normal_candidate_views": 8,
        "descriptor_dimension": auditor.EXPECTED_DESCRIPTOR_DIM,
        "weight_sum": 1.0,
    }


def test_independent_spherical_bank_is_seed_deterministic() -> None:
    values = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    weights = np.full(4, 0.25, dtype=np.float64)
    first, first_assignments = auditor._fit_spherical_bank(
        values, weights, prototype_count=2, seed=42
    )
    second, second_assignments = auditor._fit_spherical_bank(
        values, weights, prototype_count=2, seed=42
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_assignments, second_assignments)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, rtol=0.0, atol=1.0e-6)
    assert set(first_assignments.tolist()) == {0, 1}


def test_independent_view_score_and_packed_map_arithmetic() -> None:
    prototypes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    original = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    flipped = np.asarray([[0.8, 0.2], [0.0, -1.0]], dtype=np.float32)
    original_distance, flipped_distance, scores = auditor._score_views(
        original, flipped, prototypes
    )
    assert np.array_equal(scores, (0.5 * (original_distance + flipped_distance)).astype(np.float32))
    assert int(np.argmax(scores)) == 1

    masks = np.asarray([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.uint8)
    packed = np.packbits(masks.reshape(2, -1), axis=1)
    restored = auditor._unpack_masks(packed, count=2, height=2, width=2)
    assert np.array_equal(restored, masks)


def test_candidate_score_payload_schema_is_fail_closed(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    np.savez_compressed(
        valid,
        schema_version=np.asarray(1, dtype=np.int32),
        candidate_indices=np.asarray([2, 5], dtype=np.int64),
        candidate_logits=np.asarray([0.1, 0.2], dtype=np.float32),
    )
    indices, scores = auditor._load_score_payload(valid)
    assert indices.tolist() == [2, 5]
    assert scores.dtype == np.float32

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid,
        candidate_indices=np.asarray([2, 5], dtype=np.int64),
        candidate_logits=np.asarray([0.1, 0.2], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="schema"):
        auditor._load_score_payload(invalid)


def test_binding_checks_protocol_and_auditor_hash(tmp_path: Path) -> None:
    protocol = {"canonical_lf_source_hashes": {"project/science.py": "1" * 64}}
    binding = {
        "schema_version": 1,
        "experiment_id": auditor.EXPERIMENT_ID,
        "kernel": auditor.KERNEL,
        "kernel_version": 1,
        "scientific_source_commit": auditor.SOURCE_COMMIT,
        "protocol_sha256": auditor.PROTOCOL_SHA256,
        "source_hashes": protocol["canonical_lf_source_hashes"],
        "checkout_commit": "2" * 40,
        "bound_wrapper_sha256": "3" * 64,
        "independent_auditor_sha256": auditor.sha256_file(Path(auditor.__file__)),
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    assert auditor._verify_binding(path, protocol)["kernel_version"] == 1
    binding["kernel"] = "wrong"
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        auditor._verify_binding(path, protocol)


def test_auditor_source_has_no_scientific_or_evaluator_import() -> None:
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    assert "from run_mask_bag_normal_only_direct_anomaly_n1" not in source
    assert "from models.mask_bag_normal_anomaly" not in source
    assert "from evaluate_mask_bag_selector_arm" not in source


def test_frozen_protocol_hash_and_scientific_sources_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "artifacts"
        / "research_protocols"
        / "rad_dino_mask_bag_normal_only_direct_anomaly_n1_v1.json"
    )
    assert auditor.sha256_file(protocol_path) == auditor.PROTOCOL_SHA256
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["experiment_id"] == auditor.EXPERIMENT_ID
    assert protocol["coordination"]["scientific_source_commit"] == auditor.SOURCE_COMMIT
    for relative, expected in protocol["canonical_lf_source_hashes"].items():
        assert auditor.sha256_file(root / relative) == expected
