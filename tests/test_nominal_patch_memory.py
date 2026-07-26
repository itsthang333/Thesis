import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "project"
        / "models"
        / "nominal_patch_memory.py"
    )
    spec = importlib.util.spec_from_file_location("nominal_patch_memory_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MEMORY = _load_module()
FrozenNormalCalibration = MEMORY.FrozenNormalCalibration
blend_anomaly_scores = MEMORY.blend_anomaly_scores
cosine_anomaly_scores = MEMORY.cosine_anomaly_scores
flatten_context_patch_bank = MEMORY.flatten_context_patch_bank
make_seeded_random_projection = MEMORY.make_seeded_random_projection
project_features = MEMORY.project_features
projected_bank_size_bytes = MEMORY.projected_bank_size_bytes
projection_sha256 = MEMORY.projection_sha256
retrieve_normal_context = MEMORY.retrieve_normal_context
spatial_cosine_anomaly_scores = MEMORY.spatial_cosine_anomaly_scores


def test_retrieve_normal_context_is_deterministic_on_ties() -> None:
    bank = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    indices, similarities = retrieve_normal_context(
        np.array([1, 0], dtype=np.float32), bank, top_k=2
    )
    assert indices.tolist() == [0, 1]
    assert similarities.tolist() == [1.0, 1.0]


def test_random_projection_is_seeded_hash_locked_and_normalized() -> None:
    first = make_seeded_random_projection(input_dim=8, output_dim=4, seed=42)
    second = make_seeded_random_projection(input_dim=8, output_dim=4, seed=42)
    different = make_seeded_random_projection(input_dim=8, output_dim=4, seed=43)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert projection_sha256(first) == projection_sha256(second)
    features = np.eye(8, dtype=np.float32)[:3]
    projected = project_features(features, first)
    assert projected.shape == (3, 4)
    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0)


def test_projected_bank_size_is_explicit() -> None:
    assert projected_bank_size_bytes(
        images=1493,
        grid_height=28,
        grid_width=28,
        output_dim=128,
        bytes_per_value=2,
    ) == 299_651_072


def test_cosine_anomaly_score_detects_unseen_direction() -> None:
    memory = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    query = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    scores = cosine_anomaly_scores(
        query, memory, query_chunk_size=1, memory_chunk_size=1
    )
    assert np.allclose(scores, [0.0, 1.0])


def test_flatten_context_patch_bank_selects_images() -> None:
    bank = np.arange(3 * 2 * 2 * 2, dtype=np.float32).reshape(3, 2, 2, 2) + 1
    selected = flatten_context_patch_bank(bank, np.array([2, 0]))
    assert selected.shape == (8, 2)
    assert np.array_equal(selected[:4], bank[2].reshape(-1, 2))


def test_spatial_memory_does_not_match_remote_patch() -> None:
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[..., 0] = 1
    query[0, 0] = [0, 1]
    context = np.zeros((1, 2, 2, 2), dtype=np.float32)
    context[..., 0] = 1
    context[0, 1, 1] = [0, 1]
    radius_zero = spatial_cosine_anomaly_scores(query, context, radius=0)
    radius_one = spatial_cosine_anomaly_scores(query, context, radius=1)
    assert np.isclose(radius_zero[0, 0], 1.0)
    assert np.isclose(radius_one[0, 0], 0.0)


def test_frozen_normal_calibration_does_not_force_per_image_maximum() -> None:
    calibration = FrozenNormalCalibration.fit(
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    )
    low_map = calibration.transform(np.array([[0.05, 0.15]], dtype=np.float32))
    high_map = calibration.transform(np.array([[0.35, 0.50]], dtype=np.float32))
    assert float(low_map.max()) < 0.5
    assert float(high_map.min()) >= 0.75
    assert calibration.metadata()["normal_patch_scores"] == 4


def test_blend_anomaly_scores_is_fixed_convex_combination() -> None:
    first = np.zeros((2, 2), dtype=np.float32)
    second = np.ones((2, 2), dtype=np.float32)
    assert np.allclose(blend_anomaly_scores(first, second, spatial_weight=0.25), 0.25)
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        blend_anomaly_scores(first, second, spatial_weight=1.1)
