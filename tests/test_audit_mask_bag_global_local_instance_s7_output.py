from __future__ import annotations

import numpy as np
import pytest

from audit_mask_bag_global_local_instance_s7_output import (
    _independent_targets,
    _safe_child,
    _serialize_prediction_map,
    _safety,
    _target_digest,
)
from models.mask_bag_global_local_instance import build_global_local_soft_targets


def test_independent_projection_matches_producer_primitive_exactly() -> None:
    rng = np.random.default_rng(81)
    logits = [rng.normal(size=count).astype(np.float32) for count in (3, 5, 2, 4)]
    labels = [0, 1, 1, 0]
    families = [
        np.asarray([index % 2 for index in range(count)], dtype=np.int32)
        for count in (3, 5, 2, 4)
    ]
    producer_targets, producer_weights, producer_diagnostics = (
        build_global_local_soft_targets(
            logits, labels, families, target_mass=0.325
        )
    )
    audit_targets, audit_weights, audit_diagnostics = _independent_targets(
        logits, labels, families, 0.325
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(producer_targets, audit_targets)
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(producer_weights, audit_weights)
    )
    assert audit_diagnostics["locally_forced_candidates"] == 2
    assert audit_diagnostics["realized_mass_after_local"] == pytest.approx(
        producer_diagnostics["realized_mass_after_local"], abs=1.0e-12
    )
    image_ids = [f"IMG{index:06d}.jpeg" for index in range(4)]
    assert _target_digest(image_ids, producer_targets, producer_weights) == (
        _target_digest(image_ids, audit_targets, audit_weights)
    )


def test_auditor_rejects_unsafe_payload_and_path_escape(tmp_path) -> None:
    _safety(
        {
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        "safe",
    )
    with pytest.raises(ValueError):
        _safety(
            {
                "validation_gt_read": True,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            "unsafe",
        )
    with pytest.raises(ValueError):
        _safe_child(tmp_path, "../escape.npz")


def test_prediction_map_serialization_is_exact_float16() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 1:5] = 1.0
    saved = _serialize_prediction_map(mask, 0.375)
    assert saved.dtype == np.float16
    assert np.array_equal(saved, (mask * 0.375).astype(np.float16))
    with pytest.raises(ValueError):
        _serialize_prediction_map(mask + 0.1, 0.375)
