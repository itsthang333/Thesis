from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from audit_mask_bag_global_local_instance_s7_output import (
    _independent_targets,
    _target_digest,
)
from run_mask_bag_global_local_instance_s7_pair import (
    _validate_recipe,
    _write_target_snapshot,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        epochs=40,
        batch_size=16,
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        hidden_dim=128,
        dropout=0.10,
        bag_temperature=0.20,
        start_positive_mass=0.50,
        target_positive_mass=0.15,
        mass_transition_epochs=20,
        consistency_weight=0.10,
        residual_drift_weight=1.0e-3,
        seed=42,
    )


def test_recipe_is_one_shot_and_fails_closed() -> None:
    args = _args()
    _validate_recipe(args)
    args.target_positive_mass = 0.20
    with pytest.raises(ValueError):
        _validate_recipe(args)


def test_independent_target_reproduction_and_digest() -> None:
    logits = [
        np.asarray([-0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, -0.1, 0.2], dtype=np.float32),
    ]
    labels = [0, 1]
    families = [np.asarray([0, 0]), np.asarray([0, 1, 1])]
    targets, weights, diagnostics = _independent_targets(
        logits, labels, families, 0.5
    )
    assert np.array_equal(targets[0], np.zeros(2, dtype=np.float32))
    assert targets[1][0] == 1.0
    assert diagnostics["locally_forced_candidates"] == 1
    assert len(_target_digest(["normal", "tumor"], targets, weights)) == 64


def test_target_snapshot_is_physical_and_hash_bound(tmp_path) -> None:
    records = []
    logits = []
    targets = []
    weights = []
    for index in range(2981):
        records.append({"image_id": f"IMG{index:06d}.jpeg", "label": index % 2})
        logits.append(np.asarray([float(index % 7)], dtype=np.float32))
        targets.append(np.asarray([float(index % 2)], dtype=np.float32))
        weights.append(np.asarray([1.0], dtype=np.float64))
    manifest = []
    _write_target_snapshot(
        tmp_path,
        manifest,
        0,
        records,
        logits,
        targets,
        weights,
        {
            "target_positive_mass": 0.5,
            "target_sha256": _target_digest(
                [row["image_id"] for row in records], targets, weights
            ),
            "projected_mass_before_local": 0.5,
            "realized_mass_after_local": 1.0,
            "locally_forced_candidates": 1490,
        },
    )
    assert len(manifest) == 1
    payload = np.load(tmp_path / manifest[0]["snapshot_path"], allow_pickle=False)
    assert set(payload.files) == {
        "schema_version",
        "epoch_index",
        "image_ids",
        "labels",
        "offsets",
        "current_logits",
        "soft_targets",
        "candidate_weights",
    }
    assert int(payload["offsets"][-1]) == 2981
    json.dumps(manifest)
