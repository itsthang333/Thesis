from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from models.mask_bag_global_local_instance import GlobalLocalInstanceConfig
from models.mask_bag_global_local_instance_training import (
    GlobalLocalInstanceTrainingConfig,
    attach_frozen_base_logits,
    assign_global_local_targets,
    audit_zero_initialization,
    initial_global_local_state,
    padded_global_local_batch,
    score_global_local_instance,
    train_global_local_instance,
)
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


def _records() -> list[dict[str, object]]:
    rng = np.random.default_rng(17)
    result: list[dict[str, object]] = []
    for index, (label, count) in enumerate([(0, 3), (1, 4), (0, 2), (1, 3)]):
        descriptors = rng.normal(size=(count, 4)).astype(np.float32)
        flipped = (descriptors + rng.normal(scale=0.02, size=descriptors.shape)).astype(
            np.float32
        )
        base = rng.normal(scale=0.3, size=count).astype(np.float32)
        base_flipped = (base + rng.normal(scale=0.01, size=count)).astype(np.float32)
        result.append(
            {
                "image_id": f"IMG{index:06d}.jpeg",
                "label": label,
                "descriptors": descriptors,
                "flipped_descriptors": flipped,
                "candidate_indices": np.arange(count, dtype=np.int32),
                "family_ids": np.asarray([position % 2 for position in range(count)]),
                "base_candidate_logits": base,
                "base_flipped_candidate_logits": base_flipped,
            }
        )
    return result


def test_zero_initialization_is_exact_identity_and_preserves_probability() -> None:
    records = _records()
    config = GlobalLocalInstanceConfig(descriptor_dim=4, hidden_dim=3, dropout=0.0)
    initial = initial_global_local_state(config, seed=42)
    audit = audit_zero_initialization(
        records,
        model_config=config,
        initial_state=initial,
        bag_temperature=0.2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert audit == {
        "records": 4,
        "exact_candidate_score_records": 4,
        "exact_selected_index_records": 4,
        "maximum_candidate_score_delta": 0.0,
    }


def test_target_assignment_is_deterministic_and_hash_bound() -> None:
    records = _records()
    config = GlobalLocalInstanceConfig(descriptor_dim=4, hidden_dim=3, dropout=0.0)
    initial = initial_global_local_state(config, seed=42)
    from models.mask_bag_global_local_instance import GlobalLocalInstanceResidual

    model = GlobalLocalInstanceResidual(config)
    model.load_state_dict(initial, strict=True)
    first = assign_global_local_targets(
        records,
        model,
        epoch_index=0,
        model_config=config,
        batch_size=2,
        device=torch.device("cpu"),
    )
    targets = [np.asarray(record["s7_soft_targets"]).copy() for record in records]
    second = assign_global_local_targets(
        records,
        model,
        epoch_index=0,
        model_config=config,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert first["target_sha256"] == second["target_sha256"]
    assert first["target_positive_mass"] == pytest.approx(0.5)
    assert all(
        np.array_equal(before, record["s7_soft_targets"])
        for before, record in zip(targets, records)
    )
    assert np.array_equal(records[0]["s7_soft_targets"], np.zeros(3, np.float32))
    assert np.array_equal(records[2]["s7_soft_targets"], np.zeros(2, np.float32))


def test_padded_batch_requires_complete_target_contract() -> None:
    records = _records()
    with pytest.raises(ValueError):
        padded_global_local_batch(
            records, [0, 1], device=torch.device("cpu"), require_targets=True
        )
    for record in records:
        count = len(record["candidate_indices"])
        record["s7_soft_targets"] = np.zeros(count, dtype=np.float32)
        record["s7_candidate_weights"] = np.full(count, 1.0 / count)
    batch = padded_global_local_batch(
        records, [0, 1], device=torch.device("cpu"), require_targets=True
    )
    assert tuple(batch["descriptors"].shape) == (2, 4, 4)
    assert torch.allclose(batch["candidate_weights"].sum(1), torch.ones(2))


def test_short_synthetic_training_changes_rank_and_keeps_baseline_probability() -> None:
    records = _records()
    model_config = GlobalLocalInstanceConfig(
        descriptor_dim=4,
        hidden_dim=5,
        dropout=0.0,
        mass_transition_epochs=2,
        total_epochs=3,
    )
    training_config = GlobalLocalInstanceTrainingConfig(
        epochs=3,
        batch_size=2,
        learning_rate=1.0e-2,
        weight_decay=0.0,
        seed=42,
    )
    initial = initial_global_local_state(model_config, seed=42)
    initial_records = copy.deepcopy(records)
    model, history = train_global_local_instance(
        records,
        model_config=model_config,
        training_config=training_config,
        device=torch.device("cpu"),
        initial_state=initial,
    )
    assert len(history) == 3
    assert history[0]["target"]["target_positive_mass"] == pytest.approx(0.5)
    assert history[-1]["target"]["target_positive_mass"] == pytest.approx(0.15)
    assert all(np.isfinite(row["total"]) for row in history)
    trained = score_global_local_instance(
        records,
        model,
        bag_temperature=0.2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    identity_model_config = model_config
    identity_state = initial_global_local_state(identity_model_config, seed=42)
    identity_audit = audit_zero_initialization(
        initial_records,
        model_config=identity_model_config,
        initial_state=identity_state,
        bag_temperature=0.2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert identity_audit["exact_candidate_score_records"] == 4
    assert any(
        not np.array_equal(
            row["candidate_logits"], row["base_candidate_logits"]
        )
        for row in trained
    )
    for record, row in zip(initial_records, trained):
        base = 0.5 * (
            np.asarray(record["base_candidate_logits"], dtype=np.float32)
            + np.asarray(record["base_flipped_candidate_logits"], dtype=np.float32)
        )
        tensor = torch.from_numpy(base)[None, :]
        valid = torch.ones_like(tensor, dtype=torch.bool)
        from models.rad_dino_mask_bag_mil import smooth_mil_pool

        expected = float(
            torch.sigmoid(smooth_mil_pool(tensor, valid, temperature=0.2))[0]
        )
        assert row["bag_probability"] == pytest.approx(expected, abs=1.0e-7)


def test_training_epoch_contract_fails_closed() -> None:
    records = _records()
    config = GlobalLocalInstanceConfig(
        descriptor_dim=4, hidden_dim=3, mass_transition_epochs=2, total_epochs=3
    )
    initial = initial_global_local_state(config, seed=42)
    with pytest.raises(ValueError):
        train_global_local_instance(
            records,
            model_config=config,
            training_config=GlobalLocalInstanceTrainingConfig(epochs=2),
            device=torch.device("cpu"),
            initial_state=initial,
        )


def test_attach_frozen_base_logits_has_no_subtype_dependency() -> None:
    records = _records()
    scorer = RadDinoMaskBagMIL(
        MaskBagMILConfig(token_dim=1, token_layers=1, metadata_dim=1, hidden_dim=3)
    ).eval()
    for record in records:
        record.pop("base_candidate_logits")
        record.pop("base_flipped_candidate_logits")
    attach_frozen_base_logits(
        records,
        scorer,
        batch_size=2,
        device=torch.device("cpu"),
    )
    for record in records:
        count = len(record["candidate_indices"])
        assert np.asarray(record["base_candidate_logits"]).shape == (count,)
        assert np.asarray(record["base_flipped_candidate_logits"]).shape == (count,)
        assert "tumor_type" not in record


def test_training_snapshot_callback_observes_each_pre_epoch_target() -> None:
    records = _records()
    config = GlobalLocalInstanceConfig(
        descriptor_dim=4,
        hidden_dim=3,
        dropout=0.0,
        mass_transition_epochs=2,
        total_epochs=2,
    )
    seen: list[tuple[int, int, str]] = []

    def callback(epoch, rows, logits, targets, weights, diagnostics):
        assert len(rows) == len(logits) == len(targets) == len(weights) == 4
        seen.append((epoch, len(logits), diagnostics["target_sha256"]))

    train_global_local_instance(
        records,
        model_config=config,
        training_config=GlobalLocalInstanceTrainingConfig(
            epochs=2, batch_size=2, learning_rate=1.0e-2, weight_decay=0.0
        ),
        device=torch.device("cpu"),
        initial_state=initial_global_local_state(config, seed=42),
        target_snapshot_callback=callback,
    )
    assert [row[0] for row in seen] == [0, 1]
    assert all(len(row[2]) == 64 for row in seen)
