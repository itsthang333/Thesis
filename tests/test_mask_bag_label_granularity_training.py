from __future__ import annotations

import numpy as np
import pytest
import torch

from project.models.mask_bag_label_granularity import LabelGranularityConfig
from project.models.mask_bag_label_granularity_training import (
    LabelGranularityTrainingConfig,
    attach_frozen_base_logits,
    attach_tumor_type_labels,
    audit_zero_initialization,
    initial_residual_state,
    score_label_granularity_pair,
    train_label_granularity_arm,
)


class _FrozenBase(torch.nn.Module):
    def score_descriptors(
        self, descriptors: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = descriptors[..., 0].masked_fill(~valid, 0.0)
        masked = candidate.masked_fill(~valid, -torch.inf)
        bag = 0.2 * (
            torch.logsumexp(masked / 0.2, dim=1)
            - valid.sum(dim=1).to(candidate.dtype).log()
        )
        return candidate, bag


def _records() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rng = np.random.default_rng(9)
    records: list[dict[str, object]] = []
    rows: list[dict[str, str]] = []
    labels = [(0, 0), (1, 1), (1, 8), (1, 3)]
    for index, (tumor, tumor_type) in enumerate(labels):
        count = 3 + index
        records.append(
            {
                "image_id": f"IMG{index:06d}.jpeg",
                "group_id": f"group-{index}",
                "label": tumor,
                "candidate_indices": np.arange(count, dtype=np.int64),
                "descriptors": rng.normal(size=(count, 6)).astype(np.float16),
                "flipped_descriptors": rng.normal(size=(count, 6)).astype(np.float16),
            }
        )
        rows.append(
            {
                "image_id": f"IMG{index:06d}.jpeg",
                "tumor": str(tumor),
                "tumor_type": str(tumor_type),
            }
        )
    return records, rows


def test_label_join_and_frozen_base_attachment() -> None:
    records, rows = _records()
    counts = attach_tumor_type_labels(records, rows)
    assert counts == [1, 0, 1, 0, 0, 0, 0, 1, 0]
    attach_frozen_base_logits(
        records, _FrozenBase(), batch_size=2, device=torch.device("cpu")
    )
    for record in records:
        count = len(record["candidate_indices"])
        assert np.asarray(record["base_candidate_logits"]).shape == (count,)
        assert np.asarray(record["base_flipped_candidate_logits"]).shape == (count,)


def test_zero_initialization_audit_reproduces_frozen_base_exactly() -> None:
    records, rows = _records()
    attach_tumor_type_labels(records, rows)
    attach_frozen_base_logits(
        records, _FrozenBase(), batch_size=2, device=torch.device("cpu")
    )
    config = LabelGranularityConfig(descriptor_dim=6, hidden_dim=4, dropout=0.0)
    state = initial_residual_state(config, seed=42)
    audit = audit_zero_initialization(
        records,
        model_config=config,
        batch_size=2,
        device=torch.device("cpu"),
        initial_state=state,
    )
    assert audit["records"] == 4
    assert audit["exact_control_candidate_score_records"] == 4
    assert audit["exact_hierarchy_candidate_score_records"] == 4
    assert audit["maximum_candidate_score_delta"] == 0.0
    assert audit["maximum_zero_init_entropy_route_strength"] < 1.0e-6


def test_matched_training_and_label_free_validation_route() -> None:
    records, rows = _records()
    counts = attach_tumor_type_labels(records, rows)
    # Synthetic tests need every class count positive for the fixed class-weight formula.
    counts = [max(1, value) for value in counts]
    attach_frozen_base_logits(
        records, _FrozenBase(), batch_size=4, device=torch.device("cpu")
    )
    model_config = LabelGranularityConfig(
        descriptor_dim=6, hidden_dim=4, dropout=0.0
    )
    training_config = LabelGranularityTrainingConfig(
        epochs=1, batch_size=4, learning_rate=1.0e-3, seed=42
    )
    initial = initial_residual_state(model_config, seed=42)
    control, control_history = train_label_granularity_arm(
        records,
        model_config=model_config,
        training_config=training_config,
        subtype_counts=counts,
        hierarchical=False,
        device=torch.device("cpu"),
        initial_state=initial,
    )
    hierarchy, hierarchy_history = train_label_granularity_arm(
        records,
        model_config=model_config,
        training_config=training_config,
        subtype_counts=counts,
        hierarchical=True,
        device=torch.device("cpu"),
        initial_state=initial,
    )
    assert len(control_history) == len(hierarchy_history) == 1
    assert control.state_dict().keys() == hierarchy.state_dict().keys()
    arms_before, diagnostics_before = score_label_granularity_pair(
        records,
        control,
        hierarchy,
        model_config=model_config,
        batch_size=4,
        device=torch.device("cpu"),
    )
    for record in records:
        record["tumor_type"] = 9 if int(record["label"]) else 0
    arms_after, diagnostics_after = score_label_granularity_pair(
        records,
        control,
        hierarchy,
        model_config=model_config,
        batch_size=4,
        device=torch.device("cpu"),
    )
    for name in arms_before:
        assert len(arms_before[name]) == len(arms_after[name])
        for first, second in zip(arms_before[name], arms_after[name]):
            assert np.array_equal(first["candidate_logits"], second["candidate_logits"])
            assert first["bag_probability"] == second["bag_probability"]
    assert [row["predicted_tumor_type"] for row in diagnostics_before] == [
        row["predicted_tumor_type"] for row in diagnostics_after
    ]


def test_label_join_rejects_binary_subtype_mismatch() -> None:
    records, rows = _records()
    rows[0]["tumor_type"] = "1"
    with pytest.raises(ValueError, match="mismatch"):
        attach_tumor_type_labels(records, rows)
