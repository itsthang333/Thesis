from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    from models.mask_bag_orbit_relation_training import (
        OrbitRelationTrainingConfig,
        audit_orbit_initialization_records,
        initial_critical_relation_state,
        orbit_average,
        score_orbit_relation_records,
        train_orbit_relation_adapter,
    )
    from models.mask_bag_relational_selector import CriticalRelationResidual
    from models.rad_dino_mask_bag_mil import smooth_mil_pool


class _TinyBase(nn.Module if nn is not None else object):
    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(descriptor_dim, 1)

    def score_descriptors(self, descriptors, valid):
        logits = self.scorer(descriptors).squeeze(-1).masked_fill(~valid, 0.0)
        return logits, smooth_mil_pool(logits, valid, temperature=0.2)


def _records() -> list[dict[str, object]]:
    generator = np.random.default_rng(9)
    records = []
    for index, count in enumerate((3, 2, 4, 3)):
        original = generator.normal(size=(count, 6)).astype(np.float32)
        flipped = generator.normal(size=(count, 6)).astype(np.float32)
        records.append(
            {
                "image_id": f"image_{index}",
                "label": index % 2,
                "candidate_indices": np.arange(count),
                "descriptors": original,
                "flipped_descriptors": flipped,
            }
        )
    return records


def test_r4_training_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_orbit_relation_training.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "candidate_quality",
        "lesion_size",
        "ground_truth",
    ):
        assert forbidden not in lowered
    assert re.search(r"\bdice\b", lowered) is None


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_orbit_average_is_exactly_view_swap_invariant() -> None:
    first = torch.randn(2, 3, 6)
    second = torch.randn(2, 3, 6)
    assert torch.equal(orbit_average(first, second), orbit_average(second, first))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_r4_initial_audit_and_scoring_are_swap_exact() -> None:
    torch.manual_seed(3)
    base = _TinyBase(6)
    state = initial_critical_relation_state(descriptor_dim=6, hidden_dim=4, seed=42)
    adapter = CriticalRelationResidual(6, 4)
    adapter.load_state_dict(state)
    audit = audit_orbit_initialization_records(
        _records(), base, adapter, batch_size=2, device=torch.device("cpu")
    )
    assert audit["records"] == 4
    assert audit["combined_equals_averaged_frozen_base_exact"] is True
    assert audit["view_swap_candidate_logits_exact"] is True
    scored = score_orbit_relation_records(
        _records(),
        base,
        adapter,
        bag_temperature=0.2,
        batch_size=3,
        device=torch.device("cpu"),
    )
    assert all(item["view_swap_exact"] is True for item in scored)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_r4_training_updates_only_adapter_and_remains_swap_exact() -> None:
    torch.manual_seed(3)
    base = _TinyBase(6)
    base_before = {key: value.detach().clone() for key, value in base.state_dict().items()}
    state = initial_critical_relation_state(descriptor_dim=6, hidden_dim=4, seed=42)
    adapter, history = train_orbit_relation_adapter(
        _records(),
        base,
        descriptor_dim=6,
        bag_temperature=0.2,
        training_config=OrbitRelationTrainingConfig(
            epochs=3,
            batch_size=2,
            hidden_dim=4,
            instance_warmup_epochs=1,
        ),
        device=torch.device("cpu"),
        initial_state=state,
    )
    assert len(history) == 3
    assert all(torch.equal(value, base_before[key]) for key, value in base.state_dict().items())
    scored = score_orbit_relation_records(
        _records(),
        base,
        adapter,
        bag_temperature=0.2,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert all(item["view_swap_exact"] is True for item in scored)
