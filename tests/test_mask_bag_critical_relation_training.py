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
    from models.mask_bag_critical_relation_training import (
        CriticalRelationTrainingConfig,
        audit_zero_initialization_records,
        initial_critical_relation_state,
        score_critical_relation_records,
        train_critical_relation_adapter,
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
    generator = np.random.default_rng(7)
    records = []
    for index, count in enumerate((3, 2, 4, 3)):
        original = generator.normal(size=(count, 6)).astype(np.float32)
        records.append(
            {
                "image_id": f"image_{index}",
                "label": index % 2,
                "candidate_indices": np.arange(count),
                "descriptors": original,
                "flipped_descriptors": original[:, ::-1].copy(),
            }
        )
    return records


def test_r3_training_surface_has_no_segmentation_or_validation_target() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_critical_relation_training.py"
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
def test_r3_initial_audit_is_exact_and_training_keeps_base_frozen() -> None:
    torch.manual_seed(5)
    base = _TinyBase(6)
    base_before = {key: value.detach().clone() for key, value in base.state_dict().items()}
    state = initial_critical_relation_state(descriptor_dim=6, hidden_dim=4, seed=42)
    adapter = CriticalRelationResidual(6, 4)
    adapter.load_state_dict(state)
    audit = audit_zero_initialization_records(
        _records(), base, adapter, batch_size=2, device=torch.device("cpu")
    )
    assert audit["records"] == 4
    assert audit["zero_residual_exact"] is True
    trained, history = train_critical_relation_adapter(
        _records(),
        base,
        descriptor_dim=6,
        bag_temperature=0.2,
        training_config=CriticalRelationTrainingConfig(
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
    assert any(torch.count_nonzero(value).item() for value in trained.residual[-1].parameters())


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_r3_scoring_reports_aligned_gt_blind_diagnostics() -> None:
    torch.manual_seed(5)
    base = _TinyBase(6)
    state = initial_critical_relation_state(descriptor_dim=6, hidden_dim=4, seed=42)
    adapter = CriticalRelationResidual(6, 4)
    adapter.load_state_dict(state)
    scored = score_critical_relation_records(
        _records(),
        base,
        adapter,
        bag_temperature=0.2,
        batch_size=3,
        device=torch.device("cpu"),
    )
    assert [item["candidate_count"] for item in scored] == [3, 2, 4, 3]
    assert all(np.isfinite(item["candidate_logits"]).all() for item in scored)
    assert all(isinstance(item["base_critical_agreement"], bool) for item in scored)
    assert all(isinstance(item["final_selected_agreement"], bool) for item in scored)
