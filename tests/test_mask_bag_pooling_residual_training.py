from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
SOURCE = PROJECT / "models" / "mask_bag_pooling_residual_training.py"

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_pooling_residual_training import (
        DescriptorOnlyResidual,
        PoolingResidualTrainingConfig,
        pooling_residual_objective,
        score_pooling_residual_records,
        train_pooling_residual_adapter,
    )
    from models.mask_bag_residual_objective import ResidualObjectiveConfig
    from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


def test_s1_source_has_matched_modes_and_no_instance_target_or_gt() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "ground_truth",
        "candidate_quality",
        "size_group",
        "argmax",
        "self_guided_instance_loss",
    ):
        assert forbidden not in lowered
    assert 'POOL_MODES = ("standard", "family_balanced")' in source
    assert "family_balanced_smooth_mil_pool(" in source
    assert "smooth_mil_pool(" in source
    assert "initial_adapter_state: Mapping[str, torch.Tensor]" in source


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_descriptor_residual_is_identity_at_shared_initialization() -> None:
    adapter = DescriptorOnlyResidual(descriptor_dim=6, hidden_dim=4)
    descriptors = torch.randn(2, 3, 6)
    base = torch.randn(2, 3)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    combined, residual = adapter(descriptors, base, valid)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(combined, base * valid)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_only_pooling_mode_changes_matched_zero_residual_objective() -> None:
    base = torch.tensor([[2.0, 2.0, 0.0]])
    flipped = base.clone()
    residual = torch.zeros_like(base)
    valid = torch.ones_like(base, dtype=torch.bool)
    families = torch.tensor([[0, 0, 1]])
    labels = torch.tensor([1.0])
    config = ResidualObjectiveConfig(
        consistency_weight=0.0,
        residual_drift_weight=0.0,
    )
    standard, _ = pooling_residual_objective(
        base,
        flipped,
        residual,
        residual,
        valid,
        families,
        labels,
        config,
        pool_mode="standard",
    )
    balanced, _ = pooling_residual_objective(
        base,
        flipped,
        residual,
        residual,
        valid,
        families,
        labels,
        config,
        pool_mode="family_balanced",
    )
    assert not torch.allclose(standard, balanced)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_matched_training_preserves_base_and_scores_every_candidate() -> None:
    torch.manual_seed(3)
    descriptor_dim = 8
    base = RadDinoMaskBagMIL(
        MaskBagMILConfig(
            token_dim=2,
            token_layers=1,
            hidden_dim=8,
            metadata_dim=2,
        )
    )
    records = []
    for index, label in enumerate((0, 1, 0, 1)):
        count = 2 + index % 2
        records.append(
            {
                "image_id": f"image_{index}",
                "label": label,
                "descriptors": np.random.default_rng(index).normal(
                    size=(count, descriptor_dim)
                ).astype(np.float32),
                "flipped_descriptors": np.random.default_rng(index + 20).normal(
                    size=(count, descriptor_dim)
                ).astype(np.float32),
                "family_ids": np.arange(count, dtype=np.int32) % 2,
            }
        )
    initial = DescriptorOnlyResidual(
        descriptor_dim=descriptor_dim,
        hidden_dim=8,
    ).state_dict()
    initial = {key: value.detach().clone() for key, value in initial.items()}
    before = {key: value.detach().clone() for key, value in base.state_dict().items()}
    adapter, history = train_pooling_residual_adapter(
        records,
        base,
        descriptor_dim=descriptor_dim,
        pool_mode="family_balanced",
        objective_config=ResidualObjectiveConfig(),
        training_config=PoolingResidualTrainingConfig(
            epochs=2,
            batch_size=2,
            hidden_dim=8,
        ),
        device=torch.device("cpu"),
        initial_adapter_state=initial,
    )
    assert len(history) == 2
    assert all(torch.equal(before[key], value) for key, value in base.state_dict().items())
    scored = score_pooling_residual_records(
        records,
        base,
        adapter,
        pool_mode="family_balanced",
        bag_temperature=0.2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert [len(row["candidate_logits"]) for row in scored] == [2, 3, 2, 3]
    assert all(np.isfinite(row["candidate_logits"]).all() for row in scored)
