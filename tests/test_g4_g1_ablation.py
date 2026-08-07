from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from models.rad_dino_mask_bag_mil import (  # noqa: E402
    MaskBagMILConfig,
    negative_bag_instance_loss,
)
from run_g4_g1_ablation import (  # noqa: E402
    _parse_seeds,
    _reported_arm_names,
    _training_specs,
    descriptor_feature_mask,
)


def test_feature_arms_zero_exact_cumulative_blocks() -> None:
    config = MaskBagMILConfig(token_dim=2, token_layers=3, metadata_dim=4)
    expected = {
        "inside_only": (6, 0, 0, 0),
        "inside_ring": (6, 6, 0, 0),
        "inside_ring_contrast": (6, 6, 6, 0),
        "full": (6, 6, 6, 4),
    }
    for arm, counts in expected.items():
        mask = descriptor_feature_mask(config, arm)
        blocks = (mask[:6], mask[6:12], mask[12:18], mask[18:])
        assert tuple(int(block.sum().item()) for block in blocks) == counts
        assert mask.numel() == config.descriptor_dim
        assert set(mask.tolist()) <= {0.0, 1.0}


def test_negative_bag_loss_ignores_positive_bags() -> None:
    logits = torch.tensor([[0.5, -0.5], [100.0, -100.0]], requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    labels = torch.tensor([0.0, 1.0])
    loss = negative_bag_instance_loss(logits, valid, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0], torch.zeros(2)
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_negative_bag_loss_has_differentiable_zero_without_negatives() -> None:
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss = negative_bag_instance_loss(
        logits,
        torch.ones_like(logits, dtype=torch.bool),
        torch.ones(1),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_e6_runs_seven_unique_models_but_reports_eight_arms_per_seed() -> None:
    specs = _training_specs()
    assert len(specs) == 7
    assert len({spec["key"] for spec in specs}) == 7
    names = _reported_arm_names(42)
    assert len(names) == 8
    assert names["feature_full"] != names["loss_full"]
    assert any(spec["key"] == "full" for spec in specs)


def test_seed_contract_is_fixed() -> None:
    assert _parse_seeds("42,43,44") == (42, 43, 44)
    with pytest.raises(ValueError):
        _parse_seeds("42")

