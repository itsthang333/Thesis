from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_descriptor_residual import AuxiliaryDescriptorResidual


def test_descriptor_residual_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_descriptor_residual.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "lesion_size",
        "size_group",
        "ground_truth",
    ):
        assert forbidden not in lowered


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_descriptor_residual_is_exact_identity_at_initialization() -> None:
    module = AuxiliaryDescriptorResidual(
        base_descriptor_dim=6,
        auxiliary_dim=4,
        hidden_dim=8,
    )
    base = torch.randn(2, 3, 6)
    auxiliary = torch.randn(2, 3, 4)
    logits = torch.tensor([[0.1, 2.0, -1.0], [1.0, 0.0, -3.0]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    combined, residual = module(base, auxiliary, logits, valid)
    assert torch.equal(combined, logits * valid)
    assert torch.equal(residual, torch.zeros_like(residual))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_only_final_layer_receives_gradient_on_first_step() -> None:
    module = AuxiliaryDescriptorResidual(
        base_descriptor_dim=6,
        auxiliary_dim=4,
        hidden_dim=8,
    )
    combined, _residual = module(
        torch.randn(1, 2, 6),
        torch.randn(1, 2, 4),
        torch.tensor([[0.1, -0.2]]),
        torch.tensor([[True, True]]),
    )
    combined.sum().backward()
    final = module.residual[-1]
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0
    assert module.base_projection[1].weight.grad is not None
    assert torch.count_nonzero(module.base_projection[1].weight.grad) == 0


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_invalid_candidate_residual_stays_zero_after_training_change() -> None:
    module = AuxiliaryDescriptorResidual(
        base_descriptor_dim=3,
        auxiliary_dim=2,
        hidden_dim=4,
    )
    with torch.no_grad():
        module.residual[-1].weight.fill_(0.1)
        module.residual[-1].bias.fill_(0.2)
    combined, residual = module(
        torch.randn(1, 2, 3),
        torch.randn(1, 2, 2),
        torch.tensor([[1.0, 5.0]]),
        torch.tensor([[True, False]]),
    )
    assert residual[0, 1] == 0
    assert combined[0, 1] == 0
    assert torch.isfinite(combined).all()
