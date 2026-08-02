from __future__ import annotations

import numpy as np
import pytest
import torch

from models.mask_bag_global_local_instance import (
    GlobalLocalInstanceConfig,
    GlobalLocalInstanceResidual,
    adaptive_positive_mass,
    build_global_local_soft_targets,
    combined_instance_logits,
    equal_family_candidate_weights,
    global_local_instance_losses,
    project_weighted_sigmoid_mass,
)


def test_schedule_is_frozen_and_reaches_external_mass() -> None:
    config = GlobalLocalInstanceConfig(descriptor_dim=4, hidden_dim=3)
    assert adaptive_positive_mass(0, config) == pytest.approx(0.50)
    assert adaptive_positive_mass(10, config) == pytest.approx(0.325)
    assert adaptive_positive_mass(20, config) == pytest.approx(0.15)
    assert adaptive_positive_mass(40, config) == pytest.approx(0.15)
    with pytest.raises(ValueError):
        adaptive_positive_mass(41, config)


def test_zero_initialization_reproduces_base_candidate_scores() -> None:
    torch.manual_seed(4)
    config = GlobalLocalInstanceConfig(descriptor_dim=4, hidden_dim=3, dropout=0.0)
    model = GlobalLocalInstanceResidual(config).eval()
    descriptors = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    base = torch.tensor([[0.1, 0.3, 0.0], [-0.2, 0.4, 0.1]])
    residual = model(descriptors, valid)
    assert torch.equal(residual, torch.zeros_like(residual))
    combined = combined_instance_logits(base, residual, valid)
    assert torch.equal(combined, base)


def test_family_weights_equalize_family_then_candidate() -> None:
    weights = equal_family_candidate_weights(["a", "a", "b", "c", "c", "c"])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:2].sum() == pytest.approx(1.0 / 3.0)
    assert weights[2] == pytest.approx(1.0 / 3.0)
    assert weights[3:].sum() == pytest.approx(1.0 / 3.0)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[3] == pytest.approx(weights[4])


def test_weighted_sigmoid_projection_hits_exact_mass() -> None:
    logits = np.asarray([-4.0, -1.0, 0.5, 3.0], dtype=np.float32)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    probabilities, bias, realized = project_weighted_sigmoid_mass(
        logits, weights, target_mass=0.15
    )
    assert np.isfinite(bias)
    assert realized == pytest.approx(0.15, abs=1.0e-10)
    assert float(np.dot(weights, probabilities.astype(np.float64))) == pytest.approx(
        0.15, abs=2.0e-8
    )
    assert np.all(np.diff(probabilities) > 0.0)

    extreme, _extreme_bias, extreme_mass = project_weighted_sigmoid_mass(
        np.asarray([-1000.0, 1000.0]),
        np.asarray([0.5, 0.5]),
        target_mass=0.15,
    )
    assert np.isfinite(extreme).all()
    assert extreme_mass == pytest.approx(0.15, abs=1.0e-10)


def test_global_local_targets_use_all_instances_and_true_negative_bags() -> None:
    logits = [
        np.asarray([0.1, 1.2, -0.5], dtype=np.float32),
        np.asarray([-0.2, 0.4], dtype=np.float32),
        np.asarray([2.0, -1.0, 0.0, 0.5], dtype=np.float32),
    ]
    labels = [1, 0, 1]
    families = [["a", "a", "b"], ["a", "b"], ["x", "y", "y", "y"]]
    targets, weights, diagnostics = build_global_local_soft_targets(
        logits, labels, families, target_mass=0.15
    )
    assert len(targets) == len(logits)
    assert np.array_equal(targets[1], np.zeros(2, dtype=np.float32))
    assert targets[0][1] == 1.0
    assert targets[2][0] == 1.0
    assert np.all((targets[0] > 0.0) & (targets[0] <= 1.0))
    assert np.all((targets[2] > 0.0) & (targets[2] <= 1.0))
    assert all(float(weight.sum()) == pytest.approx(1.0) for weight in weights)
    assert diagnostics["projected_mass_before_local"] == pytest.approx(
        0.15, abs=1.0e-10
    )
    assert diagnostics["realized_mass_after_local"] >= 0.15
    assert diagnostics["locally_forced_candidates"] == 2


def test_instance_losses_are_weighted_per_image_and_backpropagate() -> None:
    config = GlobalLocalInstanceConfig(descriptor_dim=4, hidden_dim=3)
    original_residual = torch.tensor(
        [[0.2, -0.1, 0.0], [0.1, 0.3, -0.2]], requires_grad=True
    )
    flipped_residual = torch.tensor(
        [[0.1, -0.2, 0.0], [0.2, 0.2, -0.1]], requires_grad=True
    )
    valid = torch.tensor([[True, True, False], [True, True, True]])
    base = torch.tensor([[0.2, 0.4, 0.0], [-0.1, 0.3, 0.2]])
    targets = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.1, 1.0]])
    weights = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.25, 0.25]])
    losses = global_local_instance_losses(
        original_logits=combined_instance_logits(base, original_residual, valid),
        flipped_logits=combined_instance_logits(base, flipped_residual, valid),
        original_residuals=original_residual,
        flipped_residuals=flipped_residual,
        soft_targets=targets,
        candidate_weights=weights,
        candidate_valid=valid,
        config=config,
    )
    assert set(losses) == {"total", "instance", "consistency", "drift"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert original_residual.grad is not None
    assert flipped_residual.grad is not None
    assert torch.isfinite(original_residual.grad).all()
    assert torch.isfinite(flipped_residual.grad).all()


def test_fail_closed_contracts_reject_invalid_mass_and_weights() -> None:
    with pytest.raises(ValueError):
        GlobalLocalInstanceConfig(target_positive_mass=0.6, start_positive_mass=0.5)
    with pytest.raises(ValueError):
        equal_family_candidate_weights([])
    with pytest.raises(ValueError):
        project_weighted_sigmoid_mass(
            np.asarray([0.0, 1.0]),
            np.asarray([0.4, 0.4]),
            target_mass=0.15,
        )
