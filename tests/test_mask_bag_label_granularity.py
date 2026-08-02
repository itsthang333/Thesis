from __future__ import annotations

import math

import pytest
import torch

from project.models.mask_bag_label_granularity import (
    LabelGranularityConfig,
    LabelGranularityResidual,
    center_valid_candidates,
    entropy_route_strength,
    entropy_routed_candidate_logits,
    inverse_sqrt_subtype_weights,
    label_granularity_losses,
    pathology_bag_logits,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    descriptors = torch.randn(3, 5, 12)
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    base = torch.randn(3, 5).masked_fill(~valid, 0.0)
    return descriptors, valid, base


def test_candidate_centering_uses_only_valid_rows() -> None:
    descriptors, valid, _base = _batch()
    centered = center_valid_candidates(descriptors, valid)
    for row in range(len(centered)):
        assert torch.allclose(
            centered[row, valid[row]].mean(dim=0),
            torch.zeros(descriptors.shape[-1]),
            atol=1.0e-6,
            rtol=0.0,
        )
        assert torch.equal(
            centered[row, ~valid[row]], torch.zeros_like(centered[row, ~valid[row]])
        )


def test_zero_initialization_is_exact_baseline_for_both_routes() -> None:
    descriptors, valid, base = _batch()
    config = LabelGranularityConfig(descriptor_dim=12, hidden_dim=8, dropout=0.0)
    model = LabelGranularityResidual(config).eval()
    residual = model(descriptors, valid)
    assert torch.equal(residual, torch.zeros_like(residual))
    routed, subtype_bag, predicted, strength = entropy_routed_candidate_logits(
        base,
        residual,
        valid,
        temperature=config.bag_temperature,
    )
    assert torch.equal(routed, base)
    assert torch.equal(predicted, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(strength, torch.zeros(3), atol=1.0e-7, rtol=0.0)
    assert torch.allclose(
        subtype_bag,
        subtype_bag[:, :1].expand_as(subtype_bag),
        atol=0.0,
        rtol=0.0,
    )


def test_entropy_route_interpolates_between_coarse_and_selected_subtype() -> None:
    _descriptors, valid, base = _batch()
    uniform = torch.zeros(3, 9)
    assert torch.allclose(
        entropy_route_strength(uniform), torch.zeros(3), atol=1.0e-7, rtol=0.0
    )
    confident = torch.full((1, 9), -100.0)
    confident[0, 4] = 100.0
    assert entropy_route_strength(confident).item() == pytest.approx(1.0)

    residual = torch.zeros(3, 5, 9)
    residual[0, :, 4] = torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0])
    residual = center_valid_candidates(residual, valid)
    routed, _bag, _predicted, strength = entropy_routed_candidate_logits(
        base, residual, valid, temperature=0.2
    )
    assert routed.shape == base.shape
    assert (strength >= 0).all() and (strength <= 1).all()
    assert torch.equal(routed[~valid], torch.zeros_like(routed[~valid]))


def test_pathology_pool_is_normalized_for_duplicated_equal_logits() -> None:
    logits = torch.full((2, 9), 2.5)
    pooled = pathology_bag_logits(logits)
    assert torch.allclose(pooled[:, 0], pooled[:, 1], atol=1.0e-6, rtol=0.0)


def test_inverse_sqrt_weights_are_finite_and_mean_one() -> None:
    counts = torch.tensor([598, 211, 164, 74, 35, 41, 92, 237, 36])
    weights = inverse_sqrt_subtype_weights(counts)
    assert weights.shape == (9,)
    assert torch.isfinite(weights).all()
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[4] > weights[0]


def test_matched_losses_use_only_image_labels_and_backpropagate() -> None:
    descriptors, valid, base = _batch()
    config = LabelGranularityConfig(descriptor_dim=12, hidden_dim=8, dropout=0.0)
    initial = LabelGranularityResidual(config).state_dict()
    coarse_model = LabelGranularityResidual(config)
    hierarchy_model = LabelGranularityResidual(config)
    coarse_model.load_state_dict(initial, strict=True)
    hierarchy_model.load_state_dict(initial, strict=True)
    assert coarse_model.state_dict().keys() == hierarchy_model.state_dict().keys()

    coarse = coarse_model(descriptors, valid)
    hierarchy = hierarchy_model(descriptors, valid)
    labels = torch.tensor([0, 1, 1])
    tumor_types = torch.tensor([0, 3, 8])
    weights = inverse_sqrt_subtype_weights(torch.arange(1, 10))
    coarse_losses = label_granularity_losses(
        base_candidate_logits=base,
        residuals=coarse,
        flipped_residuals=coarse,
        candidate_valid=valid,
        tumor_labels=labels,
        tumor_type_labels=tumor_types,
        subtype_class_weights=weights,
        config=config,
        hierarchical=False,
    )
    hierarchy_losses = label_granularity_losses(
        base_candidate_logits=base,
        residuals=hierarchy,
        flipped_residuals=hierarchy,
        candidate_valid=valid,
        tumor_labels=labels,
        tumor_type_labels=tumor_types,
        subtype_class_weights=weights,
        config=config,
        hierarchical=True,
    )
    assert coarse_losses["pathology"].item() == 0.0
    assert coarse_losses["subtype"].item() == 0.0
    assert math.isfinite(hierarchy_losses["total"].item())
    hierarchy_losses["total"].backward()
    assert hierarchy_model.output.weight.grad is not None
    assert torch.isfinite(hierarchy_model.output.weight.grad).all()


def test_inconsistent_image_labels_fail_closed() -> None:
    descriptors, valid, base = _batch()
    config = LabelGranularityConfig(descriptor_dim=12, hidden_dim=8, dropout=0.0)
    model = LabelGranularityResidual(config)
    residual = model(descriptors, valid)
    with pytest.raises(ValueError, match="inconsistent"):
        label_granularity_losses(
            base_candidate_logits=base,
            residuals=residual,
            flipped_residuals=residual,
            candidate_valid=valid,
            tumor_labels=torch.tensor([0, 1, 1]),
            tumor_type_labels=torch.tensor([0, 0, 8]),
            subtype_class_weights=torch.ones(9),
            config=config,
            hierarchical=True,
        )


def test_hierarchical_all_normal_batch_keeps_only_weighted_binary_term() -> None:
    descriptors, valid, base = _batch()
    config = LabelGranularityConfig(descriptor_dim=12, hidden_dim=8, dropout=0.0)
    residual = LabelGranularityResidual(config)(descriptors, valid)
    losses = label_granularity_losses(
        base_candidate_logits=base,
        residuals=residual,
        flipped_residuals=residual,
        candidate_valid=valid,
        tumor_labels=torch.zeros(3, dtype=torch.long),
        tumor_type_labels=torch.zeros(3, dtype=torch.long),
        subtype_class_weights=torch.ones(9),
        config=config,
        hierarchical=True,
    )
    assert losses["pathology"].item() == 0.0
    assert losses["subtype"].item() == 0.0
    assert losses["total"].item() == pytest.approx(
        config.hierarchy_binary_weight * losses["binary"].item()
    )
