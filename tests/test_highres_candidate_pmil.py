from __future__ import annotations

import numpy as np
import pytest
import torch

from models.highres_candidate_pmil import (
    CandidateSetTransformer,
    HighResProposalMIL,
    aligned_view_consistency,
    area_orthogonality_penalty,
    attention_union_consistency,
    candidate_capture_purity,
    dual_stream_bag_probability,
    image_label_proposal_loss,
    masked_candidate_zone_descriptors,
    pareto_guarded_selection,
    project_candidate_supports,
    top_instance_dropout_mask,
)


def _supports() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = torch.zeros(2, 3, 4, 4)
    rings = torch.zeros_like(masks)
    masks[:, 0, :2, :2] = 1
    masks[:, 1, 1:3, 1:3] = 1
    masks[:, 2, 2:, 2:] = 1
    rings[:, 0, :3, :3] = 1
    rings[:, 0] -= masks[:, 0]
    rings[:, 1, :, :] = 1
    rings[:, 1] -= masks[:, 1]
    rings[:, 2, 1:, 1:] = 1
    rings[:, 2] -= masks[:, 2]
    valid = torch.tensor([[True, True, True], [True, True, False]])
    return masks, rings, valid


def test_zone_pooling_exact_and_invalid_zeroed() -> None:
    feature = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
    masks, rings, valid = _supports()
    result = masked_candidate_zone_descriptors(feature, masks, rings, valid)
    assert result.shape == (2, 3, 8)
    assert torch.allclose(result[0, 0, :2], feature[0, :, :2, :2].mean(dim=(1, 2)))
    assert torch.count_nonzero(result[1, 2]) == 0


def test_support_projection_preserves_validity_and_masks_invalid_candidate() -> None:
    masks, _rings, valid = _supports()
    content = torch.ones(2, 4, 4)
    projected, rings, area = project_candidate_supports(
        masks, content, valid, output_size=(8, 8), ring_radius=1
    )
    assert projected.shape == rings.shape == (2, 3, 8, 8)
    assert torch.all(area[valid] > 0)
    assert torch.count_nonzero(projected[1, 2]) == 0
    assert torch.count_nonzero(rings[1, 2]) == 0


def test_highres_model_end_to_end_shape_and_backward() -> None:
    torch.manual_seed(8)
    model = HighResProposalMIL(
        fpn_channels=32,
        set_hidden_dim=32,
        set_heads=4,
        set_layers=1,
        set_dropout=0.0,
        ring_radius=1,
    )
    model.train()
    images = torch.randn(2, 3, 64, 64)
    masks, _rings, valid = _supports()
    content = torch.ones(2, 4, 4)
    output = model(images, masks, content, valid)
    assert output.classification_logits.shape == (2, 3)
    assert output.detection_logits.shape == (2, 3)
    assert output.dense_logits.shape == (2, 16, 16)
    assert output.candidate_weights.shape == (2, 3, 16, 16)
    loss = image_label_proposal_loss(
        output.classification_logits,
        output.detection_logits,
        output.dense_logits,
        torch.tensor([0, 1]),
        valid,
    )["total"]
    loss.backward()
    assert model.fpn.stem[0].weight.grad is not None
    assert model.dense_head[-1].weight.grad is not None
    assert model.proposal_head.classification.weight.grad is not None


def test_candidate_transformer_is_permutation_equivariant_without_dropout() -> None:
    torch.manual_seed(4)
    model = CandidateSetTransformer(8, hidden_dim=16, heads=4, layers=2, dropout=0.0)
    model.eval()
    values = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    order = torch.tensor([2, 0, 1])
    classification, detection = model(values, valid)
    permuted_classification, permuted_detection = model(values[:, order], valid[:, order])
    assert torch.allclose(permuted_classification, classification[:, order], atol=1e-6)
    assert torch.allclose(permuted_detection, detection[:, order], atol=1e-6)


def test_dual_stream_probability_is_order_invariant_and_has_gradients() -> None:
    classification = torch.tensor([[0.2, -0.4, 1.1]], requires_grad=True)
    detection = torch.tensor([[0.5, 0.1, -0.3]], requires_grad=True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    first = dual_stream_bag_probability(classification, detection, valid)[
        "bag_probability"
    ]
    order = torch.tensor([2, 0, 1])
    second = dual_stream_bag_probability(
        classification[:, order], detection[:, order], valid[:, order]
    )["bag_probability"]
    assert torch.allclose(first, second)
    first.sum().backward()
    assert classification.grad is not None and torch.count_nonzero(classification.grad)
    assert detection.grad is not None and torch.count_nonzero(detection.grad)


def test_image_label_loss_uses_all_normal_candidates_and_pixels() -> None:
    classification = torch.tensor([[1.0, -1.0], [0.3, 0.5]], requires_grad=True)
    detection = torch.tensor([[0.1, 0.2], [0.0, 0.4]], requires_grad=True)
    dense = torch.zeros(2, 3, 3, requires_grad=True)
    valid = torch.ones(2, 2, dtype=torch.bool)
    output = image_label_proposal_loss(
        classification, detection, dense, torch.tensor([0, 1]), valid
    )
    assert output["normal_candidate"] > 0
    assert output["normal_pixel"] > 0
    output["total"].backward()
    assert torch.count_nonzero(classification.grad[0]) == 2
    assert torch.count_nonzero(dense.grad[0]) == 9


def test_normal_candidate_loss_weights_each_image_equally() -> None:
    classification = torch.tensor([[0.0, 0.0, 8.0], [2.0, 2.0, 2.0]])
    detection = torch.zeros_like(classification)
    dense = torch.zeros(2, 1, 1)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    output = image_label_proposal_loss(
        classification, detection, dense, torch.tensor([0, 0]), valid
    )
    expected = 0.5 * (torch.nn.functional.softplus(torch.tensor(0.0)) + torch.nn.functional.softplus(torch.tensor(2.0)))
    assert torch.allclose(output["normal_candidate"], expected)


def test_area_penalty_detects_monotone_shortcut_and_handles_degenerate_bag() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0], [4.0, 4.0, 0.0]], requires_grad=True)
    area = torch.tensor([[1.0, np.e, np.e**2], [1.0, 2.0, 1.0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    penalty = area_orthogonality_penalty(logits, area, valid)
    assert float(penalty.detach()) == pytest.approx(1.0, abs=1e-6)
    penalty.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad[0])


def test_top_instance_dropout_is_deterministic_and_preserves_one_candidate() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [4.0, 1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    kept = top_instance_dropout_mask(logits, valid, fraction=0.5)
    assert kept.tolist() == [[False, False, True, True], [False, True, False, False]]
    assert kept.any(dim=1).all()


def test_aligned_view_consistency_is_zero_only_for_matching_views() -> None:
    candidate = torch.tensor([[1.0, -1.0]], requires_grad=True)
    dense = torch.tensor([[[1.0, -1.0]]], requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    assert aligned_view_consistency(candidate, candidate, dense, dense, valid) == 0
    loss = aligned_view_consistency(candidate, -candidate, dense, -dense, valid)
    assert loss > 0
    loss.backward()
    assert candidate.grad is not None and dense.grad is not None


def test_attention_union_consistency_prefers_matching_dense_map() -> None:
    masks, _rings, valid = _supports()
    attention = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    matching = torch.where(masks[:, 0] > 0, torch.tensor(8.0), torch.tensor(-8.0))
    opposite = -matching
    matching_loss = attention_union_consistency(matching, masks, attention, valid)
    opposite_loss = attention_union_consistency(opposite, masks, attention, valid)
    assert matching_loss < opposite_loss


def test_capture_and_purity_distinguish_coverage_from_dilution() -> None:
    logits = torch.full((1, 4, 4), -8.0)
    logits[:, :2, :2] = 8.0
    masks = torch.zeros(1, 2, 4, 4)
    masks[:, 0, :2, :2] = 1
    masks[:, 1, :, :] = 1
    rings = torch.zeros_like(masks)
    rings[:, 0, :3, :3] = 1
    rings[:, 0] -= masks[:, 0]
    rings[:, 1, 0, :] = 1
    valid = torch.ones(1, 2, dtype=torch.bool)
    capture, purity = candidate_capture_purity(
        logits, masks, rings, valid, torch.ones_like(logits)
    )
    assert capture[0, 1] > capture[0, 0]
    assert purity[0, 0] > purity[0, 1]


def test_pareto_guard_falls_back_without_dominator() -> None:
    result = pareto_guarded_selection(
        np.array([0.9, 1.0, 0.8]),
        np.array([0.9, 0.8, 1.0]),
        np.array([0.9, 1.0, 0.8]),
        np.array([10, 20, 30]),
        0,
    )
    assert result.selected_index == 10
    assert result.switched is False
    assert result.dominator_count == 0


def test_pareto_guard_switches_only_to_componentwise_dominator() -> None:
    result = pareto_guarded_selection(
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([5, 7, 9]),
        0,
    )
    assert result.selected_index == 7
    assert result.switched is True
    assert result.dominator_count == 2


def test_pareto_ties_use_smallest_immutable_candidate_index() -> None:
    result = pareto_guarded_selection(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([100, 30, 20]),
        0,
    )
    assert result.selected_index == 20
