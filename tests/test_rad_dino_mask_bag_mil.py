from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    mask_pool_descriptors,
    project_direct_resize_masks_to_square,
    self_guided_instance_loss,
    smooth_mil_pool,
    winner_take_all_map,
)


def _inputs():
    torch.manual_seed(7)
    tokens = torch.randn(2, 3, 4, 4, 8)
    masks = torch.zeros(2, 3, 16, 16)
    masks[0, 0, 1:6, 1:6] = 1
    masks[0, 1, 7:15, 7:15] = 1
    masks[1, 0, 2:10, 3:9] = 1
    masks[1, 1, 5:14, 5:14] = 1
    metadata = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, True, False], [True, True, False]])
    return tokens, masks, metadata, valid


def test_mask_pool_descriptors_preserve_small_fractional_masks() -> None:
    config = MaskBagMILConfig(token_dim=8, token_layers=3, hidden_dim=16)
    tokens, masks, metadata, valid = _inputs()
    descriptors, pooled_valid = mask_pool_descriptors(
        tokens, masks, metadata, valid, config
    )
    assert descriptors.shape == (2, 3, config.descriptor_dim)
    assert pooled_valid.tolist() == valid.tolist()
    assert torch.isfinite(descriptors).all()
    assert torch.count_nonzero(descriptors[:, 2]) == 0


def test_direct_resize_mask_projection_recovers_square_content_box() -> None:
    masks = torch.zeros(2, 4, 8)
    masks[0, 1:3, 2:6] = 1
    masks[1] = 1
    projected = project_direct_resize_masks_to_square(
        masks,
        padded_side=8,
        content_box=(0, 2, 8, 6),
        output_size=8,
    )
    assert projected.shape == (2, 8, 8)
    assert torch.equal(projected[0, 2:6], masks[0])
    assert torch.equal(projected[1, 2:6], masks[1])
    assert torch.count_nonzero(projected[:, :2]) == 0
    assert torch.count_nonzero(projected[:, 6:]) == 0


def test_project_then_square_flip_preserves_asymmetric_padding_geometry() -> None:
    mask = torch.zeros(1, 3, 6)
    mask[:, :, :2] = 1
    projected = project_direct_resize_masks_to_square(
        mask,
        padded_side=6,
        content_box=(0, 1, 6, 4),
        output_size=12,
    )
    flipped_content_box = (0, 6 - 4, 6, 6 - 1)
    expected = project_direct_resize_masks_to_square(
        mask.flip(-1),
        padded_side=6,
        content_box=flipped_content_box,
        output_size=12,
    )
    assert torch.allclose(projected.flip(-1), expected)


def test_smooth_pool_ignores_padding_and_fails_closed_on_empty_bag() -> None:
    logits = torch.tensor([[0.2, 1.1, 999.0], [-0.4, 0.6, -999.0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    pooled = smooth_mil_pool(logits, valid, temperature=0.2)
    changed_padding = logits.clone()
    changed_padding[:, 2] *= -3
    assert torch.allclose(
        pooled,
        smooth_mil_pool(changed_padding, valid, temperature=0.2),
    )
    with pytest.raises(ValueError, match="at least one valid"):
        smooth_mil_pool(logits, torch.zeros_like(valid), temperature=0.2)


def test_self_guided_loss_labels_only_winner_in_positive_bag() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [0.5, -0.5, 4.0]], requires_grad=True)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    labels = torch.tensor([1.0, 0.0])
    loss = self_guided_instance_loss(logits, valid, labels)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 0] == 0
    assert logits.grad[0, 2] == 0
    assert logits.grad[1, 0] > 0
    assert logits.grad[1, 1] > 0
    assert logits.grad[1, 2] == 0


def test_winner_take_all_map_keeps_sam_shape() -> None:
    logits = torch.tensor([[0.1, 1.2, 99.0]])
    masks = torch.zeros(1, 3, 8, 8)
    masks[0, 0, :2, :2] = 1
    masks[0, 1, 4:, 4:] = 1
    masks[0, 2] = 1
    valid = torch.tensor([[True, True, False]])
    output, winner = winner_take_all_map(
        logits, masks, valid, torch.tensor([0.0])
    )
    assert winner.tolist() == [1]
    assert torch.equal(output[0] > 0, masks[0, 1].bool())
    assert torch.allclose(output.max(), torch.tensor(0.5))


def test_model_backpropagates_image_and_instance_objectives() -> None:
    config = MaskBagMILConfig(token_dim=8, token_layers=3, hidden_dim=16)
    model = RadDinoMaskBagMIL(config)
    tokens, masks, metadata, valid = _inputs()
    candidate_logits, bag_logits, pooled_valid = model(tokens, masks, metadata, valid)
    loss = image_bag_loss(bag_logits, torch.tensor([1.0, 0.0]))
    loss = loss + 0.25 * self_guided_instance_loss(
        candidate_logits,
        pooled_valid,
        torch.tensor([1.0, 0.0]),
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(loss)


def test_precomputed_descriptor_scoring_matches_full_forward() -> None:
    config = MaskBagMILConfig(token_dim=8, token_layers=3, hidden_dim=16)
    model = RadDinoMaskBagMIL(config).eval()
    tokens, masks, metadata, valid = _inputs()
    descriptors, pooled_valid = mask_pool_descriptors(
        tokens, masks, metadata, valid, config
    )
    with torch.no_grad():
        expected_logits, expected_bag, expected_valid = model(
            tokens, masks, metadata, valid
        )
        actual_logits, actual_bag = model.score_descriptors(
            descriptors, pooled_valid
        )
    assert torch.equal(expected_valid, pooled_valid)
    assert torch.allclose(actual_logits, expected_logits)
    assert torch.allclose(actual_bag, expected_bag)


def test_aligned_consistency_uses_only_valid_candidates() -> None:
    a = torch.tensor([[0.0, 1.0, 100.0]])
    b = torch.tensor([[0.0, 0.0, -100.0]])
    valid = torch.tensor([[True, True, False]])
    loss = aligned_candidate_consistency_loss(a, b, valid)
    expected = torch.nn.functional.smooth_l1_loss(
        torch.sigmoid(a[:, :2]), torch.sigmoid(b[:, :2])
    )
    assert torch.allclose(loss, expected)
