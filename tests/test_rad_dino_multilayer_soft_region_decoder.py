from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.models.rad_dino_multilayer_soft_region_decoder import (
    MultiLayerSoftRegionConfig,
    RadDinoMultiLayerSoftRegionDecoder,
    bidirectional_affinity_refinement,
    horizontal_flip_consistency_loss,
    image_level_loss,
    local_affinity,
    make_guidance,
    soft_affinity_pair_loss,
    soft_region_pseudo_loss,
    soft_region_weights,
)


def _fixture():
    torch.manual_seed(42)
    config = MultiLayerSoftRegionConfig(
        input_dim=12,
        layer_count=3,
        hidden_dim=16,
        affinity_dim=8,
        affinity_radius=1,
    )
    decoder = RadDinoMultiLayerSoftRegionDecoder(config)
    tokens = torch.randn(2, 3, 4, 4, 12)
    guidance = make_guidance(torch.rand(2, 3, 16, 16), output_size=8)
    teacher = torch.zeros(2, 1, 4, 4)
    teacher[1, 0, 1:3, 1:3] = torch.tensor(
        [[0.92, 0.96], [0.99, 1.00]]
    )
    valid = torch.ones(2, 1, 4, 4)
    labels = torch.tensor([0.0, 1.0])
    return config, decoder, tokens, guidance, teacher, valid, labels


def test_multilayer_decoder_and_all_losses_backpropagate() -> None:
    config, decoder, tokens, guidance, teacher, valid, labels = _fixture()
    logits, features, layer_weights = decoder(tokens, guidance)
    affinity_weights, learned, pair_validity = local_affinity(
        features,
        tokens[:, -1],
        radius=config.affinity_radius,
    )
    refined = bidirectional_affinity_refinement(
        teacher,
        affinity_weights,
        radius=config.affinity_radius,
    )
    image_loss, pooled = image_level_loss(logits, labels)
    pseudo_loss = soft_region_pseudo_loss(
        logits,
        refined,
        labels,
        valid_region=valid,
    )
    pair_loss = soft_affinity_pair_loss(
        learned,
        pair_validity,
        teacher,
        labels,
        radius=config.affinity_radius,
        valid_region=valid,
    )
    flipped_logits, _features, _weights = decoder(
        tokens.flip(3),
        guidance.flip(3),
    )
    consistency = horizontal_flip_consistency_loss(
        logits,
        flipped_logits,
        valid_region=valid,
    )
    total = image_loss + pseudo_loss + 0.1 * pair_loss + 0.2 * consistency
    total.backward()
    assert logits.shape == (2, 1, 8, 8)
    assert features.shape == (2, 8, 4, 4)
    assert affinity_weights.shape == learned.shape == pair_validity.shape
    assert affinity_weights.shape == (2, 9, 4, 4)
    assert layer_weights.shape == (3,)
    assert float(layer_weights.sum()) == pytest.approx(1.0)
    assert torch.isfinite(pooled).all()
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_absolute_calibration_produces_variable_area_support() -> None:
    teacher = torch.tensor(
        [
            [[[0.95, 0.20], [0.20, 0.20]]],
            [[[0.95, 0.96], [0.97, 0.20]]],
        ]
    )
    foreground, background = soft_region_weights(teacher)
    assert int((foreground > 0).sum()) == 4
    assert int((foreground[0] > 0).sum()) == 1
    assert int((foreground[1] > 0).sum()) == 3
    assert int((background > 0).sum()) == 4
    assert not torch.any((foreground > 0) & (background > 0))


def test_bidirectional_refinement_is_bounded_and_not_fixed_area() -> None:
    teacher = torch.tensor([[[[0.01, 0.10], [0.95, 0.99]]]])
    weights = torch.zeros(1, 9, 2, 2)
    weights[:, 4] = 1.0
    refined = bidirectional_affinity_refinement(
        teacher,
        weights,
        radius=1,
        steps=2,
    )
    assert torch.allclose(refined, teacher)
    assert float(refined.min()) >= 0.0
    assert float(refined.max()) <= 1.0


def test_horizontal_flip_consistency_is_zero_for_aligned_logits() -> None:
    logits = torch.randn(2, 1, 5, 7)
    assert float(
        horizontal_flip_consistency_loss(logits, logits.flip(-1))
    ) == pytest.approx(0.0)


def test_normal_soft_region_loss_is_dense_background_supervision() -> None:
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    teacher = torch.ones(1, 1, 2, 2)
    loss = soft_region_pseudo_loss(
        logits,
        teacher,
        torch.zeros(1),
    )
    loss.backward()
    assert logits.grad is not None
    assert int((logits.grad > 0).sum()) == 16


def test_positive_without_calibrated_region_uses_only_image_loss() -> None:
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    ambiguous_teacher = torch.full((1, 1, 2, 2), 0.75)
    loss = soft_region_pseudo_loss(
        logits,
        ambiguous_teacher,
        torch.ones(1),
    )
    loss.backward()
    assert float(loss) == pytest.approx(0.0)
    assert logits.grad is not None
    assert int(torch.count_nonzero(logits.grad)) == 0


def test_config_rejects_overlapping_soft_regions() -> None:
    with pytest.raises(ValueError):
        MultiLayerSoftRegionConfig(
            background_end=0.95,
            foreground_start=0.90,
        )
