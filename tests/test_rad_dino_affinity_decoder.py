from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.models.rad_dino_affinity_decoder import (
    AffinityDecoderConfig,
    RadDinoSpatialDecoder,
    affinity_pair_loss,
    confidence_masks_from_teacher,
    image_level_loss,
    local_affinity,
    make_guidance,
    masked_pseudo_loss,
    propagate_seed_preserving,
)


def _fixture():
    torch.manual_seed(42)
    config = AffinityDecoderConfig(
        input_dim=12,
        hidden_dim=16,
        affinity_dim=8,
        affinity_radius=1,
    )
    decoder = RadDinoSpatialDecoder(config)
    tokens = torch.randn(2, 4, 4, 12)
    guidance = make_guidance(torch.rand(2, 3, 16, 16), output_size=8)
    frozen = torch.randn(2, 4, 4, 6)
    teacher = torch.zeros(2, 1, 4, 4)
    teacher[1, 0, 1, 2] = 1.0
    labels = torch.tensor([0.0, 1.0])
    return config, decoder, tokens, guidance, frozen, teacher, labels


def test_decoder_affinity_losses_backpropagate() -> None:
    config, decoder, tokens, guidance, frozen, teacher, labels = _fixture()
    logits, features = decoder(tokens, guidance)
    weights, learned, valid = local_affinity(
        features,
        frozen,
        radius=config.affinity_radius,
        temperature=config.affinity_temperature,
        frozen_similarity_power=config.frozen_similarity_power,
    )
    refined = propagate_seed_preserving(
        teacher,
        weights,
        radius=config.affinity_radius,
        steps=config.propagation_steps,
        residual=config.propagation_residual,
    )
    image_loss, pooled = image_level_loss(logits, labels)
    pseudo_loss = masked_pseudo_loss(logits, refined, labels)
    pair_loss = affinity_pair_loss(
        learned,
        valid,
        teacher,
        labels,
        radius=config.affinity_radius,
    )
    total = image_loss + pseudo_loss + 0.1 * pair_loss
    total.backward()
    assert logits.shape == (2, 1, 8, 8)
    assert features.shape == (2, 8, 4, 4)
    assert weights.shape == learned.shape == valid.shape == (2, 9, 4, 4)
    assert torch.isfinite(pooled).all()
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_seed_preserving_propagation_never_reduces_source() -> None:
    config, decoder, tokens, guidance, frozen, teacher, _labels = _fixture()
    _logits, features = decoder(tokens, guidance)
    weights, _learned, _valid = local_affinity(
        features,
        frozen,
        radius=config.affinity_radius,
    )
    refined = propagate_seed_preserving(
        teacher,
        weights,
        radius=config.affinity_radius,
    )
    assert torch.all(refined >= teacher)
    assert float(refined.max()) == pytest.approx(1.0)


def test_native_rank_masks_preserve_isolated_foreground_during_loss() -> None:
    teacher = torch.zeros(1, 1, 4, 4)
    teacher[0, 0, 1, 2] = 1.0
    foreground, background = confidence_masks_from_teacher(
        teacher,
        foreground_quantile=0.99,
        background_quantile=0.50,
    )
    assert int(foreground.sum()) == 1
    assert int(background.sum()) == 8
    assert not torch.any(foreground & background)

    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    loss = masked_pseudo_loss(
        logits,
        teacher,
        torch.ones(1),
        foreground_quantile=0.99,
        background_quantile=0.50,
    )
    loss.backward()
    assert logits.grad is not None
    assert int((logits.grad < 0).sum()) == 4
    assert int((logits.grad > 0).sum()) > 0


def test_confidence_ranks_exclude_square_padding() -> None:
    teacher = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    valid = torch.zeros_like(teacher)
    valid[:, :, 1:3, :] = 1.0
    foreground, background = confidence_masks_from_teacher(
        teacher,
        foreground_quantile=0.75,
        background_quantile=0.50,
        valid_region=valid,
    )
    assert int(foreground.sum()) == 2
    assert int(background.sum()) == 4
    assert not torch.any(foreground[:, :, (0, 3), :])
    assert not torch.any(background[:, :, (0, 3), :])
    assert not torch.any(foreground & background)


def test_config_rejects_overlapping_pseudo_quantiles() -> None:
    with pytest.raises(ValueError):
        AffinityDecoderConfig(
            background_quantile=0.99,
            foreground_quantile=0.99,
        )
