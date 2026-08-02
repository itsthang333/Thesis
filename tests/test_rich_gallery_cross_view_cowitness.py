from __future__ import annotations

import torch

from project.models.rich_gallery_cross_view_cowitness import (
    CrossViewCoWitnessConfig,
    CrossViewCoWitnessHead,
    co_witness_margin_loss,
    co_witness_score,
    dense_normal_candidate_loss,
    image_bag_loss,
)


def _inputs(batch: int = 2, candidates: int = 5, dimension: int = 12):
    torch.manual_seed(4)
    appearance = torch.randn(batch, candidates, dimension)
    baseline = torch.randn(batch, candidates)
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    valid[0, -1] = False
    return appearance, baseline, valid


def test_zero_initialized_residual_reproduces_immutable_baseline() -> None:
    appearance, baseline, valid = _inputs()
    config = CrossViewCoWitnessConfig(
        appearance_dim=12,
        hidden_dim=16,
        embedding_dim=8,
    )
    model = CrossViewCoWitnessHead(config)
    combined, residual, embeddings = model(appearance, baseline, valid)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(combined[valid], baseline[valid])
    assert torch.equal(embeddings[~valid], torch.zeros_like(embeddings[~valid]))


def test_pair_score_is_permutation_and_padding_invariant() -> None:
    appearance, baseline, valid = _inputs()
    model = CrossViewCoWitnessHead(
        CrossViewCoWitnessConfig(appearance_dim=12, hidden_dim=16, embedding_dim=8)
    )
    _combined, residual, embedding = model(appearance, baseline, valid)
    score = co_witness_score(
        residual[:1], embedding[:1], valid[:1], residual[1:], embedding[1:], valid[1:],
        temperature=0.2, cosine_weight=0.5,
    )
    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted = co_witness_score(
        residual[:1, permutation], embedding[:1, permutation], valid[:1, permutation],
        residual[1:], embedding[1:], valid[1:],
        temperature=0.2, cosine_weight=0.5,
    )
    assert torch.allclose(score, permuted, atol=1.0e-6, rtol=0.0)


def test_pair_margin_and_image_losses_backpropagate() -> None:
    appearance, baseline, valid = _inputs()
    model = CrossViewCoWitnessHead(
        CrossViewCoWitnessConfig(appearance_dim=12, hidden_dim=16, embedding_dim=8)
    )
    combined, residual, embedding = model(appearance, baseline, valid)
    positive = co_witness_score(
        residual[:1], embedding[:1], valid[:1], residual[1:], embedding[1:], valid[1:],
        temperature=0.2, cosine_weight=0.5,
    )
    negative = co_witness_score(
        residual[:1], embedding[:1], valid[:1], -residual[1:], -embedding[1:], valid[1:],
        temperature=0.2, cosine_weight=0.5,
    )
    labels = torch.tensor([0.0, 1.0])
    loss = (
        co_witness_margin_loss(positive, negative, margin=0.2)
        + image_bag_loss(combined, valid, labels, temperature=0.2)
        + dense_normal_candidate_loss(combined, valid, labels)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding_head.weight.grad is not None
    assert model.residual_head.weight.grad is not None
    assert torch.isfinite(model.embedding_head.weight.grad).all()
    assert torch.isfinite(model.residual_head.weight.grad).all()


def test_maximum_243_candidate_pair_preflight_is_finite() -> None:
    torch.manual_seed(7)
    config = CrossViewCoWitnessConfig()
    model = CrossViewCoWitnessHead(config)
    appearance_a = torch.randn(1, 243, config.appearance_dim)
    appearance_b = torch.randn(1, 243, config.appearance_dim)
    baseline = torch.randn(1, 243)
    valid = torch.ones(1, 243, dtype=torch.bool)
    combined_a, residual_a, embedding_a = model(appearance_a, baseline, valid)
    _combined_b, residual_b, embedding_b = model(appearance_b, baseline, valid)
    pair = co_witness_score(
        residual_a,
        embedding_a,
        valid,
        residual_b,
        embedding_b,
        valid,
        temperature=config.pair_temperature,
        cosine_weight=config.cosine_weight,
    )
    loss = pair.mean() + image_bag_loss(
        combined_a, valid, torch.ones(1), temperature=config.bag_temperature
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.embedding_head.weight.grad).all()
