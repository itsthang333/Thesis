import pytest

torch = pytest.importorskip("torch")

from project.models.rad_dino_insight import (
    InsightDenseMILHead,
    InsightMILConfig,
    insight_mil_loss,
    resize_heatmap,
    smoothmax_pool,
)


def test_insight_head_keeps_spatial_heatmap_and_backpropagates():
    torch.manual_seed(42)
    head = InsightDenseMILHead(
        InsightMILConfig(input_dim=8, hidden_dim=12, context_kernel=5)
    )
    tokens = torch.randn(3, 4, 4, 8)
    heatmap, fused, detector, context = head(tokens)
    assert heatmap.shape == fused.shape == detector.shape == context.shape == (3, 1, 4, 4)
    total, bce, spectral, pooled = insight_mil_loss(
        fused, torch.tensor([0.0, 1.0, 1.0]), alpha=8.0
    )
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(bce)
    assert torch.isfinite(spectral)
    assert torch.isfinite(pooled).all()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_insight_head_accepts_flat_square_tokens():
    torch.manual_seed(42)
    head = InsightDenseMILHead(InsightMILConfig(input_dim=4, hidden_dim=8))
    heatmap, *_ = head(torch.randn(2, 16, 4))
    assert heatmap.shape == (2, 1, 4, 4)


def test_smoothmax_is_bounded_and_uses_more_than_one_patch():
    values = torch.tensor([[[0.0, 0.5], [0.5, 1.0]]])
    pooled = smoothmax_pool(values, alpha=8.0)
    assert 0.5 < pooled.item() < 1.0


def test_resize_heatmap_is_bounded():
    values = torch.tensor([[[[-0.1, 0.4], [1.2, 0.8]]]])
    resized = resize_heatmap(values, output_size=8)
    assert resized.shape == (1, 1, 8, 8)
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 1.0


def test_config_rejects_even_context_kernel():
    with pytest.raises(ValueError):
        InsightMILConfig(context_kernel=4)
