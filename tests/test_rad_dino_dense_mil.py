import pytest

torch = pytest.importorskip("torch")

from project.models.rad_dino_dense_mil import (
    DenseMILHead,
    dense_mil_loss,
    logsumexp_pool,
    merge_full_and_tiles,
    resize_probability_map,
)


def test_dense_head_accepts_cloned_inference_tokens_for_backward():
    torch.manual_seed(42)
    head = DenseMILHead(8)
    with torch.inference_mode():
        inference_tokens = torch.randn(2, 3, 4, 8)
    tokens = inference_tokens.clone()
    logits = head(tokens)
    loss, _ = dense_mil_loss(logits, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_dense_head_and_mil_loss_are_finite():
    torch.manual_seed(42)
    head = DenseMILHead(8)
    tokens = torch.randn(3, 4, 5, 8)
    logits = head(tokens)
    assert logits.shape == (3, 1, 4, 5)
    loss, pooled = dense_mil_loss(logits, torch.tensor([0.0, 1.0, 1.0]))
    assert loss.ndim == 0
    assert pooled.shape == (3,)
    assert torch.isfinite(loss)
    assert torch.isfinite(pooled).all()


def test_lse_pooling_preserves_a_single_high_patch():
    values = torch.full((1, 4, 4), -5.0)
    values[0, 1, 2] = 5.0
    pooled = logsumexp_pool(values, temperature=0.2)
    assert pooled.item() > 3.0


def test_resize_probability_map_is_bounded():
    logits = torch.tensor([[[[-2.0, 2.0], [0.0, 1.0]]]])
    output = resize_probability_map(logits, output_size=8)
    assert output.shape == (1, 1, 8, 8)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_merge_full_and_tiles_requires_coverage():
    full = torch.zeros((1, 1, 4, 4))
    tiles = torch.ones((4, 1, 2, 2))
    boxes = ((0, 0, 2, 2), (2, 0, 4, 2), (0, 2, 2, 4), (2, 2, 4, 4))
    merged = merge_full_and_tiles(full, tiles, tile_boxes=boxes, image_size=4)
    assert merged.shape == full.shape
    assert torch.allclose(merged, torch.full_like(merged, 0.5))
    with pytest.raises(ValueError):
        merge_full_and_tiles(full, tiles[:3], tile_boxes=boxes, image_size=4)
