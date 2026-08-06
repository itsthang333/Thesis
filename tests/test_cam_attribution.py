from __future__ import annotations

import torch

from project.models.cam_attribution import (
    gradcam_map,
    gradcam_plus_plus_map,
    linear_cam_map,
    normalize_and_resize,
)


def test_linear_cam_is_classifier_weighted_channel_sum() -> None:
    activations = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    weights = torch.tensor([[2.0, -1.0]])
    result = linear_cam_map(activations, weights)
    assert torch.equal(result, torch.tensor([[[[0.0, 0.0]]]]))


def test_gradcam_uses_spatial_mean_gradient_weight() -> None:
    activations = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    gradients = torch.tensor([[[[2.0, 4.0]], [[-1.0, -1.0]]]])
    # weights are 3 and -1, giving [0, 2] after ReLU.
    result = gradcam_map(activations, gradients)
    assert torch.equal(result, torch.tensor([[[[0.0, 2.0]]]]))


def test_gradcam_plus_plus_is_finite_and_nonnegative() -> None:
    generator = torch.Generator().manual_seed(7)
    activations = torch.randn(2, 4, 3, 3, generator=generator)
    gradients = torch.randn(2, 4, 3, 3, generator=generator)
    result = gradcam_plus_plus_map(activations, gradients)
    assert result.shape == (2, 1, 3, 3)
    assert torch.isfinite(result).all()
    assert torch.all(result >= 0)


def test_normalize_and_resize_has_unit_interval() -> None:
    cam = torch.tensor([[[[-1.0, 1.0], [2.0, 3.0]]]])
    result = normalize_and_resize(cam, (4, 4))
    assert result.shape == (1, 1, 4, 4)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
