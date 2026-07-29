from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_affinity_features import affinity_summary_features


def test_affinity_feature_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_affinity_features.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "lesion_size",
        "size_group",
        "ground_truth",
    ):
        assert forbidden not in lowered


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_coherent_tokens_have_maximal_inside_affinity() -> None:
    tokens = torch.tensor(
        [[[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]]
    )
    proposal = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    context = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    features = affinity_summary_features(
        tokens,
        proposal,
        context,
        torch.tensor([[True]]),
    ).reshape(1, 1, 1, 8)
    assert torch.allclose(features[0, 0, 0, 0], torch.tensor(1.0))
    assert torch.allclose(features[0, 0, 0, 1], torch.tensor(1.0))
    assert torch.allclose(features[0, 0, 0, 2], torch.tensor(1.0))
    assert torch.allclose(features[0, 0, 0, 3], torch.tensor(1.0))
    assert torch.allclose(features[0, 0, 0, 4], torch.tensor(0.0))
    assert torch.allclose(features[0, 0, 0, 5], torch.tensor(1.0))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_mixed_orthogonal_tokens_reduce_cohesion() -> None:
    tokens = torch.tensor(
        [[[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]]
    )
    proposal = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    context = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
    features = affinity_summary_features(
        tokens,
        proposal,
        context,
        torch.tensor([[True]]),
    ).reshape(1, 1, 1, 8)
    assert torch.allclose(features[0, 0, 0, 0], torch.tensor(0.5))
    assert torch.allclose(features[0, 0, 0, 1], torch.tensor(0.0))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_invalid_candidates_are_zero_and_empty_context_is_supported() -> None:
    tokens = torch.randn(1, 2, 2, 2, 4)
    proposal = torch.zeros(1, 2, 2, 2)
    proposal[0, 0, 0, 0] = 1
    context = torch.zeros_like(proposal)
    valid = torch.tensor([[True, False]])
    features = affinity_summary_features(tokens, proposal, context, valid)
    assert features.shape == (1, 2, 16)
    assert torch.equal(features[0, 1], torch.zeros_like(features[0, 1]))
    reshaped = features.reshape(1, 2, 2, 8)
    assert torch.equal(
        reshaped[0, 0, :, 4],
        torch.zeros_like(reshaped[0, 0, :, 4]),
    )
    assert torch.isfinite(features).all()
