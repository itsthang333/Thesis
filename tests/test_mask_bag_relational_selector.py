from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_relational_selector import (
        CriticalRelationResidual,
        build_family_overlap_graph,
        family_balanced_smooth_mil_pool,
        smooth_candidate_logits,
    )


def test_relational_selector_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_relational_selector.py"
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
def test_family_balancing_is_invariant_to_identical_within_family_duplicates() -> None:
    original, _ = family_balanced_smooth_mil_pool(
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[True, True]]),
        torch.tensor([[0, 1]]),
        temperature=0.2,
    )
    duplicated, families = family_balanced_smooth_mil_pool(
        torch.tensor([[2.0, 2.0, 0.0]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[0, 0, 1]]),
        temperature=0.2,
    )
    assert torch.allclose(original, duplicated, atol=1.0e-6, rtol=0.0)
    assert len(families) == 1
    assert families[0].shape == (2,)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_family_overlap_graph_excludes_cross_family_duplicates() -> None:
    masks = torch.zeros((1, 3, 8, 8))
    masks[0, 0, 1:5, 1:5] = 1
    masks[0, 1, 2:5, 2:5] = 1
    masks[0, 2, 1:5, 1:5] = 1
    graph = build_family_overlap_graph(
        masks,
        torch.tensor([[True, True, True]]),
        torch.tensor([[0, 0, 1]]),
    )
    assert graph[0, 0, 1] == 1
    assert graph[0, 1, 0] == 1
    assert graph[0, 0, 2] == 0
    assert graph[0, 2, 0] == 0
    assert graph[0, 2, 2] == 1


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_graph_smoothing_preserves_isolated_and_reduces_connected_contrast() -> None:
    logits = torch.tensor([[1.0, -1.0, 3.0]])
    valid = torch.tensor([[True, True, True]])
    adjacency = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    smoothed = smooth_candidate_logits(
        logits,
        valid,
        adjacency,
        alpha=0.5,
        iterations=10,
    )
    assert abs(float(smoothed[0, 0] - smoothed[0, 1])) < 2.0
    assert torch.allclose(smoothed[0, 2], logits[0, 2], atol=1.0e-6, rtol=0.0)
    assert torch.equal(
        smooth_candidate_logits(
            logits,
            valid,
            adjacency,
            alpha=0.0,
            iterations=10,
        ),
        logits,
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_critical_relation_residual_is_exactly_identity_at_initialization() -> None:
    module = CriticalRelationResidual(descriptor_dim=6, hidden_dim=4)
    descriptors = torch.randn(2, 3, 6)
    independent = torch.tensor([[0.1, 2.0, -0.5], [1.0, 0.0, -3.0]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    combined, critical, residual = module(descriptors, independent, valid)
    expected = independent * valid
    assert torch.equal(combined, expected)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(critical, torch.tensor([1, 0]))
