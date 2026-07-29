from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_proposal_clusters import (
        build_teacher_proposal_clusters,
        continuation_temperature,
        proposal_cluster_smooth_pool,
    )


def test_proposal_cluster_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_proposal_clusters.py"
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
def test_teacher_clusters_follow_score_and_overlap() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, True, True]])
    overlap = torch.tensor(
        [
            [
                [1.0, 0.8, 0.1, 0.0],
                [0.8, 1.0, 0.1, 0.0],
                [0.1, 0.1, 1.0, 0.7],
                [0.0, 0.0, 0.7, 1.0],
            ]
        ]
    )
    clusters, cluster_valid, seeds = build_teacher_proposal_clusters(
        logits,
        valid,
        overlap,
        maximum_clusters=2,
        minimum_overlap=0.5,
    )
    assert cluster_valid.tolist() == [[True, True]]
    assert seeds.tolist() == [[0, 2]]
    assert clusters[0, 0].tolist() == [True, True, False, False]
    assert clusters[0, 1].tolist() == [False, False, True, True]


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_cluster_pool_is_invariant_to_identical_member_duplication() -> None:
    original_clusters = torch.tensor([[[True, False], [False, True]]])
    original, original_bag = proposal_cluster_smooth_pool(
        torch.tensor([[2.0, 0.0]]),
        original_clusters,
        torch.tensor([[True, True]]),
        within_temperature=0.2,
        between_temperature=0.2,
    )
    duplicated_clusters = torch.tensor(
        [[[True, True, False], [False, False, True]]]
    )
    duplicated, duplicated_bag = proposal_cluster_smooth_pool(
        torch.tensor([[2.0, 2.0, 0.0]]),
        duplicated_clusters,
        torch.tensor([[True, True]]),
        within_temperature=0.2,
        between_temperature=0.2,
    )
    assert torch.allclose(original, duplicated, atol=1.0e-6, rtol=0.0)
    assert torch.allclose(
        original_bag,
        duplicated_bag,
        atol=1.0e-6,
        rtol=0.0,
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_continuation_temperature_sharpens_monotonically() -> None:
    values = [
        continuation_temperature(
            epoch,
            5,
            start_temperature=1.0,
            end_temperature=0.1,
        )
        for epoch in range(1, 6)
    ]
    assert values[0] == 1.0
    assert values[-1] == pytest.approx(0.1)
    assert all(left > right for left, right in zip(values, values[1:]))
