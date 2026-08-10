from __future__ import annotations

import inspect

import numpy as np
import torch

from project.models.hr_cbpmil_ie_plus import (
    adaptive_candidate_rings,
    cluster_balanced_detection,
    hr_cbpmil_loss,
    intra_loss_weight,
    project_candidate_masks,
)
from project.selectors.hr_cbpmil_ie_plus import (
    duplicate_cluster_ids,
    select_ie_plus,
)


def test_projection_preserves_area_and_tiny_survival() -> None:
    masks = torch.zeros((1, 2, 320, 320))
    masks[0, 0, 10:20, 30:50] = 1
    masks[0, 1, 17, 19] = 1
    fractional, survival = project_candidate_masks(masks)
    assert torch.equal(fractional.sum(dim=(-2, -1)) * 4, masks.sum(dim=(-2, -1)))
    assert survival[0, 1].sum().item() == 1


def test_adaptive_ring_never_drops_candidate() -> None:
    masks = torch.zeros((1, 2, 160, 160), dtype=torch.bool)
    masks[0, 0, 20:30, 20:30] = True
    masks[0, 1] = True
    rings = adaptive_candidate_rings(masks, torch.ones((1, 2), dtype=torch.bool))
    assert rings.flatten(2).any(dim=2).all()


def test_cluster_fairness_is_independent_of_duplicate_count() -> None:
    logits = torch.zeros((1, 11), dtype=torch.float16)
    clusters = torch.tensor([[0] + [1] * 10])
    valid = torch.ones_like(clusters, dtype=torch.bool)
    cluster_mass, within, detection = cluster_balanced_detection(logits, clusters, valid)
    assert torch.allclose(cluster_mass[0, :1], torch.tensor([0.5]))
    assert torch.allclose(cluster_mass[0, 1:], torch.full((10,), 0.5))
    assert torch.allclose(within[0, :1], torch.tensor([1.0]))
    assert torch.allclose(within[0, 1:], torch.full((10,), 0.1))
    assert torch.isfinite(detection).all()
    assert detection.dtype == torch.float32
    assert torch.allclose(detection.sum(dim=1), torch.ones(1))


def test_duplicate_clusters_use_connected_components() -> None:
    masks = np.zeros((3, 320, 320), dtype=np.uint8)
    masks[0, 10:30, 10:30] = 1
    masks[1, 10:30, 10:30] = 1
    masks[2, 100:120, 100:120] = 1
    ids = duplicate_cluster_ids(masks)
    assert ids.tolist() == [0, 0, 1]


def test_ie_plus_prefers_extent_inside_top_identity_hypotheses() -> None:
    masks = np.zeros((4, 320, 320), dtype=np.uint8)
    masks[0, 20:80, 20:80] = 1
    masks[1, 20:100, 20:100] = 1
    masks[2, 150:210, 150:210] = 1
    masks[3, 240:280, 240:280] = 1
    clusters = np.array([0, 0, 1, 2], dtype=np.int32)
    dense = np.full((160, 160), -5.0, dtype=np.float32)
    dense[10:50, 10:50] = 5.0
    result = select_ie_plus(
        masks,
        classification_logits=np.array([2.0, 2.0, 1.0, 0.0]),
        detection_logits=np.zeros(4),
        dense_logits=dense,
        cluster_ids=clusters,
    )
    assert result.selected_cluster == 0
    assert result.selected_index in {0, 1}
    assert len(result.top3_clusters) == 3


def test_loss_is_finite_and_schedule_is_exact() -> None:
    output = {
        "image_probability": torch.tensor([0.7, 0.2], requires_grad=True),
        "dense_logits": torch.randn(2, 160, 160, requires_grad=True),
        "classification_logits": torch.randn(2, 3, requires_grad=True),
        "instance_probability": torch.sigmoid(torch.randn(2, 3)),
        "detection_mass": torch.full((2, 3), 1 / 3),
        "dense_inside": torch.randn(2, 3, requires_grad=True),
        "dense_ring": torch.randn(2, 3, requires_grad=True),
        "logits10": torch.randn(2, 10, requires_grad=True),
    }
    losses = hr_cbpmil_loss(
        output,
        torch.tensor([1, 0]),
        torch.tensor([3, 0]),
        torch.ones((2, 3), dtype=torch.bool),
        epoch_number=7,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert [intra_loss_weight(i) for i in range(1, 8)] == [0, 0, 0, 0.0625, 0.125, 0.1875, 0.25]


def test_model_api_has_no_forbidden_candidate_features() -> None:
    from project.models.hr_cbpmil_ie_plus import HRCBPMILIEPlus

    signature = inspect.signature(HRCBPMILIEPlus.forward)
    forbidden = {"area", "source", "prompt", "sam_score", "g1_score", "coordinates"}
    assert forbidden.isdisjoint(signature.parameters)
