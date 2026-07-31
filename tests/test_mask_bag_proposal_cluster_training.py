from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_proposal_cluster_training import (
        ProposalClusterResidual,
        ProposalClusterTrainingConfig,
        attach_teacher_clusters,
        audit_oof_teacher_coverage,
    )


def test_s4_training_surface_has_no_segmentation_or_subgroup_interface() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_proposal_cluster_training.py"
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
def test_s4_clusters_use_conservative_view_scores_and_frozen_thresholds() -> None:
    record = {
        "image_id": "a",
        "group_id": "g-a",
        "label": 1,
        "candidate_indices": np.arange(4),
        "descriptors": np.zeros((4, 3), dtype=np.float32),
        "flipped_descriptors": np.zeros((4, 3), dtype=np.float32),
        "pairwise_iou": np.asarray(
            [
                [1.0, 0.60, 0.0, 0.0],
                [0.60, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "pairwise_containment": np.asarray(
            [
                [1.0, 0.0, 0.80, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.80, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }
    scores = [
        {
            "image_id": "a",
            "original_logits": np.asarray([5.0, 4.0, 3.0, 2.0]),
            "flipped_logits": np.asarray([1.0, 4.0, 3.0, 2.0]),
            "conservative_seed_logits": np.asarray([1.0, 4.0, 3.0, 2.0]),
            "selected_view_agreement": False,
        }
    ]
    enriched = attach_teacher_clusters(
        [record], scores, ProposalClusterTrainingConfig(maximum_clusters=2)
    )[0]
    assert enriched["seed_indices"].tolist() == [1, 2]
    assert enriched["clusters"][0].tolist() == [True, True, False, False]
    assert enriched["clusters"][1].tolist() == [False, False, True, False]
    assert enriched["teacher_selected_view_agreement"] is False


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_s4_residual_is_exact_identity_and_outside_clusters_stays_frozen() -> None:
    residual = ProposalClusterResidual(descriptor_dim=3, hidden_dim=4)
    descriptors = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
    base = torch.tensor([[0.25, -0.50]])
    members = torch.tensor([[True, False]])
    combined, values = residual(descriptors, base, members)
    assert torch.equal(combined, base)
    assert torch.count_nonzero(values).item() == 0
    with torch.no_grad():
        residual.network[-1].bias.fill_(2.0)
    combined, values = residual(descriptors, base, members)
    assert float(combined[0, 0].detach().item()) == pytest.approx(2.25)
    assert float(combined[0, 1].detach().item()) == float(base[0, 1].item())
    assert float(values[0, 1].detach().item()) == 0.0


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_s4_oof_audit_rejects_training_on_heldout_group() -> None:
    records = [
        {"image_id": "a", "group_id": "g-a"},
        {"image_id": "b", "group_id": "g-b"},
    ]
    folds = np.asarray([0, 1], dtype=np.int32)
    artifacts = [
        {
            "heldout_fold": 0,
            "training_groups": ["g-a", "g-b"],
            "heldout_scores": [
                {"image_id": "a", "group_id": "g-a", "heldout_fold": 0}
            ],
        },
        {
            "heldout_fold": 1,
            "training_groups": ["g-a"],
            "heldout_scores": [
                {"image_id": "b", "group_id": "g-b", "heldout_fold": 1}
            ],
        },
    ]
    with pytest.raises(RuntimeError, match="held-out groups"):
        audit_oof_teacher_coverage(records, folds, artifacts)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_s4_oof_audit_accepts_exact_group_exclusion_and_coverage() -> None:
    records = [
        {"image_id": "a", "group_id": "g-a"},
        {"image_id": "b", "group_id": "g-b"},
    ]
    folds = np.asarray([0, 1], dtype=np.int32)
    artifacts = [
        {
            "heldout_fold": 0,
            "training_groups": ["g-b"],
            "heldout_scores": [
                {"image_id": "a", "group_id": "g-a", "heldout_fold": 0}
            ],
        },
        {
            "heldout_fold": 1,
            "training_groups": ["g-a"],
            "heldout_scores": [
                {"image_id": "b", "group_id": "g-b", "heldout_fold": 1}
            ],
        },
    ]
    audit = audit_oof_teacher_coverage(records, folds, artifacts)
    assert audit["complete"] is True
    assert audit["records"] == 2
    assert audit["group_overlap"] == 0
    assert [row["image_id"] for row in audit["ordered_scores"]] == ["a", "b"]
