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
    from models.mask_bag_count_controlled_self_paced import (
        CountControlledResidual,
        CountControlledSelfPacedConfig,
        audit_count_controlled_oof_producer,
        build_self_paced_targets,
        count_independence_loss,
        deterministic_label_group_balanced_batches,
        weighted_supervised_contrastive_loss,
    )


def test_t1_surface_has_no_segmentation_or_subgroup_interface() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_count_controlled_self_paced.py"
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
def test_count_independence_loss_penalizes_monotone_probability_count() -> None:
    counts = torch.tensor([1.0, 2.0, 4.0, 8.0])
    correlated = count_independence_loss(
        torch.tensor([-3.0, -1.0, 1.0, 3.0], requires_grad=True), counts
    )
    orthogonal_logits = torch.tensor([-2.0, 2.0, 2.0, -2.0], requires_grad=True)
    orthogonal = count_independence_loss(orthogonal_logits, counts)
    assert float(correlated.detach().item()) > float(orthogonal.detach().item())
    correlated.backward()
    assert torch.isfinite(correlated)
    singleton = torch.tensor([0.5], requires_grad=True)
    singleton_loss = count_independence_loss(singleton, torch.tensor([3.0]))
    assert float(singleton_loss.item()) == 0.0
    singleton_loss.backward()
    assert float(singleton.grad.item()) == 0.0


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_residual_starts_as_exact_accepted_baseline_identity() -> None:
    residual = CountControlledResidual(descriptor_dim=3, hidden_dim=4)
    descriptors = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
    base = torch.tensor([[0.25, -0.50]])
    valid = torch.tensor([[True, False]])
    combined, values = residual(descriptors, base, valid)
    assert torch.equal(combined, base)
    assert torch.count_nonzero(values).item() == 0


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_supervised_contrastive_prefers_same_label_alignment() -> None:
    labels = torch.tensor([1, 1, 0, 0])
    weights = torch.full((4,), 0.25)
    coherent = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )
    mixed = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]]
    )
    coherent_loss = weighted_supervised_contrastive_loss(
        coherent, labels, weights, temperature=0.1
    )
    mixed_loss = weighted_supervised_contrastive_loss(
        mixed, labels, weights, temperature=0.1
    )
    assert float(coherent_loss.item()) < float(mixed_loss.item())


def _record(image_id: str, group_id: str, label: int, count: int) -> dict[str, object]:
    return {
        "image_id": image_id,
        "group_id": group_id,
        "label": label,
        "candidate_indices": np.arange(count),
        "family_ids": ["a" if index < 2 else "b" for index in range(count)],
    }


def _score(
    record: dict[str, object], fold: int, probability: float, logits: list[float]
) -> dict[str, object]:
    return {
        "image_id": record["image_id"],
        "group_id": record["group_id"],
        "image_label": record["label"],
        "heldout_fold": fold,
        "candidate_count": len(record["candidate_indices"]),
        "bag_probability": probability,
        "original_logits": np.asarray(logits, dtype=np.float32),
        "flipped_logits": np.asarray(logits, dtype=np.float32),
    }


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_batches_are_label_balanced_and_cover_each_record_once() -> None:
    records = [
        _record(f"n{index}", f"g-n{index}", 0, 2) for index in range(8)
    ] + [
        _record(f"p{index}", f"g-p{index}", 1, 2) for index in range(8)
    ]
    batches = deterministic_label_group_balanced_batches(
        records, batch_size=4, seed=42
    )
    flattened = [int(index) for batch in batches for index in batch]
    assert sorted(flattened) == list(range(16))
    for batch in batches:
        labels = [int(records[int(index)]["label"]) for index in batch]
        assert labels.count(0) == labels.count(1) == 2
        groups = [records[int(index)]["group_id"] for index in batch]
        assert len(groups) == len(set(groups))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_producer_gate_rejects_count_shortcut_before_consumer() -> None:
    records = [
        _record("a", "g-a", 0, 2),
        _record("b", "g-b", 1, 3),
        _record("c", "g-c", 0, 4),
        _record("d", "g-d", 1, 5),
    ]
    folds = np.asarray([0, 1, 0, 1], dtype=np.int32)
    artifacts = [
        {
            "heldout_fold": 0,
            "training_groups": ["g-b", "g-d"],
            "heldout_scores": [
                _score(records[0], 0, 0.1, [2.0, 0.0]),
                _score(records[2], 0, 0.8, [2.0, 1.0, 0.0, -1.0]),
            ],
        },
        {
            "heldout_fold": 1,
            "training_groups": ["g-a", "g-c"],
            "heldout_scores": [
                _score(records[1], 1, 0.4, [3.0, 1.0, 0.0]),
                _score(records[3], 1, 0.9, [4.0, 3.0, 2.0, 1.0, 0.0]),
            ],
        },
    ]
    audit = audit_count_controlled_oof_producer(
        records,
        folds,
        artifacts,
        CountControlledSelfPacedConfig(
            maximum_count_spearman=0.1,
            minimum_oof_auroc=0.0,
            minimum_view_agreement=0.0,
        ),
    )
    assert audit["checks"]["count_spearman"] is False
    assert audit["producer_gate_pass"] is False
    with pytest.raises(RuntimeError, match="consumer remains locked"):
        build_self_paced_targets(
            records, audit, CountControlledSelfPacedConfig()
        )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_targets_are_nested_and_negative_mass_is_equal_per_image() -> None:
    records = [
        _record("n1", "g-n1", 0, 3),
        _record("n2", "g-n2", 0, 2),
        _record("p1", "g-p1", 1, 3),
        _record("p2", "g-p2", 1, 3),
        _record("p3", "g-p3", 1, 3),
        _record("p4", "g-p4", 1, 3),
        _record("p5", "g-p5", 1, 3),
    ]
    scores = [
        _score(record, index % 2, 0.5, [5.0 - index * 0.1, 1.0, 0.0])
        for index, record in enumerate(records)
    ]
    targets = build_self_paced_targets(
        records,
        {"producer_gate_pass": True, "ordered_scores": scores},
        CountControlledSelfPacedConfig(),
    )
    stage_sets = [
        {row["image_id"] for row in stage["positive_targets"]}
        for stage in targets["stages"]
    ]
    assert [len(stage) for stage in stage_sets] == [1, 2, 3]
    assert stage_sets[0] <= stage_sets[1] <= stage_sets[2]
    mass = {}
    for row in targets["negative_targets"]:
        mass[row["image_id"]] = mass.get(row["image_id"], 0.0) + row["weight"]
    assert mass == pytest.approx({"n1": 0.5, "n2": 0.5})


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_oof_audit_rejects_training_on_heldout_group() -> None:
    records = [_record("a", "g-a", 0, 2), _record("b", "g-b", 1, 2)]
    folds = np.asarray([0, 1], dtype=np.int32)
    artifacts = [
        {
            "heldout_fold": 0,
            "training_groups": ["g-a", "g-b"],
            "heldout_scores": [_score(records[0], 0, 0.1, [2.0, 0.0])],
        },
        {
            "heldout_fold": 1,
            "training_groups": ["g-a"],
            "heldout_scores": [_score(records[1], 1, 0.9, [2.0, 0.0])],
        },
    ]
    with pytest.raises(RuntimeError, match="held-out groups"):
        audit_count_controlled_oof_producer(
            records, folds, artifacts, CountControlledSelfPacedConfig()
        )
