from __future__ import annotations

import numpy as np
import torch
from collections import OrderedDict

from project.models.matched_normal_candidate_transplant import (
    build_matched_transplants,
    frozen_selector_panel,
    matched_transplant_scores,
    matched_transplant_layerwise_scores,
    paired_dense_layer_diagnostics,
    percentile_ranks,
    select_normal_reference_pairs,
    select_random_normal_reference_pairs,
)


class _ToyDenseClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            OrderedDict(
                [
                    ("pool0", torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)),
                    ("transition1", torch.nn.AvgPool2d(2)),
                    ("transition2", torch.nn.AvgPool2d(2)),
                    ("transition3", torch.nn.AvgPool2d(2)),
                    ("norm5", torch.nn.BatchNorm2d(4)),
                ]
            )
        )
        self.classifier = torch.nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(batch))
        return self.classifier(features.mean(dim=(-2, -1)))


def _row(
    image_id: str,
    group_id: str,
    *,
    anatomy: str,
    view: str,
    center: str = "1",
    tumor: str = "0",
    width: int = 100,
    height: int = 200,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "group_id": group_id,
        "anatomy": anatomy,
        "view": view,
        "center": center,
        "tumor": tumor,
        "width": width,
        "height": height,
    }


def test_reference_pairing_prioritizes_anatomy_view_and_distinct_groups() -> None:
    query = _row("q", "gq", anatomy="tibia", view="frontal", tumor="1")
    normals = [
        _row("wrong", "g0", anatomy="femur", view="lateral"),
        _row("a", "g1", anatomy="tibia", view="frontal"),
        _row("b", "g2", anatomy="tibia", view="frontal"),
        _row("c", "g3", anatomy="tibia", view="frontal"),
        _row("d", "g4", anatomy="tibia", view="frontal"),
    ]
    first = select_normal_reference_pairs(query, normals)
    second = select_normal_reference_pairs(query, list(reversed(normals)))
    assert first == second
    selected = {
        item
        for pair in first
        for item in (pair.recipient_image_id, pair.sham_image_id)
    }
    assert selected == {"a", "b", "c", "d"}
    assert len(
        {
            item
            for pair in first
            for item in (pair.recipient_group_id, pair.sham_group_id)
        }
    ) == 4


def test_random_control_is_deterministic_but_not_metadata_sorted() -> None:
    query = _row("q", "gq", anatomy="tibia", view="frontal", tumor="1")
    normals = [
        _row(f"n{i}", f"g{i}", anatomy="tibia" if i < 4 else "femur", view="frontal")
        for i in range(12)
    ]
    first = select_random_normal_reference_pairs(query, normals)
    second = select_random_normal_reference_pairs(query, list(reversed(normals)))
    assert first == second
    assert len({pair.recipient_group_id for pair in first} | {pair.sham_group_id for pair in first}) == 4


def test_same_source_and_sham_cancel_every_candidate_geometry() -> None:
    source = torch.linspace(0.0, 1.0, 3 * 12 * 12).reshape(3, 12, 12)
    recipient = torch.flip(source, dims=(-1,))
    masks = torch.zeros((2, 12, 12))
    masks[0, 2:5, 3:7] = 1.0
    masks[1, 1:11, 1:11] = 1.0
    positive, sham = build_matched_transplants(
        source,
        recipient,
        source.clone(),
        masks,
    )
    assert torch.allclose(positive, sham)


def test_transplant_score_is_zero_for_identical_positive_and_sham_content() -> None:
    class MeanClassifier(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.mean(dim=(1, 2, 3), keepdim=False)[:, None]

    source = torch.full((3, 10, 10), 0.7)
    recipient = torch.full((3, 10, 10), 0.2)
    masks = torch.zeros((3, 10, 10))
    masks[0, 1:3, 1:3] = 1.0
    masks[1, 2:8, 2:8] = 1.0
    masks[2] = 1.0
    result = matched_transplant_scores(
        MeanClassifier(),
        source,
        masks,
        [(recipient, source.clone()), (recipient, source.clone())],
        imagenet_mean=(0.0, 0.0, 0.0),
        imagenet_std=(1.0, 1.0, 1.0),
    )
    assert torch.allclose(result["score"], torch.zeros(3), atol=1.0e-7)
    assert torch.allclose(result["recipient_std"], torch.zeros(3), atol=1.0e-7)


def test_percentile_rank_and_panel_are_fixed_and_baseline_preserving() -> None:
    baseline = np.asarray([1.0, 2.0, 3.0])
    transplant = np.asarray([3.0, 2.0, 1.0])
    control = np.asarray([1.0, 3.0, 2.0])
    assert np.allclose(percentile_ranks(baseline), [0.0, 0.5, 1.0])
    panel = frozen_selector_panel(baseline, transplant, control)
    assert np.allclose(panel["g1_upstream_baseline"], [0.0, 0.5, 1.0])
    assert np.allclose(panel["baseline_transplant_equal"], [0.5, 0.5, 0.5])
    assert set(panel) == {
        "g1_upstream_baseline",
        "transplant_only",
        "baseline_transplant_equal",
        "baseline_transplant_three_to_one",
        "baseline_random_control_three_to_one",
    }


def test_layerwise_diagnostic_is_finite_and_exactly_decomposes_logit() -> None:
    torch.manual_seed(4)
    classifier = _ToyDenseClassifier().eval()
    positive = torch.rand((3, 3, 32, 32))
    sham = torch.rand((3, 3, 32, 32))
    masks = torch.zeros((3, 32, 32))
    masks[0, 3:8, 4:11] = 1.0
    masks[1, 8:24, 7:21] = 1.0
    masks[2, 1:31, 1:31] = 1.0
    result = paired_dense_layer_diagnostics(
        classifier,
        positive,
        sham,
        masks,
        batch_size=2,
        imagenet_mean=(0.0, 0.0, 0.0),
        imagenet_std=(1.0, 1.0, 1.0),
    )
    assert result["stage_names"] == (
        "pool0",
        "transition1",
        "transition2",
        "transition3",
        "norm5",
    )
    assert result["feature_l2_inside"].shape == (3, 5)
    assert torch.isfinite(result["relative_feature_l2_inside"]).all()
    assert float(result["class_response_logit_residual"].abs().max()) < 1.0e-5


def test_layerwise_recipient_aggregation_preserves_candidate_count() -> None:
    torch.manual_seed(9)
    classifier = _ToyDenseClassifier().eval()
    source = torch.rand((3, 32, 32))
    masks = torch.zeros((2, 32, 32))
    masks[0, 2:7, 2:9] = 1.0
    masks[1, 9:27, 8:25] = 1.0
    references = [
        (torch.rand((3, 32, 32)), torch.rand((3, 32, 32))),
        (torch.rand((3, 32, 32)), torch.rand((3, 32, 32))),
    ]
    result = matched_transplant_layerwise_scores(
        classifier,
        source,
        masks,
        references,
        batch_size=1,
        imagenet_mean=(0.0, 0.0, 0.0),
        imagenet_std=(1.0, 1.0, 1.0),
    )
    assert result["score"].shape == (2,)
    assert result["recipient_std"].shape == (2,)
    assert result["feature_l2_inside_mean"].shape == (2, 5)
    assert torch.isfinite(result["class_response_logit_residual_mean"]).all()
