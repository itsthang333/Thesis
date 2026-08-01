from __future__ import annotations

import numpy as np
import torch

from project.models.matched_normal_candidate_transplant import (
    build_matched_transplants,
    frozen_selector_panel,
    matched_transplant_scores,
    percentile_ranks,
    select_normal_reference_pairs,
)


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
