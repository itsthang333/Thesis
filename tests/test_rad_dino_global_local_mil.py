from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


torch = pytest.importorskip("torch")

from models.rad_dino_global_local_mil import (  # noqa: E402
    GlobalLocalMILConfig,
    RadDinoGlobalLocalMILDecoder,
    confidence_gated_rank_fusion,
    greedy_saliency_windows,
    local_mil_loss,
    random_diverse_windows,
    stitch_local_maps,
    top_fraction_pool,
)
from run_rad_dino_global_local_mil_probe import (  # noqa: E402
    crop_from_output_box,
    proposal_boxes,
)


def test_config_rejects_invalid_patch_contract() -> None:
    with pytest.raises(ValueError, match="inference"):
        GlobalLocalMILConfig(train_patches=3, inference_patches=4).validate()


def test_decoder_preserves_bag_axes() -> None:
    model = RadDinoGlobalLocalMILDecoder(GlobalLocalMILConfig(input_dim=64))
    tokens = torch.randn(2, 3, 3, 4, 4, 64)
    guidance = torch.randn(2, 3, 3, 8, 8)
    logits, features = model(tokens, guidance)
    assert logits.shape == (2, 3, 1, 8, 8)
    assert features.shape[:2] == (2, 3)


def test_top_fraction_pool_uses_all_patches_in_one_bag() -> None:
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 2, 4)
    valid = torch.ones((1, 2, 2, 4), dtype=torch.bool)
    pooled = top_fraction_pool(logits, valid, fraction=0.25)
    assert pooled.item() == pytest.approx((12 + 13 + 14 + 15) / 4)


def test_local_mil_loss_has_dense_normal_gradient() -> None:
    logits = torch.zeros((2, 2, 1, 3, 3), requires_grad=True)
    valid = torch.ones((2, 2, 3, 3), dtype=torch.bool)
    labels = torch.tensor([0.0, 1.0])
    loss, parts = local_mil_loss(
        logits,
        valid,
        labels,
        top_fraction=0.25,
        negative_dense_weight=0.5,
        positive_sparsity_weight=0.05,
    )
    loss.backward()
    assert parts["negative_dense"].item() > 0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0]) == logits[0].numel()


def test_greedy_windows_are_deterministic_and_diverse() -> None:
    saliency = np.zeros((32, 32), dtype=np.float32)
    saliency[2:10, 2:10] = 3.0
    saliency[22:30, 22:30] = 2.0
    first = greedy_saliency_windows(
        saliency,
        window_size=8,
        count=2,
        stride=2,
        iou_limit=0.25,
    )
    second = greedy_saliency_windows(
        saliency,
        window_size=8,
        count=2,
        stride=2,
        iou_limit=0.25,
    )
    assert first == second
    assert first[0] == (2, 2, 10, 10)
    assert first[1] == (22, 22, 30, 30)


def test_random_windows_are_seeded() -> None:
    kwargs = dict(
        output_shape=(32, 32),
        window_size=8,
        count=4,
        stride=4,
        iou_limit=0.25,
    )
    assert random_diverse_windows(**kwargs, seed=9) == random_diverse_windows(
        **kwargs, seed=9
    )
    assert random_diverse_windows(**kwargs, seed=9) != random_diverse_windows(
        **kwargs, seed=10
    )


def test_stitch_local_maps_averages_overlap() -> None:
    patches = np.stack(
        [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 3.0, np.float32)]
    )
    stitched, covered = stitch_local_maps(
        patches,
        [(0, 0, 4, 4), (2, 0, 6, 4)],
        output_shape=(4, 6),
    )
    assert np.allclose(stitched[:, :2], 1.0)
    assert np.allclose(stitched[:, 2:4], 2.0)
    assert np.allclose(stitched[:, 4:], 3.0)
    assert covered.all()


def test_fusion_retains_global_and_gates_local_peaks() -> None:
    global_map = np.full((10, 10), 0.2, dtype=np.float32)
    local_map = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
    coverage = np.ones((10, 10), dtype=bool)
    fused, gate = confidence_gated_rank_fusion(
        global_map,
        local_map,
        coverage,
        local_confidence=0.9,
        normal_confidence_p99=0.5,
        keep_fraction=0.02,
        residual_weight=0.35,
        temperature=0.1,
    )
    assert gate > 0.98
    assert np.all(fused >= global_map)
    assert np.count_nonzero(fused > global_map) == 2


def test_runner_has_no_segmentation_dataset_or_annotation_access() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "project/run_rad_dino_global_local_mil_probe.py"
    ).read_text(encoding="utf-8")
    assert "datasets.btxrd" not in source
    assert "Annotations" not in source
    assert 'split="test"' not in source


def test_content_box_crop_maps_from_320_geometry() -> None:
    image = Image.new("RGB", (640, 320))
    crop = crop_from_output_box(image, (80, 80, 240, 240), output_size=320)
    assert crop.size == (320, 160)


def test_positive_and_negative_proposals_use_different_sources() -> None:
    config = GlobalLocalMILConfig(
        proposal_size=16,
        proposal_stride=4,
        train_patches=3,
        inference_patches=2,
    )
    saliency = np.zeros((32, 32), dtype=np.float32)
    saliency[:16, :16] = 1.0
    positive, positive_source = proposal_boxes(
        {"image_id": "positive.jpeg", "tumor": "1"},
        global_map=saliency,
        config=config,
        output_size=32,
        seed=42,
    )
    negative, negative_source = proposal_boxes(
        {"image_id": "negative.jpeg", "tumor": "0"},
        global_map=None,
        config=config,
        output_size=32,
        seed=42,
    )
    assert len(positive) == len(negative) == 3
    assert positive_source == "frozen_global_top_mass"
    assert negative_source == "seeded_random_negative"
