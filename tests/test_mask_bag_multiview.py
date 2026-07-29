from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    path = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_multiview.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mask_bag_multiview_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MULTIVIEW = _load_module()
candidate_tile_mass_retention = MULTIVIEW.candidate_tile_mass_retention
horizontal_flip_tile_mapping = MULTIVIEW.horizontal_flip_tile_mapping
maximum_retention_tile_weights = MULTIVIEW.maximum_retention_tile_weights
overlapping_corner_tiles = MULTIVIEW.overlapping_corner_tiles
weighted_local_descriptors = MULTIVIEW.weighted_local_descriptors


def test_multiview_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_multiview.py"
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


def test_corner_tiles_cover_the_complete_rectangular_image() -> None:
    boxes = overlapping_corner_tiles(
        image_height=8,
        image_width=12,
        crop_fraction=0.75,
    )
    assert boxes == (
        (0, 0, 9, 6),
        (3, 0, 12, 6),
        (0, 2, 9, 8),
        (3, 2, 12, 8),
    )
    coverage = np.zeros((8, 12), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] += 1
    assert np.all(coverage >= 1)


def test_mass_retention_selects_corner_and_averages_center_ties() -> None:
    boxes = overlapping_corner_tiles(
        image_height=8,
        image_width=12,
        crop_fraction=0.75,
    )
    masks = np.zeros((2, 8, 12), dtype=np.float32)
    masks[0, 0:2, 0:2] = 1
    masks[1, 3:5, 5:7] = 1
    retention = candidate_tile_mass_retention(masks, boxes)
    weights = maximum_retention_tile_weights(retention)
    assert retention[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert weights[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.allclose(retention[1], 1.0)
    assert np.allclose(weights[1], 0.25)


def test_tile_retention_and_weights_are_horizontal_flip_equivariant() -> None:
    boxes = overlapping_corner_tiles(
        image_height=10,
        image_width=14,
        crop_fraction=0.7,
    )
    mapping = horizontal_flip_tile_mapping(boxes, image_width=14)
    assert mapping.tolist() == [1, 0, 3, 2]
    masks = np.zeros((2, 10, 14), dtype=np.float32)
    masks[0, 1:4, 1:6] = 1
    masks[1, 4:8, 5:9] = 1
    original = candidate_tile_mass_retention(masks, boxes)
    flipped = candidate_tile_mass_retention(masks[..., ::-1], boxes)
    assert np.allclose(original, flipped[:, mapping])
    original_weights = maximum_retention_tile_weights(original)
    flipped_weights = maximum_retention_tile_weights(flipped)
    assert np.allclose(original_weights, flipped_weights[:, mapping])


def test_weighted_local_descriptors_preserve_tied_view_symmetry() -> None:
    descriptors = np.array(
        [
            [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    fused = weighted_local_descriptors(descriptors, weights)
    assert np.allclose(fused, [[2.0, 0.0], [0.0, 8.0]])
