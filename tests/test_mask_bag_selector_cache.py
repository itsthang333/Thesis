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
        / "mask_bag_selector_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mask_bag_selector_cache_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CACHE = _load_module()
candidate_shape_features = CACHE.candidate_shape_features
encode_candidate_families = CACHE.encode_candidate_families
pack_candidate_masks = CACHE.pack_candidate_masks
pairwise_overlap_geometry = CACHE.pairwise_overlap_geometry
unpack_candidate_masks = CACHE.unpack_candidate_masks


def test_selector_cache_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_selector_cache.py"
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


def test_family_ids_use_all_immutable_provenance_and_kept_indices() -> None:
    families, table = encode_candidate_families(
        np.array([4, 4, 4, 7]),
        np.array(["box", "box", "point", "box"]),
        np.array(["cam", "cam", "cam", "teacher"]),
        kept_indices=np.array([3, 0, 2]),
    )
    assert len(np.unique(families)) == 3
    by_key = {
        (
            row["proposal_source"],
            row["prompt_mode"],
            row["component_id"],
        ): row["original_candidate_indices"]
        for row in table
    }
    assert by_key[("cam", "box", 4)] == [0]
    assert by_key[("cam", "point", 4)] == [2]
    assert by_key[("teacher", "box", 7)] == [3]


def test_identical_candidates_in_one_family_share_family_id() -> None:
    families, table = encode_candidate_families(
        np.array([2, 2, 2]),
        np.array(["box_point", "box_point", "box_point"]),
        np.array(["cam", "cam", "cam"]),
    )
    assert families.tolist() == [0, 0, 0]
    assert table[0]["candidate_count"] == 3


def test_shape_features_are_position_free() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 1:3, 1:5] = 1
    masks[1, 5:7, 3:7] = 1
    features = candidate_shape_features(masks)
    assert np.array_equal(features[0], features[1])


def test_pairwise_geometry_is_symmetric_and_normalized() -> None:
    masks = np.zeros((3, 8, 8), dtype=np.uint8)
    masks[0, 1:5, 1:5] = 1
    masks[1, 2:6, 2:6] = 1
    masks[2, 6:8, 6:8] = 1
    iou, containment, distance = pairwise_overlap_geometry(masks)
    for matrix in (iou, containment, distance):
        assert np.allclose(matrix, matrix.T)
        assert np.isfinite(matrix).all()
    assert np.allclose(np.diag(iou), 1.0)
    assert np.allclose(np.diag(containment), 1.0)
    assert np.allclose(np.diag(distance), 0.0)
    assert iou[0, 1] > iou[0, 2]
    assert distance[0, 1] < distance[0, 2]


def test_bitpacked_masks_round_trip_exactly() -> None:
    masks = np.zeros((2, 7, 9), dtype=np.uint8)
    masks[0, 1:4, 2:8] = 1
    masks[1, 3:7, 0:5] = 1
    packed = pack_candidate_masks(masks)
    restored = unpack_candidate_masks(packed)
    assert np.array_equal(restored, masks)
    assert packed.packed.nbytes < masks.nbytes
