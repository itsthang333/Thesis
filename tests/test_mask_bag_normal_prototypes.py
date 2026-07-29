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
        / "mask_bag_normal_prototypes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mask_bag_normal_prototypes_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOTYPES = _load_module()
fit_weighted_spherical_prototypes = (
    PROTOTYPES.fit_weighted_spherical_prototypes
)
hierarchical_image_family_weights = (
    PROTOTYPES.hierarchical_image_family_weights
)
normal_prototype_features = PROTOTYPES.normal_prototype_features


def test_normal_prototype_surface_is_dataset_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_normal_prototypes.py"
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


def test_hierarchical_weights_equalize_images_and_families() -> None:
    image_ids = np.array([0, 0, 0, 1, 1])
    family_ids = np.array([0, 0, 1, 0, 0])
    weights = hierarchical_image_family_weights(image_ids, family_ids)
    assert np.isclose(weights[image_ids == 0].sum(), 0.5)
    assert np.isclose(weights[image_ids == 1].sum(), 0.5)
    assert np.isclose(weights[(image_ids == 0) & (family_ids == 0)].sum(), 0.25)
    assert np.isclose(weights[(image_ids == 0) & (family_ids == 1)].sum(), 0.25)


def test_identical_within_family_duplication_preserves_weighted_mean() -> None:
    original = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    original_weights = hierarchical_image_family_weights(
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
    )
    duplicated = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    duplicated_weights = hierarchical_image_family_weights(
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 0]),
    )
    assert np.allclose(
        (original * original_weights[:, None]).sum(axis=0),
        (duplicated * duplicated_weights[:, None]).sum(axis=0),
    )


def test_weighted_spherical_prototypes_are_seeded_and_normalized() -> None:
    descriptors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    weights = np.full(4, 0.25, dtype=np.float32)
    first, assignments = fit_weighted_spherical_prototypes(
        descriptors,
        weights,
        prototype_count=2,
        seed=17,
    )
    second, second_assignments = fit_weighted_spherical_prototypes(
        descriptors,
        weights,
        prototype_count=2,
        seed=17,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(assignments, second_assignments)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert set(assignments.tolist()) == {0, 1}


def test_prototype_features_rank_unseen_direction_as_more_abnormal() -> None:
    prototypes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    descriptors = np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    features = normal_prototype_features(
        descriptors,
        prototypes,
        temperature=0.1,
    )
    assert features.shape == (2, 4)
    assert features[1, 0] > features[0, 0]
    assert features[1, 1] > features[0, 1]
    assert np.isfinite(features).all()
