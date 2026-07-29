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
        / "mask_bag_crossfit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mask_bag_crossfit_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CROSSFIT = _load_module()
assign_group_stratified_folds = CROSSFIT.assign_group_stratified_folds
audit_crossfit_training_exclusion = CROSSFIT.audit_crossfit_training_exclusion
crossfit_assignment_manifest = CROSSFIT.crossfit_assignment_manifest


def test_crossfit_surface_is_gt_and_subgroup_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "models"
        / "mask_bag_crossfit.py"
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


def test_group_stratified_folds_are_deterministic_and_group_preserving() -> None:
    groups = np.array(
        ["n0", "n0", "n1", "n2", "n3", "p0", "p0", "p1", "p2", "p3"]
    )
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    first = assign_group_stratified_folds(
        labels,
        groups,
        fold_count=2,
        seed=42,
    )
    second = assign_group_stratified_folds(
        labels,
        groups,
        fold_count=2,
        seed=42,
    )
    assert np.array_equal(first, second)
    for group in np.unique(groups):
        assert len(np.unique(first[groups == group])) == 1
    for fold in (0, 1):
        assert set(labels[first == fold].tolist()) == {0, 1}


def test_group_stratification_balances_class_image_counts() -> None:
    groups = np.asarray(
        ["n0"] * 5
        + ["n1"] * 4
        + ["n2"] * 3
        + ["n3"] * 2
        + ["p0"] * 5
        + ["p1"] * 4
        + ["p2"] * 3
        + ["p3"] * 2
    )
    labels = np.asarray([0] * 14 + [1] * 14)
    folds = assign_group_stratified_folds(
        labels,
        groups,
        fold_count=2,
        seed=7,
    )
    for label in (0, 1):
        counts = [int(((folds == fold) & (labels == label)).sum()) for fold in (0, 1)]
        assert abs(counts[0] - counts[1]) <= 1


def test_crossfit_manifest_is_order_independent() -> None:
    images = np.array(["c.png", "a.png", "b.png", "d.png"])
    groups = np.array(["g2", "g0", "g1", "g3"])
    labels = np.array([1, 0, 0, 1])
    folds = np.array([1, 0, 1, 0])
    first = crossfit_assignment_manifest(images, groups, labels, folds)
    order = np.array([2, 0, 3, 1])
    second = crossfit_assignment_manifest(
        images[order],
        groups[order],
        labels[order],
        folds[order],
    )
    assert first == second


def test_crossfit_exclusion_rejects_training_overlap() -> None:
    groups = ["a", "b", "c", "d"]
    folds = [0, 0, 1, 1]
    accepted = audit_crossfit_training_exclusion(
        groups,
        folds,
        {0: ["c", "d"], 1: ["a", "b"]},
    )
    assert accepted["complete"]
    try:
        audit_crossfit_training_exclusion(
            groups,
            folds,
            {0: ["a", "c", "d"], 1: ["a", "b"]},
        )
    except RuntimeError as error:
        assert "trained on held-out" in str(error)
    else:
        raise AssertionError("training/held-out overlap was not rejected")
