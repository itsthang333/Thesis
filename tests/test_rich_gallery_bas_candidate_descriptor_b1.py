from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from project.run_rich_gallery_bas_candidate_descriptor_b1 import (
    VARIANTS,
    build_variant_scores,
    canonical_source,
)


def test_variant_scores_preserve_baseline_and_fixed_factorial_arms() -> None:
    g1 = np.asarray([3.0, 1.0, 2.0], dtype=np.float64)
    upstream = np.asarray([0.0, 2.0, 1.0], dtype=np.float64)
    bas = np.asarray([1.0, 0.0, 2.0], dtype=np.float64)
    scores = build_variant_scores(g1, upstream, bas)
    assert tuple(scores) == VARIANTS
    assert np.allclose(scores["g1_upstream_baseline"], [0.5, 0.5, 0.5])
    assert np.allclose(scores["bas_only"], [0.5, 0.0, 1.0])
    assert np.allclose(scores["g1_bas_two_way"], [0.75, 0.0, 0.75])
    assert np.allclose(scores["upstream_bas_two_way"], [0.25, 0.5, 0.75])
    assert np.allclose(scores["g1_upstream_bas_three_way"], [0.5, 1.0 / 3.0, 2.0 / 3.0])


def test_variant_scores_reject_shape_and_nonfinite_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        build_variant_scores(np.ones(2), np.ones(3), np.ones(2))
    with np.testing.assert_raises(ValueError):
        build_variant_scores(np.asarray([1.0, np.nan]), np.ones(2), np.ones(2))


def test_source_normalization_is_closed() -> None:
    assert canonical_source("classifier448") == "classifier448"
    assert canonical_source("external_biomed") == "external_saliency"
    assert canonical_source("layercam_anchor") == "layercam320"
    with np.testing.assert_raises(ValueError):
        canonical_source("unknown")


def test_runner_has_no_segmentation_dataset_or_test_path() -> None:
    path = Path(__file__).resolve().parents[1] / "project" / "run_rich_gallery_bas_candidate_descriptor_b1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "build_segmentation_dataset" not in imported
    assert "polygon" not in source.lower()
    assert "split=\"test\"" not in source
    assert "diagnostics_do_not_block_spatial_evaluation" in source
