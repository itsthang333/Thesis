from __future__ import annotations

import inspect

import numpy as np
import torch

from project.models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
from project.run_rich_gallery_cross_view_cowitness_pair import (
    RESIDUAL_MULTIPLIERS,
    _appearance,
    attach_immutable_baseline,
    frozen_variants,
    variant_spec,
)


def _record() -> dict[str, object]:
    rng = np.random.default_rng(3)
    return {
        "image_id": "x.jpeg",
        "descriptors": rng.normal(size=(5, 1156)).astype(np.float32),
        "flipped_descriptors": rng.normal(size=(5, 1156)).astype(np.float32),
        "upstream_scores": np.linspace(0.1, 0.9, 5, dtype=np.float32),
    }


def test_appearance_drops_all_four_candidate_metadata_fields() -> None:
    record = _record()
    appearance = _appearance(record)
    assert appearance.shape == (5, 1152)
    changed = dict(record)
    changed["descriptors"] = np.asarray(record["descriptors"]).copy()
    changed["flipped_descriptors"] = np.asarray(record["flipped_descriptors"]).copy()
    changed["descriptors"][:, -4:] = 9999.0
    changed["flipped_descriptors"][:, -4:] = -9999.0
    assert np.array_equal(appearance, _appearance(changed))


def test_immutable_baseline_is_finite_and_zero_residual_stable() -> None:
    config = MaskBagMILConfig(token_dim=128, token_layers=3)
    model = RadDinoMaskBagMIL(config)
    cache = [_record()]
    report = attach_immutable_baseline(cache, model, torch.device("cpu"))
    assert report["zero_residual_maximum_local_choice_delta"] == 0
    assert np.isfinite(cache[0]["baseline_scores"]).all()
    assert np.asarray(cache[0]["appearance"]).shape == (5, 1152)


def test_runner_has_no_segmentation_annotation_dependency() -> None:
    import project.run_rich_gallery_cross_view_cowitness_pair as module

    source = inspect.getsource(module)
    assert "annotations_path" not in source
    assert "load_polygons" not in source
    assert "segmentation_gt" not in source


def test_exploratory_residual_scale_grid_is_global_and_predeclared() -> None:
    assert RESIDUAL_MULTIPLIERS == (0.25, 0.50, 1.00, 2.00)
    variants = frozen_variants()
    assert variants[0] == "baseline"
    assert len(variants) == 1 + 2 * len(RESIDUAL_MULTIPLIERS)
    assert len(set(variants)) == len(variants)
    assert variant_spec("baseline") == (None, 0.0)
    assert variant_spec("full__residual_x2") == ("full", 2.0)
