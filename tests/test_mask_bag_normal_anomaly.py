from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from models.mask_bag_normal_anomaly import (
    DirectNormalAnomalyConfig,
    fit_direct_normal_anomaly_bank,
    normal_bank_training_arrays,
    score_direct_normal_anomaly,
)


def _record(image_id: str, descriptors: list[list[float]], families: list[int]) -> dict[str, object]:
    values = np.asarray(descriptors, dtype=np.float32)
    return {
        "image_id": image_id,
        "image_label": 0,
        "descriptors": values,
        "flipped_descriptors": values.copy(),
        "family_ids": np.asarray(families, dtype=np.int32),
    }


def test_normal_bank_weights_images_families_candidates_and_views_hierarchically() -> None:
    records = [
        _record("a", [[1, 0], [0, 1], [1, 1]], [0, 0, 1]),
        _record("b", [[1, -1]], [5]),
    ]
    _values, weights, audit = normal_bank_training_arrays(records)
    assert audit["normal_images"] == 2
    assert audit["normal_candidates"] == 4
    assert audit["normal_candidate_views"] == 8
    # Image a and b each receive one half of all nominal mass.
    assert weights[:6].sum() == pytest.approx(0.5)
    assert weights[6:].sum() == pytest.approx(0.5)
    # Within image a, both families receive equal mass despite unequal counts.
    assert weights[[0, 1, 3, 4]].sum() == pytest.approx(0.25)
    assert weights[[2, 5]].sum() == pytest.approx(0.25)


def test_normal_bank_rejects_positive_image_label() -> None:
    record = _record("positive", [[1, 0]], [0])
    record["image_label"] = 1
    with pytest.raises(ValueError, match="only image-label-normal"):
        normal_bank_training_arrays([record])


def test_direct_anomaly_score_selects_farthest_candidate_and_averages_views() -> None:
    prototypes = np.asarray([[1.0, 0.0]], dtype=np.float32)
    original = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    flipped = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    result = score_direct_normal_anomaly(original, flipped, prototypes)
    assert np.array_equal(result["original_normal_distance"], np.asarray([0.0, 1.0], dtype=np.float32))
    assert np.array_equal(result["flipped_normal_distance"], np.asarray([0.0, 2.0], dtype=np.float32))
    assert np.array_equal(result["candidate_scores"], np.asarray([0.0, 1.5], dtype=np.float32))
    assert result["selected_candidate_position"] == 1
    assert result["view_selected_agreement"] == 1


def test_fit_requires_frozen_k_and_seed() -> None:
    records = [_record("a", [[1, 0], [0, 1]], [0, 1])]
    with pytest.raises(ValueError, match="frozen K=32"):
        fit_direct_normal_anomaly_bank(
            records,
            config=DirectNormalAnomalyConfig(prototype_count=1),
        )
