from __future__ import annotations

import numpy as np
import pytest
import torch

from project.dsll_top3 import (
    average_percentile_ranks,
    conditional_disease_topk,
    normalize_cam,
    source_specific_candidate_features,
)


def test_conditional_top3_excludes_normal_and_sorts() -> None:
    logits = torch.tensor([[100.0, 1.0, 8.0, 3.0, 7.0, -1.0, 2.0, 0.0, 6.0, 4.0]])
    classes, probabilities = conditional_disease_topk(logits)
    assert classes.tolist() == [2, 4, 8]
    assert np.all(probabilities[:-1] >= probabilities[1:])
    assert np.all((probabilities > 0) & (probabilities < 1))


def test_normalize_before_late_fusion_contract() -> None:
    actual = normalize_cam(np.asarray([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(actual, [[0.0, 0.25], [0.5, 1.0]])


def test_average_percentile_ranks_handles_ties_and_singletons() -> None:
    values = np.asarray([0.1, 0.3, 0.3, 0.4])
    groups = np.asarray([1, 1, 1, 2])
    np.testing.assert_allclose(average_percentile_ranks(values, groups), [0.0, 0.75, 0.75, 1.0])


def test_source_specific_features_do_not_use_anchor_map() -> None:
    masks = np.asarray([
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
    ], dtype=np.uint8)
    maps = {
        10: np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        20: np.asarray([[0.0, 0.0], [0.0, 0.8]], dtype=np.float32),
    }
    means, mass, density, score = source_specific_candidate_features(
        masks, np.asarray([10, 20]), maps, np.asarray([0.2, 0.1])
    )
    np.testing.assert_allclose(means, [1.0, 0.8])
    np.testing.assert_allclose(mass, [1.0, 1.0])
    np.testing.assert_allclose(density, [1.0, 1.0])
    np.testing.assert_allclose(score, [1.0, 1.0])


def test_dsll_requires_ten_logits() -> None:
    with pytest.raises(ValueError):
        conditional_disease_topk(torch.zeros(1, 2))
