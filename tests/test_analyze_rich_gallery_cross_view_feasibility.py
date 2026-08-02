from __future__ import annotations

import numpy as np

from project.analyze_rich_gallery_cross_view_feasibility import (
    cosine_max_support,
)


def test_cosine_max_support_finds_best_reference_candidate() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reference = np.asarray([[0.8, 0.2], [-1.0, 0.0]], dtype=np.float32)
    support = cosine_max_support(query, reference)
    assert support.shape == (2,)
    assert support[0] > 0.9
    assert 0.2 < support[1] < 0.3


def test_cosine_max_support_is_scale_invariant() -> None:
    query = np.asarray([[3.0, 4.0]], dtype=np.float32)
    reference = np.asarray([[6.0, 8.0]], dtype=np.float32)
    assert np.allclose(cosine_max_support(query, reference), [1.0])


def test_cosine_max_support_rejects_incompatible_descriptor_width() -> None:
    with np.testing.assert_raises(ValueError):
        cosine_max_support(np.ones((2, 3)), np.ones((2, 4)))
