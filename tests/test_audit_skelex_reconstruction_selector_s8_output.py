from __future__ import annotations

import numpy as np

from project.audit_skelex_reconstruction_selector_s8_output import _rank_serialized_lcb


def test_s8_serialized_float32_rank_preserves_tie_boundary() -> None:
    values = np.asarray(
        [-0.0120913380, -0.01209133805, -0.0120913381, -0.5],
        dtype=np.float64,
    )
    valid = np.ones(4, dtype=bool)
    ranks = _rank_serialized_lcb(values, valid)
    assert ranks[0] == ranks[1] == ranks[2]
    assert ranks[3] == 0.0
