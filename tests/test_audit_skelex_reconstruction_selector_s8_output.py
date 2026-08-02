from __future__ import annotations

import numpy as np
import torch

from project.audit_skelex_reconstruction_selector_s8_output import (
    _null_improvements,
    _rank_serialized_lcb,
)


def test_s8_serialized_float32_rank_preserves_tie_boundary() -> None:
    values = np.asarray(
        [-0.0120913380, -0.01209133805, -0.0120913381, -0.5],
        dtype=np.float64,
    )
    valid = np.ones(4, dtype=bool)
    ranks = _rank_serialized_lcb(values, valid)
    assert ranks[0] == ranks[1] == ranks[2]
    assert ranks[3] == 0.0


def test_s8_null_replay_stays_on_producer_cpu_when_cuda_is_available(monkeypatch) -> None:
    original_to = torch.Tensor.to

    def reject_cuda(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            raise AssertionError("independent null replay must match producer CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", reject_cuda)
    errors = np.asarray(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.4, 0.3], [0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    observed = np.ones((2, 2, 2), dtype=bool)
    candidates = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    result = _null_improvements(
        errors,
        observed,
        candidates,
        np.ones((2, 2), dtype=np.float32),
        np.asarray([1.0, 0.5], dtype=np.float32),
        0,
    )
    assert result.shape == (255,)
    assert np.isfinite(result).all()
