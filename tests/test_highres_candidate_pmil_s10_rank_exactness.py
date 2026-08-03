from __future__ import annotations

import numpy as np
import torch

import project.audit_highres_candidate_pmil_s10_output as auditor
from project.models.bas_candidate_localizer import equal_rank_aggregate


def test_float32_rank_vector_and_argmax_exact_for_every_candidate_count() -> None:
    rng = np.random.default_rng(20261206)
    for count in range(1, 82):
        # Small integer support deliberately creates repeated ties and fused ties.
        values = [rng.integers(-3, 4, size=count).astype(np.float32) for _ in range(3)]
        valid = torch.ones(1, count, dtype=torch.bool)
        producer = equal_rank_aggregate(
            tuple(torch.from_numpy(value)[None] for value in values), valid
        )[0].numpy()
        independent = np.stack(
            tuple(auditor._rank(value) for value in values), axis=0
        ).mean(axis=0, dtype=np.float32)
        np.testing.assert_array_equal(independent, producer)
        assert int(np.argmax(independent)) == int(np.argmax(producer))
