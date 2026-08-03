from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import project.audit_skelex_candidate_marginal_s9_output as auditor
from project.models.skelex_candidate_marginal import (
    NonlinearTokenEvidenceHead,
    SKELEX_TOKEN_DIM,
    candidate_spatial_log_likelihood,
)
import project.run_skelex_candidate_marginal_s9 as producer


def test_reference_projection_is_byte_identical_to_producer() -> None:
    masks = np.zeros((2, 48, 80), dtype=np.uint8)
    masks[0, 10:20, 15:30] = 1
    masks[1, 25:42, 40:65] = 1
    projection = SimpleNamespace(padded_side=80, content_box=(0, 16, 80, 64))
    expected = producer._project_supports(masks, projection=projection)
    observed = auditor._reference_supports(masks, projection=projection)
    for first, second in zip(expected, observed):
        np.testing.assert_array_equal(first, second)


def test_reference_head_and_likelihood_are_independent_numeric_matches() -> None:
    torch.manual_seed(42)
    model = NonlinearTokenEvidenceHead().eval()
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.01)
        model.output.bias.fill_(0.03)
    rng = np.random.default_rng(42)
    tokens = rng.normal(size=(4, SKELEX_TOKEN_DIM)).astype(np.float16)
    candidates = rng.uniform(0.0, 1.0, size=(3, 4)).astype(np.float16)
    rings = rng.uniform(0.0, 1.0, size=(3, 4)).astype(np.float16)
    with torch.inference_mode():
        logits = model(torch.from_numpy(tokens)[None])[0].numpy().astype(np.float32)
        likelihood = candidate_spatial_log_likelihood(
            torch.from_numpy(logits)[None],
            torch.from_numpy(candidates.astype(np.float32))[None],
            torch.from_numpy(rings.astype(np.float32))[None],
            torch.ones((1, 3), dtype=torch.bool),
        )[0].numpy()
    observed_logits = auditor._reference_head_logits(tokens, model.state_dict(), torch.device("cpu"))
    observed_likelihood = auditor._reference_likelihood(observed_logits, candidates, rings)
    np.testing.assert_allclose(observed_logits, logits, atol=1.0e-6, rtol=0)
    np.testing.assert_allclose(observed_likelihood, likelihood, atol=1.0e-6, rtol=0)


def test_reference_rank_handles_ties_and_singleton() -> None:
    np.testing.assert_array_equal(auditor._rank(np.asarray([4.0])), [1.0])
    np.testing.assert_array_equal(
        auditor._rank(np.asarray([2.0, 1.0, 2.0, 3.0])),
        [0.5, 0.0, 0.5, 1.0],
    )


def test_s9_auditor_composes_arms_only_with_canonical_aggregate() -> None:
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    composition = source[source.index("rank_inputs = tuple(") :]
    assert "base.equal_rank_aggregate(rank_inputs[:2], rank_valid)" in composition
    assert "base.equal_rank_aggregate(rank_inputs, rank_valid)" in composition
    assert "0.5 * (base_rank + upstream_rank)" not in composition
