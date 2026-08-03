from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from project.models.skelex_candidate_marginal import (
    CosineTokenEvidenceHead,
    candidate_marginal_image_label_loss,
    candidate_spatial_log_likelihood,
    finite_readout,
    normalized_candidate_logmeanexp,
)


def test_candidate_likelihood_matches_manual_fractional_reference() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    candidates = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]]])
    rings = torch.tensor([[[0.0, 0.5, 1.0], [1.0, 0.75, 0.0]]])
    valid = torch.tensor([[True, True]])
    actual = candidate_spatial_log_likelihood(logits, candidates, rings, valid)
    expected = []
    for candidate, ring in zip(candidates[0], rings[0]):
        inside = (candidate * F.logsigmoid(logits[0])).sum() / candidate.sum()
        outside = (ring * F.logsigmoid(-logits[0])).sum() / ring.sum()
        expected.append(0.5 * (inside + outside))
    assert torch.allclose(actual[0], torch.stack(expected))


def test_logmeanexp_is_normalized_and_candidate_order_invariant() -> None:
    values = torch.tensor([[-2.0, -1.0, -9.0]])
    valid = torch.tensor([[True, True, False]])
    expected = torch.logsumexp(values[0, :2], dim=0) - math.log(2)
    assert torch.allclose(normalized_candidate_logmeanexp(values, valid), expected[None])
    permutation = torch.tensor([1, 2, 0])
    assert torch.allclose(
        normalized_candidate_logmeanexp(values[:, permutation], valid[:, permutation]),
        expected[None],
    )


def test_image_loss_uses_dense_normals_and_soft_tumor_marginal() -> None:
    logits = torch.tensor([[1.0, -1.0], [0.4, -0.2]], requires_grad=True)
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    rings = 1.0 - candidates
    valid = torch.ones((2, 2), dtype=torch.bool)
    output = candidate_marginal_image_label_loss(
        logits,
        torch.tensor([0, 1]),
        candidates,
        rings,
        valid,
        torch.ones_like(logits, dtype=torch.bool),
    )
    expected_normal = F.softplus(logits[0]).mean()
    expected_tumor = -normalized_candidate_logmeanexp(
        output["candidate_likelihood"][1:2], valid[1:2]
    )[0]
    assert torch.allclose(output["total"], 0.5 * (expected_normal + expected_tumor))
    output["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_likelihood_fails_closed_for_empty_valid_ring() -> None:
    with pytest.raises(ValueError, match="zero ring mass"):
        candidate_spatial_log_likelihood(
            torch.zeros((1, 2)),
            torch.tensor([[[1.0, 0.0]]]),
            torch.zeros((1, 1, 2)),
            torch.tensor([[True]]),
        )


def test_token_head_is_one_direction_and_zero_initialized() -> None:
    head = CosineTokenEvidenceHead(3)
    tokens = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    assert torch.equal(head(tokens), torch.zeros((1, 2)))
    assert sum(parameter.numel() for parameter in head.parameters()) == 4


def test_finite_readout_reproduces_control_and_adds_one_equal_rank() -> None:
    output = finite_readout(
        np.asarray([1.0, 3.0, 2.0]),
        np.asarray([2.0, 1.0, 3.0]),
        np.asarray([3.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(output["control"], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(output["primary"], [0.5, 1.0 / 3.0, 2.0 / 3.0])

