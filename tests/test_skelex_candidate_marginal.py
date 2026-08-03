from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from project.models.bas_candidate_localizer import equal_rank_aggregate

from project.models.skelex_candidate_marginal import (
    NonlinearTokenEvidenceHead,
    SKELEX_HEAD_HIDDEN_DIM,
    SKELEX_HIDDEN_LAYERS,
    SKELEX_HIDDEN_SIZE,
    SKELEX_PATCHES,
    SKELEX_TOKEN_DIM,
    SkelexMultiLayerTokenEncoder,
    candidate_marginal_image_label_loss,
    candidate_spatial_log_likelihood,
    finite_readout,
    fractional_candidate_ring_supports,
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


def test_token_head_is_nonlinear_bounded_and_zero_initialized() -> None:
    torch.manual_seed(42)
    head = NonlinearTokenEvidenceHead(feature_dim=6, hidden_dim=2, layer_dim=3)
    tokens = torch.tensor(
        [[[3.0, 0.0, 0.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0, 5.0]]]
    )
    assert torch.equal(head(tokens), torch.zeros((1, 2)))
    assert sum(parameter.numel() for parameter in head.parameters()) == 17
    head(tokens).sum().backward()
    assert head.output.weight.grad is not None
    assert float(head.output.weight.grad.abs().sum()) > 0


def test_finite_readout_reproduces_control_and_adds_one_equal_rank() -> None:
    output = finite_readout(
        np.asarray([1.0, 3.0, 2.0]),
        np.asarray([2.0, 1.0, 3.0]),
        np.asarray([3.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(output["control"], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(output["primary"], [0.5, 1.0 / 3.0, 2.0 / 3.0])


def test_single_candidate_readout_matches_accepted_rank_contract() -> None:
    output = finite_readout(
        np.asarray([4.0]),
        np.asarray([-3.0]),
        np.asarray([0.2]),
    )
    np.testing.assert_array_equal(output["control"], [1.0])
    np.testing.assert_array_equal(output["primary"], [1.0])


def test_finite_readout_is_byte_identical_for_all_candidate_counts() -> None:
    for count in range(1, 82):
        for case in range(10):
            generator = np.random.default_rng(10_000 * count + case)
            arrays = [
                generator.normal(size=count).astype(np.float32)
                for _ in range(3)
            ]
            if count > 2:
                arrays[0][1] = arrays[0][0]
            if count > 4 and case % 2 == 0:
                arrays[1][3:5] = arrays[1][2]
            observed = finite_readout(*arrays)
            tensors = tuple(torch.from_numpy(values)[None] for values in arrays)
            valid = torch.ones((1, count), dtype=torch.bool)
            control = equal_rank_aggregate(tensors[:2], valid)[0].numpy()
            primary = equal_rank_aggregate(tensors, valid)[0].numpy()
            np.testing.assert_array_equal(observed["control"], control)
            np.testing.assert_array_equal(observed["primary"], primary)
            assert int(np.argmax(observed["control"])) == int(np.argmax(control))
            assert int(np.argmax(observed["primary"])) == int(np.argmax(primary))


def test_finite_readout_preserves_canonical_winner_at_float32_fused_tie() -> None:
    geometry = np.asarray(
        [1.6420046, -0.10410781, -0.03629668, 1.324037, -0.42114514, 0.5349967],
        dtype=np.float32,
    )
    upstream = np.asarray(
        [-0.6356979, 0.42562026, 1.623146, 0.3142767, -0.13453506, 0.43792695],
        dtype=np.float32,
    )
    likelihood = np.arange(6, dtype=np.float32)
    observed = finite_readout(geometry, upstream, likelihood)["control"]
    assert observed.tolist() == [
        0.5,
        0.4000000059604645,
        0.699999988079071,
        0.6000000238418579,
        0.10000000149011612,
        0.7000000476837158,
    ]
    assert int(np.argmax(observed)) == 5


def test_fractional_supports_preserve_candidate_and_construct_local_ring() -> None:
    candidates = torch.zeros((1, 512, 512))
    candidates[:, 240:272, 240:272] = 1.0
    content = torch.ones((512, 512))
    inside, ring, valid = fractional_candidate_ring_supports(candidates, content)
    assert inside.shape == ring.shape == (1, SKELEX_PATCHES)
    assert valid.shape == (SKELEX_PATCHES,) and valid.all()
    assert float(inside.sum()) == pytest.approx(4.0)
    assert float(ring.sum()) > float(inside.sum())
    assert not bool(((inside > 0) & (ring > 0)).any())


def test_fractional_supports_fail_closed_for_full_content_candidate() -> None:
    with pytest.raises(ValueError, match="zero projected ring mass"):
        fractional_candidate_ring_supports(
            torch.ones((1, 320, 320)),
            torch.ones((320, 320)),
            grid_size=20,
        )


class _FakeOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _FakeEncoder(nn.Module):
    def forward(self, **kwargs: object) -> _FakeOutput:
        pixels = kwargs["pixel_values"]
        assert isinstance(pixels, torch.Tensor)
        assert kwargs["interpolate_pos_encoding"] is True
        noise = kwargs["noise"]
        assert isinstance(noise, torch.Tensor)
        assert torch.equal(noise[0].cpu(), torch.arange(SKELEX_PATCHES))
        hidden = tuple(
            torch.full(
                (len(pixels), SKELEX_PATCHES + 1, SKELEX_HIDDEN_SIZE),
                float(layer),
            )
            for layer in range(25)
        )
        return _FakeOutput(hidden)


def test_layer_encoder_concatenates_only_frozen_intermediate_layers() -> None:
    output = SkelexMultiLayerTokenEncoder(_FakeEncoder())(
        torch.zeros((2, 3, 512, 512))
    )
    assert output.shape == (2, SKELEX_PATCHES, SKELEX_TOKEN_DIM)
    chunks = output.reshape(2, SKELEX_PATCHES, len(SKELEX_HIDDEN_LAYERS), SKELEX_HIDDEN_SIZE)
    for offset, layer in enumerate(SKELEX_HIDDEN_LAYERS):
        assert torch.equal(chunks[:, :, offset], torch.full_like(chunks[:, :, offset], float(layer)))


def test_default_head_capacity_is_frozen() -> None:
    head = NonlinearTokenEvidenceHead()
    expected = (
        SKELEX_TOKEN_DIM * SKELEX_HEAD_HIDDEN_DIM
        + SKELEX_HEAD_HIDDEN_DIM
        + SKELEX_HEAD_HIDDEN_DIM
        + 1
    )
    assert sum(parameter.numel() for parameter in head.parameters()) == expected
    assert expected == 524_801
