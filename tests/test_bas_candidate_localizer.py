from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.models.bas_candidate_localizer import (
    BASForwardOutput,
    BASLossConfig,
    bas_activation_suppression_loss,
    candidate_activation_evidence,
    classifier_output_activation,
    equal_rank_aggregate,
    equal_rank_fusion,
    minmax_normalize_activation,
    within_bag_percentile_ranks,
)


def test_softplus_binary_transfer_preserves_nonnegative_map_and_negative_gradient() -> None:
    relu_leaf = torch.tensor([-2.0, -0.5], requires_grad=True)
    relu_input = relu_leaf * 1.0
    relu_output = classifier_output_activation("relu")(relu_input)
    relu_output.sum().backward()
    assert torch.count_nonzero(relu_output) == 0
    assert torch.count_nonzero(relu_leaf.grad) == 0

    softplus_input = torch.tensor([-2.0, -0.5], requires_grad=True)
    softplus_output = classifier_output_activation("softplus")(softplus_input)
    softplus_output.sum().backward()
    assert torch.all(softplus_output > 0)
    assert torch.all(softplus_input.grad > 0)


def test_unknown_classifier_output_activation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported BAS"):
        classifier_output_activation("sigmoid")  # type: ignore[arg-type]


def test_bas_loss_matches_background_ratio_and_area() -> None:
    output = BASForwardOutput(
        class_logits=torch.tensor([[2.0, 4.0], [3.0, 2.0]]),
        foreground_logits=torch.zeros(2, 2),
        class_activation_maps=torch.zeros(2, 2, 1, 1),
        localization_maps=torch.tensor(
            [[[[0.25, 0.25]]], [[[0.50, 0.50]]]]
        ),
        background_logits=torch.tensor([[1.0, 1.0], [4.0, 1.0]]),
    )
    loss = bas_activation_suppression_loss(
        output,
        torch.tensor([1, 0]),
        config=BASLossConfig(area_weight=1.2),
    )
    # Row 0: 1/4 + 1.2*0.25. Row 1 has background >= full, so ratio is zero.
    expected = ((0.25 + 0.3) + (0.0 + 0.6)) / 2.0
    assert float(loss) == pytest.approx(expected)


def test_bas_loss_keeps_epsilon_arithmetic_in_float32() -> None:
    output = BASForwardOutput(
        class_logits=torch.tensor([[0.0, 2.0]], dtype=torch.float16),
        foreground_logits=torch.zeros(1, 2, dtype=torch.float16),
        class_activation_maps=torch.zeros(1, 2, 1, 1, dtype=torch.float16),
        localization_maps=torch.full((1, 1, 2, 2), 0.25, dtype=torch.float16),
        background_logits=torch.tensor([[0.0, 1.0]], dtype=torch.float16),
    )
    loss = bas_activation_suppression_loss(output, torch.tensor([1]))
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(0.5 + 1.2 * 0.25, abs=1.0e-5)


def test_forward_output_is_namedtuple_for_data_parallel_gather() -> None:
    assert BASForwardOutput._fields == (
        "class_logits",
        "foreground_logits",
        "class_activation_maps",
        "localization_maps",
        "background_logits",
    )


def test_candidate_activation_evidence_balances_coverage_and_purity() -> None:
    activation = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    masks = torch.tensor(
        [[
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]]
    )
    valid = torch.tensor([[True, True, False]])
    coverage, purity, harmonic = candidate_activation_evidence(
        activation,
        masks,
        valid,
    )
    assert torch.allclose(coverage, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(purity, torch.tensor([[1.0, 0.5, 0.0]]))
    assert float(harmonic[0, 0]) == pytest.approx(1.0)
    assert float(harmonic[0, 1]) == pytest.approx(2.0 / 3.0)
    assert float(harmonic[0, 2]) == 0.0


def test_activation_normalization_is_per_image_and_constant_safe() -> None:
    activation = torch.tensor(
        [
            [[[2.0, 4.0], [6.0, 8.0]]],
            [[[3.0, 3.0], [3.0, 3.0]]],
        ]
    )
    normalized = minmax_normalize_activation(activation)
    assert torch.allclose(
        normalized[0],
        torch.tensor([[[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]]),
    )
    assert torch.count_nonzero(normalized[1]) == 0


def test_percentile_ranks_are_tie_aware_and_ignore_invalid() -> None:
    scores = torch.tensor([[3.0, 1.0, 3.0, 99.0]])
    valid = torch.tensor([[True, True, True, False]])
    ranks = within_bag_percentile_ranks(scores, valid)
    assert torch.allclose(ranks, torch.tensor([[0.75, 0.0, 0.75, 0.0]]))


def test_equal_rank_fusion_can_preserve_complementary_order() -> None:
    baseline = torch.tensor([[3.0, 2.0, 1.0]])
    activation = torch.tensor([[1.0, 3.0, 2.0]])
    valid = torch.ones_like(baseline, dtype=torch.bool)
    fused = equal_rank_fusion(baseline, activation, valid)
    assert torch.allclose(fused, torch.tensor([[0.5, 0.75, 0.25]]))
    assert int(fused.argmax(dim=1).item()) == 1


def test_equal_three_way_rank_aggregate_has_no_hidden_weight() -> None:
    first = torch.tensor([[3.0, 2.0, 1.0]])
    second = torch.tensor([[1.0, 3.0, 2.0]])
    third = torch.tensor([[2.0, 1.0, 3.0]])
    valid = torch.ones_like(first, dtype=torch.bool)
    combined = equal_rank_aggregate((first, second, third), valid)
    assert torch.allclose(combined, torch.full_like(first, 0.5))


def test_invalid_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="share shape"):
        equal_rank_fusion(
            torch.zeros(1, 2),
            torch.zeros(1, 3),
            torch.ones(1, 2, dtype=torch.bool),
        )
