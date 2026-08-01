from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from project.models.rich_gallery_g2_objective import (
    average_percentile_rank,
    geometric_continuation_temperature,
    hierarchical_source_candidate_weights,
    hierarchical_source_smooth_pool,
    negative_bag_instance_loss,
    rank_fusion_scores,
    shared_source_validity,
    stable_select,
)
from project.models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
from project.run_rich_gallery_g2_selector_pair import ARM_NAMES, train_arm


def test_hierarchical_pool_is_invariant_to_duplicate_identical_candidate() -> None:
    first, _ = hierarchical_source_smooth_pool(
        torch.tensor([[2.0, 0.0]]),
        torch.ones((1, 2), dtype=torch.bool),
        torch.tensor([[0, 1]]),
        temperature=0.5,
    )
    duplicated, sources = hierarchical_source_smooth_pool(
        torch.tensor([[2.0, 2.0, 0.0]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.tensor([[0, 0, 1]]),
        temperature=0.5,
    )
    assert torch.allclose(first, duplicated)
    assert len(sources) == 1 and sources[0].shape == (2,)


def test_hierarchical_candidate_weights_are_exact_gradients() -> None:
    logits = torch.tensor(
        [[1.2, -0.4, 0.7, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, True, False]])
    sources = torch.tensor([[0, 0, 1, 1]])
    pooled, _ = hierarchical_source_smooth_pool(
        logits,
        valid,
        sources,
        temperature=0.6,
    )
    pooled.sum().backward()
    exact = hierarchical_source_candidate_weights(
        logits.detach(),
        valid,
        sources,
        temperature=0.6,
    )
    assert torch.allclose(exact, logits.grad, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(exact.sum(dim=1), torch.ones(1, dtype=torch.float64))
    assert exact[0, 3].item() == 0.0


def test_negative_instance_loss_never_backpropagates_positive_bag() -> None:
    logits = torch.tensor([[1.0, -1.0], [2.0, 3.0]], requires_grad=True)
    loss = negative_bag_instance_loss(
        logits,
        torch.ones_like(logits, dtype=torch.bool),
        torch.tensor([0.0, 1.0]),
    )
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]).item() == 2
    assert torch.count_nonzero(logits.grad[1]).item() == 0


def test_shared_source_validity_rejects_external_only_bag() -> None:
    with np.testing.assert_raises(ValueError):
        shared_source_validity(
            torch.ones((1, 2), dtype=torch.bool),
            torch.tensor([[2, 2]]),
        )


def test_temperature_schedule_has_fixed_endpoints_and_is_monotone() -> None:
    values = [
        geometric_continuation_temperature(epoch, 16)
        for epoch in range(1, 17)
    ]
    assert values[0] == 1.0
    assert np.isclose(values[-1], 0.2)
    assert all(left > right for left, right in zip(values, values[1:]))


def test_average_percentile_rank_and_equal_fusion() -> None:
    assert np.allclose(
        average_percentile_rank(np.asarray([3.0, 1.0, 3.0, 2.0])),
        [2.5 / 3.0, 0.0, 2.5 / 3.0, 1.0 / 3.0],
    )
    fused = rank_fusion_scores(
        np.asarray([3.0, 1.0, 2.0]),
        np.asarray([1.0, 3.0, 2.0]),
    )
    assert np.allclose(fused, [0.5, 0.5, 0.5])


def test_stable_select_uses_raw_then_lower_index_for_fusion_ties() -> None:
    assert stable_select(
        np.asarray([0.5, 0.5, 0.5]),
        np.asarray([0.1, 0.9, 0.9]),
    ) == 1


def test_all_matched_arms_complete_finite_cpu_smoke() -> None:
    config = MaskBagMILConfig(
        token_dim=2,
        token_layers=1,
        hidden_dim=8,
        metadata_dim=4,
    )
    generator = np.random.default_rng(7)
    cache = []
    for index, label in enumerate((0, 1, 0, 1)):
        count = 4 + index % 2
        cache.append(
            {
                "image_id": f"image-{index}",
                "label": label,
                "descriptors": generator.normal(
                    size=(count, config.descriptor_dim)
                ).astype(np.float32),
                "flipped_descriptors": generator.normal(
                    size=(count, config.descriptor_dim)
                ).astype(np.float32),
                "source_ids": np.asarray(
                    ([0, 1, 2, 0, 1][:count]), dtype=np.int16
                ),
            }
        )
    torch.manual_seed(42)
    initial = {
        key: value.detach().clone()
        for key, value in RadDinoMaskBagMIL(config).state_dict().items()
    }
    args = SimpleNamespace(
        seed=42,
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        epochs=2,
        train_batch_size=2,
        instance_warmup_epochs=0,
        instance_loss_weight=0.25,
        consistency_loss_weight=0.10,
    )
    for name in ARM_NAMES:
        model, history = train_arm(
            name,
            cache,
            config,
            initial,
            args,
            torch.device("cpu"),
        )
        assert len(history) == 2
        assert all(
            np.isfinite(value)
            for row in history
            for value in row.values()
        )
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
