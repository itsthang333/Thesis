from __future__ import annotations

import pytest
import torch

from models.cam_conditioned_geometry import (
    cam_conditioned_descriptor_extension,
    cam_conditioned_extension_dim,
)
from models.rad_dino_mask_bag_mil import MaskBagMILConfig


def _config() -> MaskBagMILConfig:
    return MaskBagMILConfig(
        token_dim=1,
        token_layers=1,
        metadata_dim=1,
        context_radius=1,
        minimum_grid_mass=0.25,
    )


def test_extension_has_frozen_default_width() -> None:
    config = MaskBagMILConfig()
    assert cam_conditioned_extension_dim(config) == 1152
    assert config.descriptor_dim + cam_conditioned_extension_dim(config) == 2308


def test_core_low_interior_and_positive_ring_are_separate() -> None:
    config = _config()
    token_maps = torch.tensor(
        [[[[[2.0], [3.0], [4.0]], [[5.0], [7.0], [11.0]], [[13.0], [17.0], [19.0]]]]]
    )
    masks = torch.zeros((1, 1, 3, 3))
    masks[0, 0, 1, 1] = 1.0
    prompt = torch.zeros((1, 3, 3))
    prompt[0, 1, 1] = 1.0
    prompt[0, 0, 0] = 1.0

    extension, valid = cam_conditioned_descriptor_extension(
        token_maps,
        masks,
        prompt,
        torch.ones((1, 1), dtype=torch.bool),
        config,
    )

    assert torch.equal(valid, torch.ones((1, 1), dtype=torch.bool))
    assert torch.allclose(extension[0, 0], torch.tensor([7.0, 0.0, 2.0]))


def test_fractional_core_uses_geometry_v3_denominator_contract() -> None:
    config = _config()
    token_maps = torch.tensor([[[[[8.0]]]]])
    masks = torch.tensor([[[[0.5]]]])
    prompt = torch.tensor([[[0.5]]])

    extension, valid = cam_conditioned_descriptor_extension(
        token_maps,
        masks,
        prompt,
        torch.ones((1, 1), dtype=torch.bool),
        config,
    )

    assert bool(valid[0, 0])
    assert torch.allclose(extension[0, 0], torch.tensor([2.0, 2.0, 0.0]))


def test_padding_cannot_create_positive_exterior_evidence() -> None:
    config = _config()
    token_maps = torch.tensor(
        [[[[[1000.0], [1000.0], [1000.0]], [[1000.0], [3.0], [1000.0]], [[1000.0], [1000.0], [1000.0]]]]]
    )
    masks = torch.zeros((1, 1, 3, 3))
    masks[0, 0, 1, 1] = 1.0
    prompt = torch.ones((1, 3, 3))
    content = torch.zeros((1, 3, 3))
    content[0, 1, 1] = 1.0

    extension, _valid = cam_conditioned_descriptor_extension(
        token_maps,
        masks,
        prompt,
        torch.ones((1, 1), dtype=torch.bool),
        config,
        content_masks=content,
    )

    assert torch.allclose(extension[0, 0], torch.tensor([3.0, 0.0, 0.0]))


def test_joint_horizontal_flip_is_descriptor_invariant() -> None:
    config = _config()
    token_maps = torch.arange(1.0, 13.0).reshape(1, 1, 3, 4, 1)
    masks = torch.tensor(
        [[[[1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    prompt = torch.tensor(
        [[[0.9, 0.4, 0.1, 0.0], [0.8, 0.6, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    content = torch.tensor(
        [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]]
    )
    candidate_valid = torch.ones((1, 1), dtype=torch.bool)

    original, valid = cam_conditioned_descriptor_extension(
        token_maps,
        masks,
        prompt,
        candidate_valid,
        config,
        content_masks=content,
    )
    flipped, flipped_valid = cam_conditioned_descriptor_extension(
        token_maps.flip(-2),
        masks.flip(-1),
        prompt.flip(-1),
        candidate_valid,
        config,
        content_masks=content.flip(-1),
    )

    assert torch.equal(valid, flipped_valid)
    assert torch.allclose(original, flipped)


def test_invalid_candidate_is_zero_without_reordering() -> None:
    config = _config()
    token_maps = torch.ones((1, 1, 2, 2, 1))
    masks = torch.ones((1, 3, 2, 2))
    prompt = torch.full((1, 2, 2), 0.5)
    candidate_valid = torch.tensor([[True, False, True]])

    extension, valid = cam_conditioned_descriptor_extension(
        token_maps,
        masks,
        prompt,
        candidate_valid,
        config,
    )

    assert torch.equal(valid, candidate_valid)
    assert torch.count_nonzero(extension[0, 1]) == 0
    assert torch.allclose(extension[0, 0], extension[0, 2])


@pytest.mark.parametrize(
    "prompt",
    [torch.tensor([[[float("nan")]]]), torch.tensor([[[1.1]]]), torch.tensor([[[-0.1]]])],
)
def test_invalid_prompt_map_fails_closed(prompt: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        cam_conditioned_descriptor_extension(
            torch.ones((1, 1, 1, 1, 1)),
            torch.ones((1, 1, 1, 1)),
            prompt,
            torch.ones((1, 1), dtype=torch.bool),
            _config(),
        )
