from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from project.models.skelex_mask_bag_descriptor import (
    SKELEX_HIDDEN_SIZE,
    SKELEX_PATCHES,
    SkelexDescriptorConfig,
    SkelexProjectedMultiLayerEncoder,
    exact_fractional_mask_pool_descriptors,
)


class _Output:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _FakeMaeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_noise: torch.Tensor | None = None

    def forward(self, pixel_values, noise, output_hidden_states, return_dict):
        assert output_hidden_states and return_dict
        self.last_noise = noise.detach().cpu()
        batch = pixel_values.shape[0]
        base = torch.ones(
            batch,
            SKELEX_PATCHES + 1,
            SKELEX_HIDDEN_SIZE,
            device=pixel_values.device,
        )
        return _Output(tuple(base * float(index + 1) for index in range(25)))


def test_encoder_preserves_unmasked_patch_order() -> None:
    fake = _FakeMaeEncoder()
    projection = torch.zeros(SKELEX_HIDDEN_SIZE, 2)
    projection[0, 0] = 1.0
    projection[1, 1] = 1.0
    model = SkelexProjectedMultiLayerEncoder(fake, projection)
    result = model(torch.zeros(2, 3, 224, 224))
    assert result.shape == (2, 3, 14, 14, 2)
    assert fake.last_noise is not None
    assert torch.equal(fake.last_noise[0], torch.arange(SKELEX_PATCHES).float())


def test_fractional_pool_keeps_subpatch_candidate_and_exactly_normalizes() -> None:
    config = SkelexDescriptorConfig(token_dim=1, token_layers=1, metadata_dim=1)
    tokens = torch.arange(196, dtype=torch.float32).reshape(1, 1, 14, 14, 1)
    masks = torch.zeros(1, 1, 56, 56)
    masks[0, 0, 1, 1] = 1.0
    metadata = torch.tensor([[[0.25]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    descriptors, retained, mass = exact_fractional_mask_pool_descriptors(
        tokens, masks, metadata, valid, config
    )
    assert retained.item()
    assert 0.0 < mass.item() < 1.0
    assert descriptors.shape == (1, 1, config.descriptor_dim)
    # Exact fractional normalization must not attenuate the inside value by
    # clamping its sub-unit mass denominator to one.
    assert descriptors[0, 0, 0].item() == 0.0


def test_fractional_pool_fails_instead_of_silently_dropping_zero_support() -> None:
    config = SkelexDescriptorConfig(token_dim=1, token_layers=1, metadata_dim=1)
    tokens = torch.zeros(1, 1, 14, 14, 1)
    masks = torch.zeros(1, 1, 56, 56)
    metadata = torch.zeros(1, 1, 1)
    valid = torch.ones(1, 1, dtype=torch.bool)
    try:
        exact_fractional_mask_pool_descriptors(tokens, masks, metadata, valid, config)
    except RuntimeError as error:
        assert "silent dropping is forbidden" in str(error)
    else:
        raise AssertionError("zero-support immutable candidate must fail closed")
