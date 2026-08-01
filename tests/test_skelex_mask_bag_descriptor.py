from __future__ import annotations

import pytest
import numpy as np

torch = pytest.importorskip("torch")
from torch import nn

from project.models.skelex_mask_bag_descriptor import (
    SKELEX_HIDDEN_SIZE,
    SKELEX_PATCHES,
    SkelexDescriptorConfig,
    SkelexProjectedMultiLayerEncoder,
    exact_fractional_mask_pool_descriptors,
    mass_symmetry_tolerances,
)
from project.audit_skelex_mask_bag_selector_s5_output import _rank32
from project.models.bas_candidate_localizer import within_bag_percentile_ranks


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


def test_mass_symmetry_uses_the_proven_four_float32_ulp_budget() -> None:
    masses = np.asarray([0.25, 1.0, 17.0, 196.0], dtype=np.float32)
    four_ulp = masses.copy()
    five_ulp = masses.copy()
    for _ in range(4):
        four_ulp = np.nextafter(four_ulp, np.float32(np.inf), dtype=np.float32)
        five_ulp = np.nextafter(five_ulp, np.float32(np.inf), dtype=np.float32)
    five_ulp = np.nextafter(five_ulp, np.float32(np.inf), dtype=np.float32)
    assert np.all(
        np.abs(four_ulp.astype(np.float64) - masses.astype(np.float64))
        <= mass_symmetry_tolerances(masses, four_ulp)
    )
    assert np.all(
        np.abs(five_ulp.astype(np.float64) - masses.astype(np.float64))
        > mass_symmetry_tolerances(masses, five_ulp)
    )


def test_independent_rank_reproduction_matches_generator_float32_exactly() -> None:
    logits = torch.tensor(
        [[0.125, -0.5, 0.125, 3.0, 1.25, -2.0, 0.75]], dtype=torch.float32
    )
    valid = torch.ones_like(logits, dtype=torch.bool)
    generated = within_bag_percentile_ranks(logits, valid)[0].numpy()
    reproduced = _rank32(logits[0].numpy())
    assert generated.dtype == np.float32
    assert reproduced.dtype == np.float32
    assert np.array_equal(reproduced, generated)
