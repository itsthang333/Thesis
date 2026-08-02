from __future__ import annotations

import pytest
import torch

from project.models.skelex_reconstruction_selector import (
    SkelexReconstructionConfig,
    make_skelex_mask_bank,
    masked_patch_squared_error,
    reconstruction_lcb,
    select_with_spatial_null,
)
from project.models.mae_reconstruction import patchify


def _config(**kwargs: object) -> SkelexReconstructionConfig:
    return SkelexReconstructionConfig(
        input_size=8,
        patch_size=2,
        num_masks=4,
        null_permutations=19,
        **kwargs,
    )


def test_mask_bank_is_seeded_and_has_frozen_mask_ratio() -> None:
    config = _config()
    first = make_skelex_mask_bank(config)
    second = make_skelex_mask_bank(config)
    assert torch.equal(first, second)
    assert first.shape == (4, 16)
    assert torch.all(first.sum(dim=1) == 12)


def test_masked_patch_error_has_no_visible_patch_signal() -> None:
    target = torch.zeros((1, 1, 4, 4))
    prediction = patchify(target, 2).clone()
    prediction[:, 0] = 4.0
    prediction[:, 3] = 2.0
    errors, observed = masked_patch_squared_error(
        prediction,
        target,
        torch.tensor([[1, 0, 0, 1]], dtype=torch.bool),
        patch_size=2,
    )
    assert observed.tolist() == [[True, False, False, True]]
    assert errors.tolist() == [[16.0, 0.0, 0.0, 4.0]]


def test_reconstruction_lcb_prefers_anomalous_candidate_over_normal_ring() -> None:
    config = _config()
    errors = torch.zeros((4, 4, 4), dtype=torch.float32)
    errors[:, 1, 1] = 10.0
    observed = torch.ones_like(errors, dtype=torch.bool)
    candidates = torch.zeros((2, 4, 4), dtype=torch.float32)
    candidates[0, 1, 1] = 1.0
    candidates[1, 0, 0] = 1.0
    result = reconstruction_lcb(errors, observed, candidates, torch.ones((4, 4)), config)
    assert result["candidate_valid"].tolist() == [True, True]
    assert float(result["lcb"][0]) > float(result["lcb"][1])


def test_spatial_null_is_reproducible_and_family_gate_can_force_baseline() -> None:
    config = _config()
    errors = torch.zeros((4, 4, 4), dtype=torch.float32)
    errors[:, 1, 1] = 10.0
    flip_errors = torch.zeros_like(errors)
    flip_errors[:, 0, 0] = 10.0
    observed = torch.ones_like(errors, dtype=torch.bool)
    candidates = torch.zeros((2, 4, 4), dtype=torch.float32)
    candidates[0, 1, 1] = 1.0
    candidates[1, 0, 0] = 1.0
    # Make the reconstructed winner's family intentionally disagree with the
    # branch winners.  The selector must then return the frozen baseline even
    # when a reconstruction score is positive.
    first = select_with_spatial_null(
        base_scores=torch.tensor([0.0, 0.0]),
        accepted_index=1,
        families=("anomaly", "baseline"),
        original_errors=errors,
        original_observed=observed,
        aligned_flip_errors=flip_errors,
        aligned_flip_observed=observed,
        candidate_masks=candidates,
        content_mask=torch.ones((4, 4)),
        config=config,
    )
    second = select_with_spatial_null(
        base_scores=torch.tensor([0.0, 0.0]),
        accepted_index=1,
        families=("anomaly", "baseline"),
        original_errors=errors,
        original_observed=observed,
        aligned_flip_errors=flip_errors,
        aligned_flip_observed=observed,
        candidate_masks=candidates,
        content_mask=torch.ones((4, 4)),
        config=config,
    )
    assert first["selected_index"] == 1
    assert first["family_consistent"] is False
    assert first["permutation_p_value"] == second["permutation_p_value"]
    assert torch.equal(first["null_max_improvements"], second["null_max_improvements"])


def test_normalized_pixel_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="normalized-pixel-loss disabled"):
        masked_patch_squared_error(
            torch.zeros((1, 4, 4)),
            torch.zeros((1, 1, 4, 4)),
            torch.ones((1, 4), dtype=torch.bool),
            patch_size=2,
            norm_pix_loss=True,
        )
