"""Frozen SKELEX decoder-reconstruction evidence for same-gallery reranking.

This module deliberately contains no dataset or validation annotation access.  It
turns masked-patch reconstruction errors into a conservative candidate-local
score and supplies a spatial randomization gate.  The latter keeps the error
histogram, candidate geometry and baseline scores fixed while removing their
spatial alignment, so a selector cannot win merely from area or anatomy bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.mae_reconstruction import make_noise_bank, patchify


@dataclass(frozen=True)
class SkelexReconstructionConfig:
    input_size: int = 224
    patch_size: int = 16
    num_masks: int = 10
    mask_ratio: float = 0.75
    mask_seed: int = 42
    null_permutations: int = 255
    null_seed: int = 20261203
    context_radius: int = 2
    lcb_z: float = 1.96
    geometry_weight: float = 0.75
    reconstruction_weight: float = 0.25
    support_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.patch_size <= 0:
            raise ValueError("SKELEX input and patch sizes must be positive")
        if self.input_size % self.patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        if self.num_masks < 2 or not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("SKELEX mask recipe is invalid")
        if self.null_permutations <= 0:
            raise ValueError("the null must contain at least one permutation")
        if self.context_radius < 1 or self.lcb_z <= 0.0:
            raise ValueError("SKELEX spatial controls are invalid")
        if self.geometry_weight < 0.0 or self.reconstruction_weight < 0.0:
            raise ValueError("fusion weights cannot be negative")
        if abs(self.geometry_weight + self.reconstruction_weight - 1.0) > 1.0e-12:
            raise ValueError("fusion weights must sum to one")

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_size * self.grid_size


def make_skelex_mask_bank(config: SkelexReconstructionConfig) -> torch.Tensor:
    """Return the exact [M,P] binary mask bank used by ViT-MAE."""

    noise = make_noise_bank(
        num_masks=config.num_masks,
        num_patches=config.patch_count,
        seed=config.mask_seed,
    )
    keep = int(config.patch_count * (1.0 - config.mask_ratio))
    if keep <= 0 or keep >= config.patch_count:
        raise ValueError("mask_ratio leaves no visible or no masked patches")
    order = torch.argsort(noise, dim=1, stable=True)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    mask.scatter_(1, order[:, keep:], True)
    if int(mask.sum(dim=1).min()) != config.patch_count - keep:
        raise RuntimeError("SKELEX mask bank has an inconsistent masked count")
    return mask


def masked_patch_squared_error(
    prediction_patches: torch.Tensor,
    pixel_values: torch.Tensor,
    patch_mask: torch.Tensor,
    *,
    patch_size: int,
    norm_pix_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-patch reconstruction MSE, retaining masked patches only.

    ``ViTMAEForPreTraining`` exposes logits in original patch order.  S8 freezes
    normalized-pixel-loss off, hence the target is exactly ``patchify`` output;
    silently accepting the alternate target would invalidate the decoder score.
    """

    if norm_pix_loss:
        raise ValueError("S8 requires normalized-pixel-loss disabled")
    if prediction_patches.ndim != 3 or pixel_values.ndim != 4:
        raise ValueError("unexpected MAE prediction/input dimensions")
    target = patchify(pixel_values, patch_size)
    if prediction_patches.shape != target.shape:
        raise ValueError("prediction patches do not align with pixel target")
    if patch_mask.shape != prediction_patches.shape[:2]:
        raise ValueError("MAE patch mask does not align with predictions")
    errors = (prediction_patches.float() - target.float()).square().mean(dim=-1)
    observed = patch_mask.to(dtype=torch.bool)
    if not torch.isfinite(errors).all() or not torch.isfinite(target).all():
        raise ValueError("MAE reconstruction contains non-finite values")
    return errors, observed


def _validate_maps(
    error_maps: torch.Tensor,
    observed_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
    content_mask: torch.Tensor,
) -> tuple[int, int, int, int]:
    if error_maps.ndim < 3 or observed_maps.ndim != 3:
        raise ValueError("error maps must be [...,M,H,W] and observations [M,H,W]")
    if error_maps.shape[-3:] != observed_maps.shape:
        raise ValueError("error and observation maps are misaligned")
    if candidate_masks.ndim != 3 or candidate_masks.shape[-2:] != observed_maps.shape[-2:]:
        raise ValueError("candidate masks must be [N,H,W] on the decoder grid")
    if content_mask.shape != observed_maps.shape[-2:]:
        raise ValueError("content mask must align with decoder grid")
    for name, values in (
        ("error_maps", error_maps),
        ("candidate_masks", candidate_masks),
        ("content_mask", content_mask),
    ):
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
    if (candidate_masks < 0).any() or (candidate_masks > 1).any():
        raise ValueError("candidate masks must be in [0,1]")
    if (content_mask < 0).any() or (content_mask > 1).any():
        raise ValueError("content mask must be in [0,1]")
    return (
        int(error_maps.numel() // np.prod(error_maps.shape[-3:])),
        int(error_maps.shape[-3]),
        int(error_maps.shape[-2]),
        int(error_maps.shape[-1]),
    )


def _candidate_weights(
    candidate_masks: torch.Tensor,
    content_mask: torch.Tensor,
    observed_maps: torch.Tensor,
    *,
    context_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build content-aware inside/ring weights and fixed valid patch cells."""

    candidates, height, width = candidate_masks.shape
    content = content_mask.float().clamp(0.0, 1.0)
    candidate = candidate_masks.float().clamp(0.0, 1.0)
    inside = candidate * content[None]
    dilated = F.max_pool2d(
        candidate[:, None],
        kernel_size=2 * context_radius + 1,
        stride=1,
        padding=context_radius,
    )[:, 0]
    ring = (dilated - candidate).clamp_min(0.0) * content[None]
    observed = observed_maps.bool().float()
    inside_w = observed[:, None] * inside[None]
    ring_w = observed[:, None] * ring[None]
    inside_mass = inside_w.sum(dim=(-2, -1))
    ring_mass = ring_w.sum(dim=(-2, -1))
    return inside_w, ring_w, inside_mass, ring_mass


def _lcb_from_weights(
    error_maps: torch.Tensor,
    inside_w: torch.Tensor,
    ring_w: torch.Tensor,
    *,
    lcb_z: float,
    support_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contrasts, LCB and valid candidates for one or many error banks."""

    if error_maps.ndim == 3:
        error_maps = error_maps.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    lead = error_maps.shape[:-3]
    maps = error_maps.reshape(*lead, error_maps.shape[-3], -1).float()
    inside = inside_w.reshape(inside_w.shape[0], inside_w.shape[1], -1).float()
    ring = ring_w.reshape(ring_w.shape[0], ring_w.shape[1], -1).float()
    numerator_inside = torch.einsum("...mp,mnp->...mn", maps, inside)
    numerator_ring = torch.einsum("...mp,mnp->...mn", maps, ring)
    inside_mass = inside.sum(dim=-1)
    ring_mass = ring.sum(dim=-1)
    finite = (inside_mass > support_epsilon) & (ring_mass > support_epsilon)
    contrast = numerator_inside / inside_mass.clamp_min(support_epsilon)
    contrast = contrast - numerator_ring / ring_mass.clamp_min(support_epsilon)
    contrast = torch.where(finite[None], contrast, torch.full_like(contrast, float("nan")))
    count = torch.isfinite(contrast).sum(dim=-2)
    safe = torch.nan_to_num(contrast, nan=0.0)
    total = safe.sum(dim=-2)
    mean = total / count.clamp_min(1).to(safe.dtype)
    centered = torch.where(torch.isfinite(contrast), (safe - mean[..., None, :]).square(), torch.zeros_like(safe))
    variance = centered.sum(dim=-2) / (count - 1).clamp_min(1).to(safe.dtype)
    se = torch.sqrt(variance / count.clamp_min(1).to(safe.dtype))
    lcb = mean - float(lcb_z) * se
    candidate_valid = count >= 2
    lcb = torch.where(candidate_valid, lcb, torch.full_like(lcb, float("-inf")))
    if squeeze:
        return contrast[0], lcb[0], candidate_valid[0], count[0]
    return contrast, lcb, candidate_valid, count


def reconstruction_lcb(
    error_maps: torch.Tensor,
    observed_maps: torch.Tensor,
    candidate_masks: torch.Tensor,
    content_mask: torch.Tensor,
    config: SkelexReconstructionConfig,
) -> dict[str, torch.Tensor]:
    """Score each immutable candidate using signed inside-minus-ring LCB."""

    _validate_maps(error_maps, observed_maps, candidate_masks, content_mask)
    inside_w, ring_w, inside_mass, ring_mass = _candidate_weights(
        candidate_masks,
        content_mask,
        observed_maps,
        context_radius=config.context_radius,
    )
    contrast, lcb, valid, count = _lcb_from_weights(
        error_maps,
        inside_w,
        ring_w,
        lcb_z=config.lcb_z,
        support_epsilon=config.support_epsilon,
    )
    return {
        "contrast": contrast,
        "lcb": lcb,
        "candidate_valid": valid,
        "observation_count": count,
        "inside_mass": inside_mass,
        "ring_mass": ring_mass,
    }


def _rank_descending(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("candidate values/validity must be one-dimensional")
    if not bool(valid.any()):
        return torch.zeros_like(values)
    selected = torch.nonzero(valid, as_tuple=False).flatten()
    source = values[selected]
    if len(selected) == 1:
        result = torch.zeros_like(values)
        result[selected[0]] = 1.0
        return result
    less = (source[:, None] > source[None, :]).sum(dim=1).to(source.dtype)
    equal = (source[:, None] == source[None, :]).sum(dim=1).to(source.dtype)
    ranks = (less + 0.5 * (equal - 1.0)) / float(len(selected) - 1)
    result = torch.zeros_like(values)
    result[selected] = ranks
    return result


def _null_permutation_bank(
    observed_maps: torch.Tensor,
    content_mask: torch.Tensor,
    *,
    permutations: int,
    seed: int,
) -> torch.Tensor:
    """Build deterministic source indices for uniform within-valid-cell shuffles."""

    maps, height, width = observed_maps.shape
    cells = height * width
    valid = (observed_maps.bool() & (content_mask[None] > 0.0)).reshape(maps, cells)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bank = torch.full((maps, permutations, cells), -1, dtype=torch.long)
    for map_index in range(maps):
        indices = torch.nonzero(valid[map_index], as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        for permutation in range(permutations):
            order = torch.randperm(cells, generator=generator)
            source = order[valid[map_index][order]]
            bank[map_index, permutation, : indices.numel()] = source
    return bank


def _permute_errors(
    error_maps: torch.Tensor,
    observed_maps: torch.Tensor,
    content_mask: torch.Tensor,
    bank: torch.Tensor,
) -> torch.Tensor:
    maps, height, width = observed_maps.shape
    permutations = bank.shape[1]
    cells = height * width
    source_errors = error_maps.reshape(maps, cells)
    valid = (observed_maps.bool() & (content_mask[None] > 0.0)).reshape(maps, cells)
    result = torch.zeros((permutations, maps, cells), dtype=error_maps.dtype, device=error_maps.device)
    for map_index in range(maps):
        targets = torch.nonzero(valid[map_index], as_tuple=False).flatten()
        if targets.numel() < 2:
            continue
        source = bank[map_index, :, : targets.numel()].to(error_maps.device)
        result[:, map_index, targets] = source_errors[map_index][source]
    return result.reshape(permutations, maps, height, width)


def select_with_spatial_null(
    *,
    base_scores: torch.Tensor,
    accepted_index: int,
    families: Sequence[str],
    original_errors: torch.Tensor,
    original_observed: torch.Tensor,
    aligned_flip_errors: torch.Tensor,
    aligned_flip_observed: torch.Tensor,
    candidate_masks: torch.Tensor,
    content_mask: torch.Tensor,
    config: SkelexReconstructionConfig,
) -> dict[str, Any]:
    """Apply branch-consistency and max-statistic randomization before rerank."""

    if base_scores.ndim != 1 or len(families) != base_scores.numel():
        raise ValueError("base scores and candidate families are misaligned")
    if not 0 <= accepted_index < base_scores.numel():
        raise ValueError("accepted candidate index is outside the immutable bag")
    original = reconstruction_lcb(
        original_errors, original_observed, candidate_masks, content_mask, config
    )
    flip = reconstruction_lcb(
        aligned_flip_errors,
        aligned_flip_observed,
        candidate_masks,
        content_mask,
        config,
    )
    combined_errors = torch.cat((original_errors, aligned_flip_errors), dim=0)
    combined_observed = torch.cat((original_observed, aligned_flip_observed), dim=0)
    combined = reconstruction_lcb(
        combined_errors, combined_observed, candidate_masks, content_mask, config
    )
    base_rank = _rank_descending(base_scores.float(), torch.ones_like(base_scores, dtype=torch.bool))
    combined_rank = _rank_descending(combined["lcb"], combined["candidate_valid"])
    original_rank = _rank_descending(original["lcb"], original["candidate_valid"])
    flip_rank = _rank_descending(flip["lcb"], flip["candidate_valid"])
    combined_fused = config.geometry_weight * base_rank + config.reconstruction_weight * combined_rank
    original_fused = config.geometry_weight * base_rank + config.reconstruction_weight * original_rank
    flip_fused = config.geometry_weight * base_rank + config.reconstruction_weight * flip_rank
    combined_fused = torch.where(combined["candidate_valid"], combined_fused, torch.full_like(combined_fused, float("-inf")))
    original_fused = torch.where(original["candidate_valid"], original_fused, torch.full_like(original_fused, float("-inf")))
    flip_fused = torch.where(flip["candidate_valid"], flip_fused, torch.full_like(flip_fused, float("-inf")))
    original_winner = int(torch.argmax(original_fused).item()) if bool(original["candidate_valid"].any()) else -1
    flip_winner = int(torch.argmax(flip_fused).item()) if bool(flip["candidate_valid"].any()) else -1
    combined_winner = int(torch.argmax(combined_fused).item()) if bool(combined["candidate_valid"].any()) else -1
    observed_improvement = float("-inf")
    if combined_winner >= 0:
        accepted_value = float(combined_fused[accepted_index].item()) if bool(combined["candidate_valid"][accepted_index]) else float("-inf")
        observed_improvement = float(combined_fused[combined_winner].item() - accepted_value)
    family_consistent = bool(
        combined_winner >= 0
        and original_winner >= 0
        and flip_winner >= 0
        and families[combined_winner] == families[original_winner] == families[flip_winner]
    )
    bank = _null_permutation_bank(
        combined_observed,
        content_mask,
        permutations=config.null_permutations,
        seed=config.null_seed,
    )
    permuted = _permute_errors(combined_errors, combined_observed, content_mask, bank)
    inside_w, ring_w, _, _ = _candidate_weights(
        candidate_masks, content_mask, combined_observed, context_radius=config.context_radius
    )
    _, null_lcb, null_valid, _ = _lcb_from_weights(
        permuted,
        inside_w,
        ring_w,
        lcb_z=config.lcb_z,
        support_epsilon=config.support_epsilon,
    )
    null_rank = torch.stack(
        [_rank_descending(row, valid) for row, valid in zip(null_lcb, null_valid)],
        dim=0,
    )
    null_fused = config.geometry_weight * base_rank[None] + config.reconstruction_weight * null_rank
    null_fused = torch.where(null_valid, null_fused, torch.full_like(null_fused, float("-inf")))
    null_max = torch.max(null_fused, dim=1).values
    accepted_null = null_fused[:, accepted_index]
    null_improvement = null_max - accepted_null
    exceedances = int((null_improvement >= observed_improvement - 1.0e-12).sum().item()) if np.isfinite(observed_improvement) else config.null_permutations
    p_value = (1.0 + exceedances) / float(config.null_permutations + 1)
    switch = bool(
        family_consistent
        and combined_winner != accepted_index
        and combined_winner >= 0
        and observed_improvement > 0.0
        and p_value <= 0.05
    )
    return {
        "selected_index": combined_winner if switch else int(accepted_index),
        "switched": switch,
        "original_winner": original_winner,
        "aligned_flip_winner": flip_winner,
        "combined_winner": combined_winner,
        "family_consistent": family_consistent,
        "observed_improvement": observed_improvement,
        "permutation_exceedances": exceedances,
        "permutation_p_value": p_value,
        "base_rank": base_rank,
        "combined_fused": combined_fused,
        "original_lcb": original["lcb"],
        "aligned_flip_lcb": flip["lcb"],
        "combined_lcb": combined["lcb"],
        "combined_candidate_valid": combined["candidate_valid"],
        "null_max_improvements": null_improvement,
    }


__all__ = [
    "SkelexReconstructionConfig",
    "make_skelex_mask_bank",
    "masked_patch_squared_error",
    "reconstruction_lcb",
    "select_with_spatial_null",
]
