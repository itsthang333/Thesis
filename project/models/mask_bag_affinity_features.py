from __future__ import annotations

"""Compact DINO-affinity summaries for immutable proposal descriptors."""

import torch
import torch.nn.functional as F


def _weighted_affinity_statistics(
    normalized_tokens: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mass = weights.sum(dim=-1)
    squared_mass = weights.square().sum(dim=-1)
    sums = torch.einsum("bnp,blpd->bnld", weights, normalized_tokens)
    vector_norm_squared = sums.square().sum(dim=-1)

    inclusive = (
        vector_norm_squared / mass.square()[:, :, None].clamp_min(epsilon)
    ).clamp(0.0, 1.0)
    off_diagonal_denominator = mass.square() - squared_mass
    off_diagonal = torch.where(
        off_diagonal_denominator[:, :, None] > epsilon,
        (vector_norm_squared - squared_mass[:, :, None])
        / off_diagonal_denominator[:, :, None].clamp_min(epsilon),
        torch.zeros_like(vector_norm_squared),
    ).clamp(-1.0, 1.0)
    effective_count = mass.square() / squared_mass.clamp_min(epsilon)
    return inclusive, off_diagonal, effective_count, sums


def affinity_summary_features(
    token_maps: torch.Tensor,
    proposal_weights: torch.Tensor,
    context_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Summarize within-proposal, context and cross-boundary token affinity.

    This computes exact weighted mean-pair cosine statistics without materializing
    a quadratic token-affinity matrix. The output has eight features per layer:
    inclusive/off-diagonal proposal cohesion, inclusive/off-diagonal context
    cohesion, proposal-context cosine, cohesion contrast and two normalized
    effective-token counts.
    """

    if token_maps.ndim != 5:
        raise ValueError("token_maps must have shape [B,L,H,W,D]")
    if proposal_weights.ndim != 4 or context_weights.shape != proposal_weights.shape:
        raise ValueError("proposal/context weights must share shape [B,N,H,W]")
    batch, layers, height, width, channels = token_maps.shape
    if proposal_weights.shape[0] != batch or proposal_weights.shape[2:] != (
        height,
        width,
    ):
        raise ValueError("proposal weights must align with token maps")
    if candidate_valid.shape != proposal_weights.shape[:2]:
        raise ValueError("candidate_valid must align with proposal weights")
    if channels < 2 or epsilon <= 0:
        raise ValueError("token dimension and epsilon must be positive")
    if (
        not torch.isfinite(token_maps).all()
        or not torch.isfinite(proposal_weights).all()
        or not torch.isfinite(context_weights).all()
    ):
        raise ValueError("affinity inputs must be finite")
    if (proposal_weights < 0).any() or (context_weights < 0).any():
        raise ValueError("proposal/context weights must be nonnegative")

    valid = candidate_valid.bool()
    proposal_mass = proposal_weights.sum(dim=(-2, -1))
    if (proposal_mass[valid] <= epsilon).any():
        raise ValueError("every valid candidate must have positive proposal mass")
    if not valid.any(dim=1).all():
        raise ValueError("every bag must contain at least one valid candidate")

    tokens = F.normalize(
        token_maps.reshape(batch, layers, height * width, channels).float(),
        dim=-1,
        eps=epsilon,
    )
    proposal = proposal_weights.reshape(batch, proposal_weights.shape[1], -1).float()
    context = context_weights.reshape(batch, context_weights.shape[1], -1).float()
    (
        proposal_inclusive,
        proposal_off_diagonal,
        proposal_effective,
        proposal_sums,
    ) = _weighted_affinity_statistics(tokens, proposal, epsilon=epsilon)
    (
        context_inclusive,
        context_off_diagonal,
        context_effective,
        context_sums,
    ) = _weighted_affinity_statistics(tokens, context, epsilon=epsilon)

    context_mass = context.sum(dim=-1)
    has_context = context_mass > epsilon
    cross_affinity = F.cosine_similarity(
        proposal_sums,
        context_sums,
        dim=-1,
        eps=epsilon,
    )
    cross_affinity = torch.where(
        has_context[:, :, None],
        cross_affinity,
        torch.zeros_like(cross_affinity),
    ).clamp(-1.0, 1.0)
    grid_tokens = proposal.new_tensor(float(height * width))
    proposal_effective_log = torch.log1p(proposal_effective) / torch.log1p(
        grid_tokens
    )
    context_effective_log = torch.log1p(context_effective) / torch.log1p(
        grid_tokens
    )
    context_effective_log = torch.where(
        has_context,
        context_effective_log,
        torch.zeros_like(context_effective_log),
    )
    cohesion_contrast = proposal_inclusive - cross_affinity

    features = torch.stack(
        (
            proposal_inclusive,
            proposal_off_diagonal,
            context_inclusive,
            context_off_diagonal,
            cross_affinity,
            cohesion_contrast,
            proposal_effective_log[:, :, None].expand(-1, -1, layers),
            context_effective_log[:, :, None].expand(-1, -1, layers),
        ),
        dim=-1,
    ).flatten(start_dim=2)
    features = features * valid[:, :, None].to(features.dtype)
    expected = (batch, proposal_weights.shape[1], 8 * layers)
    if features.shape != expected or not torch.isfinite(features).all():
        raise RuntimeError("affinity summary feature layout is invalid")
    return features


__all__ = ["affinity_summary_features"]
