from __future__ import annotations

"""Candidate-level cross-view co-witness primitives.

The module is deliberately dataset and annotation agnostic.  It consumes
candidate appearance descriptors, immutable baseline scores and validity
masks.  No function accepts a spatial target, proposal source, absolute
coordinate or candidate-area feature.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CrossViewCoWitnessConfig:
    appearance_dim: int = 1152
    hidden_dim: int = 256
    embedding_dim: int = 64
    residual_scale: float = 0.50
    bag_temperature: float = 0.20
    pair_temperature: float = 0.20
    cosine_weight: float = 0.50
    pair_margin: float = 0.20

    def __post_init__(self) -> None:
        for name in ("appearance_dim", "hidden_dim", "embedding_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("residual_scale", "bag_temperature", "pair_temperature"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cosine_weight < 0 or self.pair_margin < 0:
            raise ValueError("cosine_weight/pair_margin must be nonnegative")


def _validate_candidate_inputs(
    appearance: torch.Tensor,
    baseline_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    appearance_dim: int,
) -> None:
    if appearance.ndim != 3 or appearance.shape[-1] != int(appearance_dim):
        raise ValueError("appearance must have shape [B,N,appearance_dim]")
    if baseline_scores.shape != appearance.shape[:2]:
        raise ValueError("baseline_scores must align with candidate bags")
    if candidate_valid.shape != appearance.shape[:2]:
        raise ValueError("candidate_valid must align with candidate bags")
    if not candidate_valid.bool().any(dim=1).all():
        raise ValueError("every candidate bag must contain valid candidates")
    if not torch.isfinite(appearance).all() or not torch.isfinite(baseline_scores).all():
        raise ValueError("candidate inputs must be finite")


def normalized_logmeanexp(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Normalized LogSumExp, invariant to padded invalid entries."""

    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values/valid must share shape [B,N]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every row must contain valid values")
    scaled = (values / float(temperature)).masked_fill(~valid, -torch.inf)
    counts = valid.sum(dim=1).to(values.dtype)
    return float(temperature) * (torch.logsumexp(scaled, dim=1) - counts.log())


class CrossViewCoWitnessHead(nn.Module):
    """Learn an appearance embedding and a bounded residual on frozen scores."""

    def __init__(self, config: CrossViewCoWitnessConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.appearance_dim),
            nn.Linear(config.appearance_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.embedding_head = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.residual_head = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        appearance: torch.Tensor,
        baseline_scores: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_candidate_inputs(
            appearance,
            baseline_scores,
            candidate_valid,
            self.config.appearance_dim,
        )
        valid = candidate_valid.bool()
        hidden = self.trunk(appearance)
        embeddings = F.normalize(self.embedding_head(hidden), dim=-1, eps=1.0e-8)
        raw_residual = self.residual_head(hidden).squeeze(-1)
        residual = self.config.residual_scale * torch.tanh(raw_residual)
        residual = residual.masked_fill(~valid, 0.0)
        combined = (baseline_scores + residual).masked_fill(~valid, 0.0)
        embeddings = embeddings * valid[:, :, None].to(embeddings.dtype)
        return combined, residual, embeddings


def co_witness_score(
    residual_a: torch.Tensor,
    embedding_a: torch.Tensor,
    valid_a: torch.Tensor,
    residual_b: torch.Tensor,
    embedding_b: torch.Tensor,
    valid_b: torch.Tensor,
    *,
    temperature: float,
    cosine_weight: float,
) -> torch.Tensor:
    """Smoothly pool candidate pairs without selecting a detached winner."""

    if residual_a.ndim != 2 or residual_b.ndim != 2:
        raise ValueError("residual bags must have shape [B,N]")
    if embedding_a.ndim != 3 or embedding_b.ndim != 3:
        raise ValueError("embedding bags must have shape [B,N,D]")
    if embedding_a.shape[:2] != residual_a.shape or embedding_b.shape[:2] != residual_b.shape:
        raise ValueError("residual/embedding bags are misaligned")
    if embedding_a.shape[0] != embedding_b.shape[0] or embedding_a.shape[2] != embedding_b.shape[2]:
        raise ValueError("paired embedding bags are incompatible")
    if valid_a.shape != residual_a.shape or valid_b.shape != residual_b.shape:
        raise ValueError("validity must align with residual bags")
    if temperature <= 0 or cosine_weight < 0:
        raise ValueError("pair temperature/weight are invalid")
    valid_a = valid_a.bool()
    valid_b = valid_b.bool()
    if not valid_a.any(dim=1).all() or not valid_b.any(dim=1).all():
        raise ValueError("every paired bag must contain valid candidates")
    cosine = torch.einsum("bnd,bmd->bnm", embedding_a, embedding_b)
    pair_values = (
        residual_a[:, :, None]
        + residual_b[:, None, :]
        + float(cosine_weight) * cosine
    )
    pair_valid = valid_a[:, :, None] & valid_b[:, None, :]
    flattened = pair_values.flatten(start_dim=1)
    flattened_valid = pair_valid.flatten(start_dim=1)
    return normalized_logmeanexp(
        flattened,
        flattened_valid,
        temperature=temperature,
    )


def co_witness_margin_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if positive_scores.shape != negative_scores.shape or positive_scores.ndim != 1:
        raise ValueError("positive/negative scores must share vector shape")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    return F.softplus(negative_scores - positive_scores + float(margin)).mean()


def image_bag_loss(
    combined_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    image_labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    logits = normalized_logmeanexp(
        combined_scores,
        candidate_valid,
        temperature=temperature,
    )
    labels = image_labels.to(dtype=logits.dtype).reshape_as(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)


def dense_normal_candidate_loss(
    combined_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    image_labels: torch.Tensor,
) -> torch.Tensor:
    if combined_scores.ndim != 2 or candidate_valid.shape != combined_scores.shape:
        raise ValueError("candidate scores/validity must share shape [B,N]")
    labels = image_labels.reshape(-1).bool()
    if labels.numel() != combined_scores.shape[0]:
        raise ValueError("image labels do not align with candidate bags")
    normal_valid = candidate_valid.bool() & (~labels)[:, None]
    values = combined_scores[normal_valid]
    if not values.numel():
        return combined_scores.sum() * 0.0
    return F.binary_cross_entropy_with_logits(values, torch.zeros_like(values))


__all__ = [
    "CrossViewCoWitnessConfig",
    "CrossViewCoWitnessHead",
    "co_witness_margin_loss",
    "co_witness_score",
    "dense_normal_candidate_loss",
    "image_bag_loss",
    "normalized_logmeanexp",
]
