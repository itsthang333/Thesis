"""S9 candidate-marginalized spatial likelihood primitives.

These functions are independent of BTXRD I/O and segmentation ground truth.
They consume frozen token features, image labels, and class-agnostic fractional
candidate/ring supports only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


class CosineTokenEvidenceHead(nn.Module):
    """One affine tumor direction over frozen, L2-normalized patch tokens."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.weight = nn.Parameter(torch.zeros(feature_dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.weight.numel():
            raise ValueError("tokens must be BPD with the configured feature dimension")
        _require_finite("tokens", tokens)
        normalized = F.normalize(tokens.float(), dim=-1, eps=1.0e-6)
        return torch.einsum("bpd,d->bp", normalized, self.weight.float()) + self.bias.float()


def candidate_spatial_log_likelihood(
    token_logits: torch.Tensor,
    candidate_weights: torch.Tensor,
    ring_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Balanced inside-positive/ring-negative likelihood for each candidate."""

    if token_logits.ndim != 2 or candidate_weights.ndim != 3:
        raise ValueError("token logits/candidates must be BP/BCP")
    if ring_weights.shape != candidate_weights.shape:
        raise ValueError("candidate and ring weights differ")
    if (
        candidate_weights.shape[0] != token_logits.shape[0]
        or candidate_weights.shape[2] != token_logits.shape[1]
    ):
        raise ValueError("candidate and token shapes differ")
    if candidate_valid.shape != candidate_weights.shape[:2]:
        raise ValueError("candidate_valid must be BC")
    _require_finite("token_logits", token_logits)
    _require_finite("candidate_weights", candidate_weights)
    _require_finite("ring_weights", ring_weights)
    if bool((candidate_weights < 0).any()) or bool((ring_weights < 0).any()):
        raise ValueError("fractional weights must be non-negative")
    inside_mass = candidate_weights.sum(dim=-1)
    ring_mass = ring_weights.sum(dim=-1)
    if bool(((inside_mass <= 0) & candidate_valid).any()):
        raise ValueError("valid candidate has zero inside mass")
    if bool(((ring_mass <= 0) & candidate_valid).any()):
        raise ValueError("valid candidate has zero ring mass")
    inside = (
        candidate_weights * F.logsigmoid(token_logits)[:, None, :]
    ).sum(dim=-1) / inside_mass.clamp_min(1.0e-12)
    ring = (
        ring_weights * F.logsigmoid(-token_logits)[:, None, :]
    ).sum(dim=-1) / ring_mass.clamp_min(1.0e-12)
    likelihood = 0.5 * (inside + ring)
    return likelihood.masked_fill(~candidate_valid, -torch.inf)


def normalized_candidate_logmeanexp(
    candidate_likelihood: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-image log-mean-exp, invariant to candidate order."""

    if (
        candidate_likelihood.ndim != 2
        or candidate_valid.shape != candidate_likelihood.shape
    ):
        raise ValueError("candidate likelihood/valid must be BC")
    if not candidate_valid.any(dim=1).all():
        raise ValueError("every image requires at least one valid candidate")
    if not torch.isfinite(candidate_likelihood[candidate_valid]).all():
        raise ValueError("valid candidate likelihood is non-finite")
    counts = candidate_valid.sum(dim=1).to(candidate_likelihood.dtype)
    return torch.logsumexp(candidate_likelihood.masked_fill(~candidate_valid, -torch.inf), dim=1) - counts.log()


def candidate_marginal_image_label_loss(
    token_logits: torch.Tensor,
    tumor: torch.Tensor,
    candidate_weights: torch.Tensor,
    ring_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
    content_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Image-label-only loss with dense normal negatives and latent tumor masks."""

    labels = tumor.float().reshape(-1)
    if token_logits.ndim != 2 or labels.shape[0] != token_logits.shape[0]:
        raise ValueError("tumor labels and token logits differ")
    if content_valid.shape != token_logits.shape:
        raise ValueError("content_valid must be BP")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("tumor labels must be binary")
    if not content_valid.any(dim=1).all():
        raise ValueError("every image needs a valid content token")
    likelihood = candidate_spatial_log_likelihood(
        token_logits, candidate_weights, ring_weights, candidate_valid
    )
    marginal = normalized_candidate_logmeanexp(likelihood, candidate_valid)
    image_losses: list[torch.Tensor] = []
    negative_losses: list[torch.Tensor] = []
    positive_losses: list[torch.Tensor] = []
    for index in range(len(labels)):
        if float(labels[index].detach()) < 0.5:
            value = F.softplus(token_logits[index][content_valid[index]]).mean()
            negative_losses.append(value)
        else:
            value = -marginal[index]
            positive_losses.append(value)
        image_losses.append(value)
    zero = token_logits.sum() * 0.0
    return {
        "total": torch.stack(image_losses).mean(),
        "normal_dense": torch.stack(negative_losses).mean() if negative_losses else zero,
        "tumor_candidate_marginal": torch.stack(positive_losses).mean() if positive_losses else zero,
        "candidate_likelihood": likelihood,
    }


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("rank values must be finite and non-empty")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return ranks / max(1, len(values) - 1)


def finite_readout(
    geometry_scores: np.ndarray,
    upstream_scores: np.ndarray,
    likelihood_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    geometry_rank = average_percentile_rank(geometry_scores)
    upstream_rank = average_percentile_rank(upstream_scores)
    likelihood_rank = average_percentile_rank(likelihood_scores)
    if not (geometry_rank.shape == upstream_rank.shape == likelihood_rank.shape):
        raise ValueError("candidate score shapes differ")
    return {
        "control": 0.5 * (geometry_rank + upstream_rank),
        "primary": (geometry_rank + upstream_rank + likelihood_rank) / 3.0,
    }
