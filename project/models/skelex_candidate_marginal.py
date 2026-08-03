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


SKELEX_INPUT_SIZE = 512
SKELEX_PATCH_SIZE = 16
SKELEX_GRID_SIZE = SKELEX_INPUT_SIZE // SKELEX_PATCH_SIZE
SKELEX_PATCHES = SKELEX_GRID_SIZE * SKELEX_GRID_SIZE
SKELEX_HIDDEN_SIZE = 1024
SKELEX_HIDDEN_LAYERS = (8, 16)
SKELEX_TOKEN_DIM = len(SKELEX_HIDDEN_LAYERS) * SKELEX_HIDDEN_SIZE
SKELEX_HEAD_HIDDEN_DIM = 256
SKELEX_RING_RADIUS = 2


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


class NonlinearTokenEvidenceHead(nn.Module):
    """Bounded-capacity nonlinear tumor head over normalized layer tokens."""

    def __init__(
        self,
        feature_dim: int = SKELEX_TOKEN_DIM,
        hidden_dim: int = SKELEX_HEAD_HIDDEN_DIM,
        layer_dim: int = SKELEX_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or layer_dim <= 0:
            raise ValueError("head dimensions must be positive")
        if feature_dim % layer_dim:
            raise ValueError("feature_dim must contain whole SKELEX layers")
        self.feature_dim = feature_dim
        self.layer_dim = layer_dim
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.feature_dim:
            raise ValueError("tokens must be BPD with the configured feature dimension")
        _require_finite("tokens", tokens)
        grouped = tokens.float().reshape(*tokens.shape[:-1], -1, self.layer_dim)
        normalized = F.normalize(grouped, dim=-1, eps=1.0e-6).flatten(start_dim=-2)
        hidden = F.gelu(self.projection(normalized))
        return self.output(hidden)[..., 0]


class SkelexMultiLayerTokenEncoder(nn.Module):
    """Expose fixed, unmasked SKELEX intermediate patch-token grids."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[-2:] != (
            SKELEX_INPUT_SIZE,
            SKELEX_INPUT_SIZE,
        ):
            raise ValueError("S9 pixels must be Bx3x512x512")
        batch = pixel_values.shape[0]
        noise = torch.arange(
            SKELEX_PATCHES,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )[None].expand(batch, -1)
        output = self.encoder(
            pixel_values=pixel_values,
            noise=noise,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != 25:
            raise RuntimeError("SKELEX must expose embedding plus 24 hidden states")
        expected = (batch, SKELEX_PATCHES + 1, SKELEX_HIDDEN_SIZE)
        selected: list[torch.Tensor] = []
        for layer_index in SKELEX_HIDDEN_LAYERS:
            hidden = hidden_states[layer_index]
            if hidden.shape != expected:
                raise RuntimeError(
                    f"Unexpected SKELEX layer-{layer_index} shape {tuple(hidden.shape)}"
                )
            selected.append(hidden[:, 1:].float())
        return torch.cat(selected, dim=-1)


def fractional_candidate_ring_supports(
    square_candidate_masks: torch.Tensor,
    square_content_mask: torch.Tensor,
    *,
    grid_size: int = SKELEX_GRID_SIZE,
    ring_radius: int = SKELEX_RING_RADIUS,
    support_epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Area-project square supports and construct a local fractional ring."""

    if square_candidate_masks.ndim != 3 or square_content_mask.ndim != 2:
        raise ValueError("candidate/content supports must be CHW/HW")
    if square_candidate_masks.shape[-2:] != square_content_mask.shape:
        raise ValueError("candidate/content square shapes differ")
    if grid_size <= 0 or ring_radius <= 0 or support_epsilon <= 0:
        raise ValueError("support projection controls are invalid")
    _require_finite("square_candidate_masks", square_candidate_masks)
    _require_finite("square_content_mask", square_content_mask)
    if bool((square_candidate_masks < 0).any()) or bool((square_content_mask < 0).any()):
        raise ValueError("square supports must be non-negative")
    count = square_candidate_masks.shape[0]
    candidates = F.interpolate(
        square_candidate_masks[:, None].float(),
        size=(grid_size, grid_size),
        mode="area",
    )[:, 0].clamp_(0.0, 1.0)
    content = F.interpolate(
        square_content_mask[None, None].float(),
        size=(grid_size, grid_size),
        mode="area",
    )[0, 0].clamp_(0.0, 1.0)
    candidates = candidates * content[None]
    kernel = 2 * ring_radius + 1
    dilated = F.max_pool2d(
        candidates[:, None],
        kernel_size=kernel,
        stride=1,
        padding=ring_radius,
    )[:, 0]
    rings = (dilated - candidates).clamp_min(0.0) * content[None]
    inside_mass = candidates.sum(dim=(-2, -1))
    ring_mass = rings.sum(dim=(-2, -1))
    if count == 0 or bool((inside_mass <= support_epsilon).any()):
        raise ValueError("S9 candidate has zero projected inside mass")
    if bool((ring_mass <= support_epsilon).any()):
        raise ValueError("S9 candidate has zero projected ring mass")
    content_valid = content.reshape(-1) > support_epsilon
    if not bool(content_valid.any()):
        raise ValueError("S9 image has no projected content token")
    return candidates.reshape(count, -1), rings.reshape(count, -1), content_valid


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
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
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
