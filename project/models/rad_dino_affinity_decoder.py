from __future__ import annotations

"""Lightweight spatial decoder and local affinity refinement for RAD-DINO.

The encoder stays outside this module and remains frozen.  The decoder is
trained from image labels and frozen nominal-memory pseudo evidence.  Its
learned local feature relationships filter frozen-token cosine relationships
before seed-preserving propagation, following the transferable mechanism of
WeCLIP without copying its CLIP/text pipeline, checkpoints, or datasets.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AffinityDecoderConfig:
    input_dim: int = 768
    hidden_dim: int = 128
    affinity_dim: int = 64
    guidance_channels: int = 3
    decoder_scale: int = 2
    smoothmax_alpha: float = 12.0
    affinity_radius: int = 2
    affinity_temperature: float = 0.20
    frozen_similarity_power: float = 2.0
    propagation_steps: int = 2
    propagation_residual: float = 0.50
    foreground_quantile: float = 0.99
    background_quantile: float = 0.50

    def __post_init__(self) -> None:
        integer_values = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "affinity_dim": self.affinity_dim,
            "guidance_channels": self.guidance_channels,
            "decoder_scale": self.decoder_scale,
            "propagation_steps": self.propagation_steps,
        }
        if any(value <= 0 for value in integer_values.values()):
            raise ValueError(f"Decoder dimensions/steps must be positive: {integer_values}")
        if self.affinity_radius < 1:
            raise ValueError("affinity_radius must be positive")
        if self.smoothmax_alpha <= 0 or self.affinity_temperature <= 0:
            raise ValueError("SmoothMax alpha and affinity temperature must be positive")
        if self.frozen_similarity_power <= 0:
            raise ValueError("frozen_similarity_power must be positive")
        if not 0.0 <= self.propagation_residual <= 1.0:
            raise ValueError("propagation_residual must lie in [0,1]")
        if not (
            0.0 <= self.background_quantile
            < self.foreground_quantile
            <= 1.0
        ):
            raise ValueError("Pseudo-label quantiles are invalid")


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(values)
        update = self.pointwise(update)
        update = self.normalization(update)
        return values + self.activation(update)


class RadDinoSpatialDecoder(nn.Module):
    """Decode a 32x32 token grid into a guidance-aware 64x64 tumor map."""

    def __init__(self, config: AffinityDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AffinityDecoderConfig()
        c = self.config.hidden_dim
        self.projection = nn.Sequential(
            nn.Conv2d(self.config.input_dim, c, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            _ResidualSpatialBlock(c),
            _ResidualSpatialBlock(c),
            _ResidualSpatialBlock(c),
        )
        self.affinity_projection = nn.Conv2d(
            c, self.config.affinity_dim, kernel_size=1, bias=False
        )
        guidance_dim = max(16, c // 4)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(
                self.config.guidance_channels,
                guidance_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(guidance_dim), guidance_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(c + guidance_dim, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
            _ResidualSpatialBlock(c),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def _token_grid(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(patch_tokens)
        if values.ndim == 4:
            if values.shape[-1] != self.config.input_dim:
                raise ValueError("Patch-token embedding dimension does not match decoder")
            return values.permute(0, 3, 1, 2).contiguous()
        if values.ndim == 3:
            if values.shape[-1] != self.config.input_dim:
                raise ValueError("Patch-token embedding dimension does not match decoder")
            side = int(values.shape[1] ** 0.5)
            if side * side != values.shape[1]:
                raise ValueError("Flat patch-token count must be a square")
            values = values.reshape(values.shape[0], side, side, values.shape[-1])
            return values.permute(0, 3, 1, 2).contiguous()
        raise ValueError("patch_tokens must have shape [B,H,W,D] or [B,N,D]")

    def forward(
        self,
        patch_tokens: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._token_grid(patch_tokens)
        features = self.spatial(self.projection(tokens))
        affinity_features = self.affinity_projection(features)
        target_size = (
            features.shape[-2] * self.config.decoder_scale,
            features.shape[-1] * self.config.decoder_scale,
        )
        guidance_values = torch.as_tensor(
            guidance, device=features.device, dtype=features.dtype
        )
        if (
            guidance_values.ndim != 4
            or guidance_values.shape[0] != features.shape[0]
            or guidance_values.shape[1] != self.config.guidance_channels
        ):
            raise ValueError("guidance must have shape [B,guidance_channels,H,W]")
        guidance_values = F.interpolate(
            guidance_values,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        upsampled = F.interpolate(
            features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.fusion(
            torch.cat(
                [upsampled, self.guidance_projection(guidance_values)],
                dim=1,
            )
        )
        return logits, affinity_features


def smoothmax_probability(
    probabilities: torch.Tensor,
    *,
    alpha: float = AffinityDecoderConfig.smoothmax_alpha,
) -> torch.Tensor:
    values = torch.as_tensor(probabilities)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3 or alpha <= 0 or not torch.isfinite(values).all():
        raise ValueError("probabilities must be finite [B,H,W] and alpha positive")
    flat = values.flatten(1)
    weights = torch.softmax(float(alpha) * flat, dim=1)
    return (weights * flat).sum(dim=1)


def image_level_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = AffinityDecoderConfig.smoothmax_alpha,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(logits)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,H,W]")
    targets = torch.as_tensor(
        labels, device=values.device, dtype=values.dtype
    ).reshape(-1)
    pooled = smoothmax_probability(torch.sigmoid(values), alpha=alpha)
    if pooled.shape != targets.shape:
        raise ValueError("labels must contain one binary target per image")
    return F.binary_cross_entropy(pooled, targets), pooled


def _shift(
    values: torch.Tensor,
    *,
    dy: int,
    dx: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 4 or abs(dy) > radius or abs(dx) > radius:
        raise ValueError("Invalid tensor/offset for local shift")
    height, width = values.shape[-2:]
    padded = F.pad(values, (radius, radius, radius, radius))
    shifted = padded[
        :,
        :,
        radius + dy : radius + dy + height,
        radius + dx : radius + dx + width,
    ]
    valid_source = torch.ones(
        (values.shape[0], 1, height, width),
        device=values.device,
        dtype=values.dtype,
    )
    valid_padded = F.pad(valid_source, (radius, radius, radius, radius))
    valid = valid_padded[
        :,
        :,
        radius + dy : radius + dy + height,
        radius + dx : radius + dx + width,
    ]
    return shifted, valid


def local_affinity(
    decoder_features: torch.Tensor,
    frozen_tokens: torch.Tensor,
    *,
    radius: int = AffinityDecoderConfig.affinity_radius,
    temperature: float = AffinityDecoderConfig.affinity_temperature,
    frozen_similarity_power: float = (
        AffinityDecoderConfig.frozen_similarity_power
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized filtered weights, learned affinities and validity."""
    features = torch.as_tensor(decoder_features)
    tokens = torch.as_tensor(
        frozen_tokens, device=features.device, dtype=features.dtype
    )
    if features.ndim != 4 or tokens.ndim != 4:
        raise ValueError("features and frozen_tokens must be four-dimensional")
    if tokens.shape[0] != features.shape[0] or tokens.shape[1:3] != features.shape[2:]:
        raise ValueError("frozen_tokens must have shape [B,H,W,D] aligned to features")
    if radius < 1 or temperature <= 0 or frozen_similarity_power <= 0:
        raise ValueError("Invalid affinity hyperparameters")
    features = F.normalize(features, dim=1)
    tokens = F.normalize(tokens.permute(0, 3, 1, 2).contiguous(), dim=1)
    learned_values: list[torch.Tensor] = []
    filtered_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted_features, valid = _shift(
                features, dy=dy, dx=dx, radius=radius
            )
            shifted_tokens, _ = _shift(tokens, dy=dy, dx=dx, radius=radius)
            learned = torch.sigmoid(
                (features * shifted_features).sum(dim=1, keepdim=True)
                / float(temperature)
            )
            frozen = torch.clamp(
                (tokens * shifted_tokens).sum(dim=1, keepdim=True),
                min=0.0,
                max=1.0,
            ).pow(float(frozen_similarity_power))
            if dy == 0 and dx == 0:
                filtered = valid
            else:
                filtered = learned * frozen * valid
            learned_values.append(learned)
            filtered_values.append(filtered)
            valid_values.append(valid)
    learned_stack = torch.cat(learned_values, dim=1)
    filtered_stack = torch.cat(filtered_values, dim=1)
    valid_stack = torch.cat(valid_values, dim=1)
    weights = filtered_stack / filtered_stack.sum(dim=1, keepdim=True).clamp_min(
        1.0e-6
    )
    return weights, learned_stack, valid_stack


def propagate_seed_preserving(
    source: torch.Tensor,
    weights: torch.Tensor,
    *,
    radius: int,
    steps: int = AffinityDecoderConfig.propagation_steps,
    residual: float = AffinityDecoderConfig.propagation_residual,
) -> torch.Tensor:
    values = torch.as_tensor(source)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    expected_neighbors = (2 * radius + 1) ** 2
    if (
        values.ndim != 4
        or values.shape[1] != 1
        or weights.ndim != 4
        or weights.shape[0] != values.shape[0]
        or weights.shape[1] != expected_neighbors
        or weights.shape[2:] != values.shape[2:]
        or steps <= 0
        or not 0.0 <= residual <= 1.0
    ):
        raise ValueError("Source/weight geometry or propagation settings are invalid")
    current = values
    for _ in range(steps):
        propagated = torch.zeros_like(current)
        index = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor, _valid = _shift(
                    current, dy=dy, dx=dx, radius=radius
                )
                propagated = propagated + weights[:, index : index + 1] * neighbor
                index += 1
        blended = float(residual) * current + (1.0 - float(residual)) * propagated
        current = torch.maximum(current, blended)
    return current.clamp(0.0, 1.0)


def confidence_masks_from_teacher(
    teacher: torch.Tensor,
    *,
    foreground_quantile: float = AffinityDecoderConfig.foreground_quantile,
    background_quantile: float = AffinityDecoderConfig.background_quantile,
    valid_region: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mine disjoint top/bottom-rank masks without using spatial GT.

    Ranking is performed at the native teacher resolution before any resize,
    so an isolated high-confidence seed cannot disappear through bilinear
    interpolation. A validity mask may exclude deterministic square padding.
    """
    values = torch.as_tensor(teacher)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] != 1 or not torch.isfinite(values).all():
        raise ValueError("teacher must be finite with shape [B,1,H,W]")
    if not 0.0 <= background_quantile < foreground_quantile <= 1.0:
        raise ValueError("Pseudo-label quantiles are invalid")
    if valid_region is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid_values = torch.as_tensor(valid_region, device=values.device)
        if valid_values.ndim == 3:
            valid_values = valid_values.unsqueeze(1)
        if valid_values.shape != values.shape:
            raise ValueError("valid_region must match teacher geometry")
        valid = valid_values > 0.5
    foreground = torch.zeros_like(valid)
    background = torch.zeros_like(valid)
    for batch_index in range(values.shape[0]):
        flat_valid = torch.nonzero(
            valid[batch_index, 0].reshape(-1), as_tuple=False
        ).reshape(-1)
        if flat_valid.numel() == 0:
            raise ValueError("Every teacher map needs at least one valid pixel")
        flat_values = values[batch_index, 0].reshape(-1)
        ordered = torch.argsort(flat_values[flat_valid], stable=True)
        count = int(flat_valid.numel())
        foreground_count = max(
            1, math.ceil((1.0 - foreground_quantile) * count)
        )
        background_count = max(
            1, math.ceil(background_quantile * count)
        )
        if foreground_count + background_count > count:
            raise ValueError("Pseudo-label quantiles leave no disjoint ranks")
        background_indices = flat_valid[ordered[:background_count]]
        foreground_indices = flat_valid[ordered[-foreground_count:]]
        foreground[batch_index, 0].reshape(-1)[foreground_indices] = True
        background[batch_index, 0].reshape(-1)[background_indices] = True
    return foreground, background


def _resize_confidence_mask(
    mask: torch.Tensor,
    *,
    size: tuple[int, int],
    preserve_any: bool,
) -> torch.Tensor:
    if mask.shape[-2:] == size:
        return mask
    if size[0] >= mask.shape[-2] and size[1] >= mask.shape[-1]:
        return F.interpolate(mask.float(), size=size, mode="nearest") > 0.5
    if preserve_any:
        resized = F.adaptive_max_pool2d(mask.float(), output_size=size)
    else:
        resized = 1.0 - F.adaptive_max_pool2d((~mask).float(), output_size=size)
    return resized > 0.5


def affinity_pair_loss(
    learned_affinity: torch.Tensor,
    validity: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    radius: int,
    foreground_quantile: float = AffinityDecoderConfig.foreground_quantile,
    background_quantile: float = AffinityDecoderConfig.background_quantile,
    valid_region: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced local pair loss mined only from confident positive images."""
    values = torch.as_tensor(teacher)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    targets = torch.as_tensor(labels, device=values.device).reshape(-1) > 0.5
    expected_neighbors = (2 * radius + 1) ** 2
    if (
        values.ndim != 4
        or values.shape[1] != 1
        or learned_affinity.shape != validity.shape
        or learned_affinity.shape[0] != values.shape[0]
        or learned_affinity.shape[1] != expected_neighbors
        or learned_affinity.shape[2:] != values.shape[2:]
        or targets.shape[0] != values.shape[0]
    ):
        raise ValueError("Affinity supervision geometry mismatch")
    foreground, background = confidence_masks_from_teacher(
        values,
        foreground_quantile=foreground_quantile,
        background_quantile=background_quantile,
        valid_region=valid_region,
    )
    confident = foreground | background
    losses: list[torch.Tensor] = []
    index = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                index += 1
                continue
            shifted_foreground, _ = _shift(
                foreground.float(), dy=dy, dx=dx, radius=radius
            )
            shifted_background, _ = _shift(
                background.float(), dy=dy, dx=dx, radius=radius
            )
            shifted_confident = (shifted_foreground + shifted_background) > 0.5
            pair_valid = (
                confident
                & shifted_confident
                & (validity[:, index : index + 1] > 0.5)
                & targets[:, None, None, None]
            )
            same = (
                (foreground & (shifted_foreground > 0.5))
                | (background & (shifted_background > 0.5))
            ) & pair_valid
            different = pair_valid & ~same
            prediction = learned_affinity[:, index : index + 1].clamp(
                1.0e-6, 1.0 - 1.0e-6
            )
            local: list[torch.Tensor] = []
            if same.any():
                local.append(-torch.log(prediction[same]).mean())
            if different.any():
                local.append(-torch.log1p(-prediction[different]).mean())
            if local:
                losses.append(torch.stack(local).mean())
            index += 1
    if not losses:
        return learned_affinity.sum() * 0.0
    return torch.stack(losses).mean()


def masked_pseudo_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    foreground_quantile: float = AffinityDecoderConfig.foreground_quantile,
    background_quantile: float = AffinityDecoderConfig.background_quantile,
    valid_region: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced confident-region BCE; normal images are explicit all-background."""
    values = torch.as_tensor(logits)
    teacher_values = torch.as_tensor(
        teacher, device=values.device, dtype=values.dtype
    )
    if teacher_values.ndim == 3:
        teacher_values = teacher_values.unsqueeze(1)
    if (
        values.ndim != 4
        or values.shape[1] != 1
        or teacher_values.ndim != 4
        or teacher_values.shape[1] != 1
        or teacher_values.shape[0] != values.shape[0]
    ):
        raise ValueError("Logit/teacher geometry mismatch")
    foreground_native, background_native = confidence_masks_from_teacher(
        teacher_values,
        foreground_quantile=foreground_quantile,
        background_quantile=background_quantile,
        valid_region=valid_region,
    )
    foreground_masks = _resize_confidence_mask(
        foreground_native,
        size=values.shape[-2:],
        preserve_any=True,
    )
    background_masks = _resize_confidence_mask(
        background_native,
        size=values.shape[-2:],
        preserve_any=False,
    )
    targets = torch.as_tensor(
        labels, device=values.device, dtype=values.dtype
    ).reshape(-1)
    if targets.shape[0] != values.shape[0]:
        raise ValueError("labels must contain one target per image")
    losses: list[torch.Tensor] = []
    for index in range(values.shape[0]):
        if targets[index] <= 0.5:
            losses.append(
                F.binary_cross_entropy_with_logits(
                    values[index],
                    torch.zeros_like(values[index]),
                )
            )
            continue
        foreground = foreground_masks[index]
        background = background_masks[index]
        if not foreground.any() or not background.any():
            raise RuntimeError(
                "Positive image lacks disjoint foreground/background pseudo pixels"
            )
        parts: list[torch.Tensor] = []
        parts.append(
            F.binary_cross_entropy_with_logits(
                values[index][foreground],
                torch.ones_like(values[index][foreground]),
            )
        )
        parts.append(
            F.binary_cross_entropy_with_logits(
                values[index][background],
                torch.zeros_like(values[index][background]),
            )
        )
        losses.append(torch.stack(parts).mean())
    return torch.stack(losses).mean()


def make_guidance(pixel_values_01: torch.Tensor, *, output_size: int) -> torch.Tensor:
    """Create label-free grayscale and signed finite-difference guidance."""
    values = torch.as_tensor(pixel_values_01)
    if (
        values.ndim != 4
        or values.shape[1] != 3
        or output_size <= 0
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Expected finite RGB pixels [B,3,H,W] and positive size")
    gray = (
        0.2989 * values[:, 0:1]
        + 0.5870 * values[:, 1:2]
        + 0.1140 * values[:, 2:3]
    )
    gray = F.interpolate(
        gray,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    return torch.cat([gray, dx, dy], dim=1)
