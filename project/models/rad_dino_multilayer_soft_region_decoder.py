from __future__ import annotations

"""Multi-layer RAD-DINO decoder with variable-area soft-region supervision.

The encoder and its random projection are frozen outside this module.  The
decoder consumes several projected spatial-token layers, image-derived
guidance, image labels, and a train-normal-calibrated anomaly teacher.  No
task-specific spatial annotation is required by any function in this file.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MultiLayerSoftRegionConfig:
    input_dim: int = 128
    layer_count: int = 3
    hidden_dim: int = 128
    affinity_dim: int = 64
    guidance_channels: int = 3
    decoder_scale: int = 2
    smoothmax_alpha: float = 12.0
    affinity_radius: int = 2
    affinity_temperature: float = 0.20
    frozen_similarity_power: float = 2.0
    refinement_steps: int = 3
    refinement_residual: float = 0.50
    foreground_start: float = 0.90
    background_end: float = 0.50

    def __post_init__(self) -> None:
        dimensions = {
            "input_dim": self.input_dim,
            "layer_count": self.layer_count,
            "hidden_dim": self.hidden_dim,
            "affinity_dim": self.affinity_dim,
            "guidance_channels": self.guidance_channels,
            "decoder_scale": self.decoder_scale,
            "refinement_steps": self.refinement_steps,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(f"Decoder dimensions must be positive: {dimensions}")
        if self.affinity_radius < 1:
            raise ValueError("affinity_radius must be positive")
        if self.smoothmax_alpha <= 0 or self.affinity_temperature <= 0:
            raise ValueError("SmoothMax alpha and affinity temperature must be positive")
        if self.frozen_similarity_power <= 0:
            raise ValueError("frozen_similarity_power must be positive")
        if not 0.0 <= self.refinement_residual <= 1.0:
            raise ValueError("refinement_residual must lie in [0,1]")
        if not 0.0 <= self.background_end < self.foreground_start < 1.0:
            raise ValueError("Soft-region confidence thresholds are invalid")


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


class RadDinoMultiLayerSoftRegionDecoder(nn.Module):
    """Fuse projected intermediate/final tokens into a 64x64 tumor map."""

    def __init__(
        self,
        config: MultiLayerSoftRegionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultiLayerSoftRegionConfig()
        hidden = self.config.hidden_dim
        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.config.input_dim,
                        hidden,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(hidden), hidden),
                    nn.GELU(),
                )
                for _ in range(self.config.layer_count)
            ]
        )
        self.layer_logits = nn.Parameter(torch.zeros(self.config.layer_count))
        self.fusion = nn.Sequential(
            nn.Conv2d(
                hidden * self.config.layer_count,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            _ResidualSpatialBlock(hidden),
            _ResidualSpatialBlock(hidden),
            _ResidualSpatialBlock(hidden),
        )
        self.affinity_projection = nn.Conv2d(
            hidden,
            self.config.affinity_dim,
            kernel_size=1,
            bias=False,
        )
        guidance_dim = max(16, hidden // 4)
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
        self.prediction = nn.Sequential(
            nn.Conv2d(
                hidden + guidance_dim,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            _ResidualSpatialBlock(hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def _token_layers(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(patch_tokens)
        if values.ndim == 5:
            if (
                values.shape[1] != self.config.layer_count
                or values.shape[-1] != self.config.input_dim
            ):
                raise ValueError("Projected token layer/dimension mismatch")
            return values
        if values.ndim == 4:
            if (
                values.shape[1] != self.config.layer_count
                or values.shape[-1] != self.config.input_dim
            ):
                raise ValueError("Projected token layer/dimension mismatch")
            side = int(values.shape[2] ** 0.5)
            if side * side != values.shape[2]:
                raise ValueError("Flat patch-token count must be square")
            return values.reshape(
                values.shape[0],
                values.shape[1],
                side,
                side,
                values.shape[-1],
            )
        raise ValueError("patch_tokens must have shape [B,L,H,W,D] or [B,L,N,D]")

    def forward(
        self,
        patch_tokens: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layers = self._token_layers(patch_tokens)
        projected: list[torch.Tensor] = []
        weights = torch.softmax(self.layer_logits, dim=0)
        for layer_index, projection in enumerate(self.layer_projections):
            values = layers[:, layer_index].permute(0, 3, 1, 2).contiguous()
            projected.append(
                projection(values) * weights[layer_index] * self.config.layer_count
            )
        features = self.fusion(torch.cat(projected, dim=1))
        affinity_features = self.affinity_projection(features)
        target_size = (
            features.shape[-2] * self.config.decoder_scale,
            features.shape[-1] * self.config.decoder_scale,
        )
        guidance_values = torch.as_tensor(
            guidance,
            device=features.device,
            dtype=features.dtype,
        )
        if (
            guidance_values.ndim != 4
            or guidance_values.shape[0] != features.shape[0]
            or guidance_values.shape[1] != self.config.guidance_channels
        ):
            raise ValueError("guidance must have shape [B,C,H,W]")
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
        logits = self.prediction(
            torch.cat(
                [upsampled, self.guidance_projection(guidance_values)],
                dim=1,
            )
        )
        return logits, affinity_features, weights


def make_guidance(pixel_values_01: torch.Tensor, *, output_size: int) -> torch.Tensor:
    values = torch.as_tensor(pixel_values_01)
    if (
        values.ndim != 4
        or values.shape[1] != 3
        or output_size <= 0
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Expected finite RGB pixels [B,3,H,W]")
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


def smoothmax_probability(
    probabilities: torch.Tensor,
    *,
    alpha: float = MultiLayerSoftRegionConfig.smoothmax_alpha,
) -> torch.Tensor:
    values = torch.as_tensor(probabilities)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3 or alpha <= 0 or not torch.isfinite(values).all():
        raise ValueError("probabilities must be finite [B,H,W]")
    flattened = values.flatten(1)
    weights = torch.softmax(float(alpha) * flattened, dim=1)
    return (weights * flattened).sum(dim=1)


def image_level_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = MultiLayerSoftRegionConfig.smoothmax_alpha,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(logits)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,H,W]")
    targets = torch.as_tensor(
        labels,
        device=values.device,
        dtype=values.dtype,
    ).reshape(-1)
    pooled = smoothmax_probability(torch.sigmoid(values), alpha=alpha)
    if pooled.shape != targets.shape:
        raise ValueError("labels must contain one binary target per image")
    return F.binary_cross_entropy(pooled, targets), pooled


def soft_region_weights(
    teacher: torch.Tensor,
    *,
    foreground_start: float = MultiLayerSoftRegionConfig.foreground_start,
    background_end: float = MultiLayerSoftRegionConfig.background_end,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(teacher)
    if (
        not torch.isfinite(values).all()
        or not 0.0 <= background_end < foreground_start < 1.0
    ):
        raise ValueError("Teacher/soft-region thresholds are invalid")
    foreground = ((values - float(foreground_start)) / (1.0 - foreground_start)).clamp(
        0.0,
        1.0,
    )
    background = ((float(background_end) - values) / background_end).clamp(
        0.0,
        1.0,
    )
    return foreground, background


def soft_region_pseudo_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_region: torch.Tensor | None = None,
    foreground_start: float = MultiLayerSoftRegionConfig.foreground_start,
    background_end: float = MultiLayerSoftRegionConfig.background_end,
) -> torch.Tensor:
    values = torch.as_tensor(logits)
    teacher_values = torch.as_tensor(
        teacher,
        device=values.device,
        dtype=values.dtype,
    )
    if teacher_values.ndim == 3:
        teacher_values = teacher_values.unsqueeze(1)
    if (
        values.ndim != 4
        or values.shape[1] != 1
        or teacher_values.ndim != 4
        or teacher_values.shape[0] != values.shape[0]
        or teacher_values.shape[1] != 1
    ):
        raise ValueError("Logit/teacher geometry mismatch")
    targets = torch.as_tensor(
        labels,
        device=values.device,
        dtype=values.dtype,
    ).reshape(-1)
    if targets.shape[0] != values.shape[0]:
        raise ValueError("labels must contain one target per image")
    teacher_values = F.interpolate(
        teacher_values,
        size=values.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    if valid_region is None:
        valid = torch.ones_like(values)
    else:
        valid = torch.as_tensor(
            valid_region,
            device=values.device,
            dtype=values.dtype,
        )
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[0] != values.shape[0] or valid.shape[1] != 1:
            raise ValueError("valid_region batch/channel geometry mismatch")
        valid = F.interpolate(valid, size=values.shape[-2:], mode="nearest")
    foreground, background = soft_region_weights(
        teacher_values,
        foreground_start=foreground_start,
        background_end=background_end,
    )
    losses: list[torch.Tensor] = []
    for index in range(values.shape[0]):
        item_valid = valid[index].clamp(0.0, 1.0)
        if targets[index] <= 0.5:
            dense = F.binary_cross_entropy_with_logits(
                values[index],
                torch.zeros_like(values[index]),
                reduction="none",
            )
            losses.append(
                (dense * item_valid).sum() / item_valid.sum().clamp_min(1.0)
            )
            continue
        components: list[torch.Tensor] = []
        foreground_weight = foreground[index] * item_valid
        background_weight = background[index] * item_valid
        if foreground_weight.sum() > 0:
            dense_foreground = F.binary_cross_entropy_with_logits(
                values[index],
                torch.ones_like(values[index]),
                reduction="none",
            )
            components.append(
                (dense_foreground * foreground_weight).sum()
                / foreground_weight.sum().clamp_min(1.0e-6)
            )
        if background_weight.sum() > 0:
            dense_background = F.binary_cross_entropy_with_logits(
                values[index],
                torch.zeros_like(values[index]),
                reduction="none",
            )
            components.append(
                (dense_background * background_weight).sum()
                / background_weight.sum().clamp_min(1.0e-6)
            )
        if not components:
            raise RuntimeError("Positive image has no calibrated soft-region evidence")
        losses.append(torch.stack(components).mean())
    return torch.stack(losses).mean()


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
    source_validity = torch.ones(
        (values.shape[0], 1, height, width),
        device=values.device,
        dtype=values.dtype,
    )
    valid_padded = F.pad(
        source_validity,
        (radius, radius, radius, radius),
    )
    validity = valid_padded[
        :,
        :,
        radius + dy : radius + dy + height,
        radius + dx : radius + dx + width,
    ]
    return shifted, validity


def local_affinity(
    decoder_features: torch.Tensor,
    frozen_tokens: torch.Tensor,
    *,
    radius: int = MultiLayerSoftRegionConfig.affinity_radius,
    temperature: float = MultiLayerSoftRegionConfig.affinity_temperature,
    frozen_similarity_power: float = (
        MultiLayerSoftRegionConfig.frozen_similarity_power
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(decoder_features)
    tokens = torch.as_tensor(
        frozen_tokens,
        device=features.device,
        dtype=features.dtype,
    )
    if features.ndim != 4 or tokens.ndim != 4:
        raise ValueError("features and frozen_tokens must be four-dimensional")
    if tokens.shape[:3] != (
        features.shape[0],
        features.shape[2],
        features.shape[3],
    ):
        raise ValueError("frozen_tokens must have shape [B,H,W,D]")
    if radius < 1 or temperature <= 0 or frozen_similarity_power <= 0:
        raise ValueError("Invalid affinity hyperparameters")
    normalized_features = F.normalize(features, dim=1)
    normalized_tokens = F.normalize(
        tokens.permute(0, 3, 1, 2).contiguous(),
        dim=1,
    )
    learned_values: list[torch.Tensor] = []
    filtered_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted_features, validity = _shift(
                normalized_features,
                dy=dy,
                dx=dx,
                radius=radius,
            )
            shifted_tokens, _ = _shift(
                normalized_tokens,
                dy=dy,
                dx=dx,
                radius=radius,
            )
            learned = torch.sigmoid(
                (normalized_features * shifted_features).sum(
                    dim=1,
                    keepdim=True,
                )
                / float(temperature)
            )
            frozen = torch.clamp(
                (normalized_tokens * shifted_tokens).sum(dim=1, keepdim=True),
                min=0.0,
                max=1.0,
            ).pow(float(frozen_similarity_power))
            filtered = validity if (dy == 0 and dx == 0) else (
                learned * frozen * validity
            )
            learned_values.append(learned)
            filtered_values.append(filtered)
            valid_values.append(validity)
    learned_stack = torch.cat(learned_values, dim=1)
    filtered_stack = torch.cat(filtered_values, dim=1)
    valid_stack = torch.cat(valid_values, dim=1)
    weights = filtered_stack / filtered_stack.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    return weights, learned_stack, valid_stack


def _propagate(
    source: torch.Tensor,
    weights: torch.Tensor,
    *,
    radius: int,
    steps: int,
    residual: float,
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
        raise ValueError("Propagation geometry/settings are invalid")
    current = values
    for _ in range(steps):
        update = torch.zeros_like(current)
        neighbor_index = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor, _ = _shift(
                    current,
                    dy=dy,
                    dx=dx,
                    radius=radius,
                )
                update = update + weights[:, neighbor_index : neighbor_index + 1] * neighbor
                neighbor_index += 1
        blended = float(residual) * current + (1.0 - float(residual)) * update
        current = torch.maximum(current, blended)
    return current.clamp(0.0, 1.0)


def bidirectional_affinity_refinement(
    teacher: torch.Tensor,
    weights: torch.Tensor,
    *,
    radius: int,
    steps: int = MultiLayerSoftRegionConfig.refinement_steps,
    residual: float = MultiLayerSoftRegionConfig.refinement_residual,
) -> torch.Tensor:
    values = torch.as_tensor(teacher)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if (
        values.ndim != 4
        or values.shape[1] != 1
        or not torch.isfinite(values).all()
    ):
        raise ValueError("teacher must be finite [B,1,H,W]")
    values = values.clamp(0.0, 1.0)
    foreground = _propagate(
        values,
        weights,
        radius=radius,
        steps=steps,
        residual=residual,
    )
    background = _propagate(
        1.0 - values,
        weights,
        radius=radius,
        steps=steps,
        residual=residual,
    )
    return foreground / (foreground + background).clamp_min(1.0e-6)


def soft_affinity_pair_loss(
    learned_affinity: torch.Tensor,
    validity: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    radius: int,
    foreground_start: float = MultiLayerSoftRegionConfig.foreground_start,
    background_end: float = MultiLayerSoftRegionConfig.background_end,
    valid_region: torch.Tensor | None = None,
) -> torch.Tensor:
    values = torch.as_tensor(
        teacher,
        device=learned_affinity.device,
        dtype=learned_affinity.dtype,
    )
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
    foreground_weight, background_weight = soft_region_weights(
        values,
        foreground_start=foreground_start,
        background_end=background_end,
    )
    if valid_region is not None:
        region = torch.as_tensor(
            valid_region,
            device=values.device,
            dtype=values.dtype,
        )
        if region.ndim == 3:
            region = region.unsqueeze(1)
        if region.shape != values.shape:
            raise ValueError("valid_region must match teacher geometry")
        foreground_weight = foreground_weight * region
        background_weight = background_weight * region
    losses: list[torch.Tensor] = []
    neighbor_index = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                neighbor_index += 1
                continue
            shifted_foreground, _ = _shift(
                foreground_weight,
                dy=dy,
                dx=dx,
                radius=radius,
            )
            shifted_background, _ = _shift(
                background_weight,
                dy=dy,
                dx=dx,
                radius=radius,
            )
            same_weight = torch.maximum(
                torch.minimum(foreground_weight, shifted_foreground),
                torch.minimum(background_weight, shifted_background),
            )
            different_weight = torch.maximum(
                torch.minimum(foreground_weight, shifted_background),
                torch.minimum(background_weight, shifted_foreground),
            )
            pair_valid = (
                validity[:, neighbor_index : neighbor_index + 1]
                * targets[:, None, None, None]
            )
            same_weight = same_weight * pair_valid
            different_weight = different_weight * pair_valid
            prediction = learned_affinity[
                :,
                neighbor_index : neighbor_index + 1,
            ].clamp(1.0e-6, 1.0 - 1.0e-6)
            components: list[torch.Tensor] = []
            if same_weight.sum() > 0:
                components.append(
                    (-torch.log(prediction) * same_weight).sum()
                    / same_weight.sum().clamp_min(1.0e-6)
                )
            if different_weight.sum() > 0:
                components.append(
                    (-torch.log1p(-prediction) * different_weight).sum()
                    / different_weight.sum().clamp_min(1.0e-6)
                )
            if components:
                losses.append(torch.stack(components).mean())
            neighbor_index += 1
    if not losses:
        return learned_affinity.sum() * 0.0
    return torch.stack(losses).mean()


def horizontal_flip_consistency_loss(
    logits: torch.Tensor,
    flipped_logits: torch.Tensor,
    *,
    valid_region: torch.Tensor | None = None,
) -> torch.Tensor:
    values = torch.as_tensor(logits)
    flipped = torch.as_tensor(
        flipped_logits,
        device=values.device,
        dtype=values.dtype,
    ).flip(-1)
    if values.shape != flipped.shape or values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("Aligned logits must share [B,1,H,W] geometry")
    difference = (
        torch.sigmoid(values) - torch.sigmoid(flipped)
    ).pow(2)
    if valid_region is None:
        return difference.mean()
    valid = torch.as_tensor(
        valid_region,
        device=values.device,
        dtype=values.dtype,
    )
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if valid.shape[0] != values.shape[0] or valid.shape[1] != 1:
        raise ValueError("valid_region batch/channel geometry mismatch")
    valid = F.interpolate(valid, size=values.shape[-2:], mode="nearest")
    return (difference * valid).sum() / valid.sum().clamp_min(1.0)
