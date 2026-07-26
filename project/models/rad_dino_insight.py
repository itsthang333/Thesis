from __future__ import annotations

"""Image-label-only local/context heatmap head inspired by INSIGHT.

The frozen encoder is intentionally kept outside this module.  The head
receives spatial RAD-DINO patch tokens and learns a dense heatmap using only
the image-level tumor flag.  It transfers the paper's mechanism (a
fine-detail detector, a broad context-suppression branch, and SmoothMax
pooling) without copying its data pipeline or checkpoints.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class InsightMILConfig:
    input_dim: int = 768
    hidden_dim: int = 128
    detection_kernel: int = 3
    context_kernel: int = 9
    smoothmax_alpha: float = 12.0
    spectral_lambda: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        for name, value in (
            ("detection_kernel", self.detection_kernel),
            ("context_kernel", self.context_kernel),
        ):
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.smoothmax_alpha <= 0:
            raise ValueError("smoothmax_alpha must be positive")
        if self.spectral_lambda < 0:
            raise ValueError("spectral_lambda must be non-negative")


class _DepthwiseContext(nn.Module):
    """Large-receptive-field context branch with bounded parameter count."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pointwise(self.depthwise(features)))


class InsightDenseMILHead(nn.Module):
    """Dense local/context MIL head for frozen spatial patch embeddings."""

    def __init__(self, config: InsightMILConfig | None = None) -> None:
        super().__init__()
        self.config = config or InsightMILConfig()
        c = self.config.hidden_dim
        self.projection = nn.Conv2d(self.config.input_dim, c, kernel_size=1)
        self.detector = nn.Sequential(
            nn.Conv2d(
                c,
                c,
                kernel_size=self.config.detection_kernel,
                padding=self.config.detection_kernel // 2,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        self.context = nn.Sequential(
            _DepthwiseContext(c, self.config.context_kernel),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def _features(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(patch_tokens)
        if values.ndim == 4:
            if values.shape[-1] != self.config.input_dim:
                raise ValueError("Patch-token embedding dimension does not match head")
            values = values.permute(0, 3, 1, 2).contiguous()
        elif values.ndim == 3:
            if values.shape[-1] != self.config.input_dim:
                raise ValueError("Patch-token embedding dimension does not match head")
            side = int(values.shape[1] ** 0.5)
            if side * side != values.shape[1]:
                raise ValueError("Flat patch-token count must be a square")
            values = values.reshape(values.shape[0], side, side, values.shape[-1])
            values = values.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("patch_tokens must have shape [B,H,W,D] or [B,N,D]")
        return self.projection(values)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return heatmap, fused logits, detector logits and context logits."""
        features = self._features(patch_tokens)
        detector_logits = self.detector(features)
        context_logits = self.context(features)
        # INSIGHT-style broad-context suppression:
        # H = sigmoid((1 - sigmoid(H_context)) * H_detector).
        suppression = 1.0 - torch.sigmoid(context_logits)
        fused_logits = suppression * detector_logits
        heatmap = torch.sigmoid(fused_logits)
        return heatmap, fused_logits, detector_logits, context_logits


def smoothmax_pool(
    heatmap: torch.Tensor, *, alpha: float = InsightMILConfig.smoothmax_alpha
) -> torch.Tensor:
    """SmoothMax weighted average over spatial heatmap activations."""
    values = torch.as_tensor(heatmap)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError("heatmap must have shape [B,H,W] or [B,1,H,W]")
    if alpha <= 0 or not torch.isfinite(values).all():
        raise ValueError("alpha must be positive and heatmap finite")
    flat = values.flatten(1)
    weights = torch.softmax(float(alpha) * flat, dim=1)
    return (weights * flat).sum(dim=1)


def insight_mil_loss(
    fused_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = InsightMILConfig.smoothmax_alpha,
    spectral_lambda: float = InsightMILConfig.spectral_lambda,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Image-level BCE plus fixed spectral-decoupling regularization."""
    values = torch.as_tensor(fused_logits)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError("fused_logits must have shape [B,H,W] or [B,1,H,W]")
    targets = torch.as_tensor(labels, device=values.device, dtype=values.dtype).view(-1)
    pooled = smoothmax_pool(torch.sigmoid(values), alpha=alpha)
    if pooled.shape != targets.shape:
        raise ValueError("labels must contain one binary target per image")
    bce = F.binary_cross_entropy(pooled, targets)
    spectral = float(spectral_lambda) * values.square().mean()
    return bce + spectral, bce, spectral, pooled


def resize_heatmap(heatmap: torch.Tensor, *, output_size: int) -> torch.Tensor:
    """Resize a dense heatmap while preserving its probability range."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    values = torch.as_tensor(heatmap)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("heatmap must have shape [B,1,H,W] or [B,H,W]")
    return F.interpolate(
        values,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
