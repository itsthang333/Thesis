from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class HRNetMILOutput:
    dense_logits: torch.Tensor
    class_logits: torch.Tensor
    tumor_logit: torch.Tensor
    tumor_map: torch.Tensor
    features: torch.Tensor


def multi_topk_pool(logits: torch.Tensor, fractions: tuple[float, ...]) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("dense logits must have shape [B,C,H,W]")
    flat = logits.flatten(2)
    pooled: list[torch.Tensor] = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("top-k fractions must lie in (0,1]")
        count = max(1, min(flat.shape[-1], round(flat.shape[-1] * fraction)))
        pooled.append(flat.topk(count, dim=-1).values.mean(dim=-1))
    return torch.stack(pooled).mean(dim=0)


class HRNetDenseMIL(nn.Module):
    """HRNet-W48 with a stride-4 dense head and image-label MIL pooling."""

    def __init__(
        self,
        *,
        backbone_name: str = "hrnet_w48.ms_in1k",
        pretrained: bool = True,
        classes: int = 10,
        dense_channels: int = 512,
        dropout: float = 0.1,
        topk_fractions: tuple[float, ...] = (0.0005, 0.002, 0.01),
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("HRNetDenseMIL requires timm") from exc
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self.gradient_checkpointing = gradient_checkpointing
        channels = tuple(self.backbone.stage4_cfg["num_channels"])
        if len(channels) != 4:
            raise RuntimeError(f"Expected four HRNet resolutions, found {channels}")
        self.topk_fractions = topk_fractions
        self.head = nn.Sequential(
            nn.Conv2d(sum(channels), dense_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, dense_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dense_channels, dense_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(32, dense_channels // 2),
            nn.GELU(),
            nn.Conv2d(dense_channels // 2, classes, 1),
        )

    def parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict[str, object]]:
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

    def forward(self, images: torch.Tensor) -> HRNetMILOutput:
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        if self.training and self.gradient_checkpointing:
            features = checkpoint(
                lambda tensor: tuple(self.backbone.stages(tensor)),
                x,
                use_reentrant=False,
            )
        else:
            features = self.backbone.stages(x)
        target_size = features[0].shape[-2:]
        aligned = [
            feature
            if feature.shape[-2:] == target_size
            else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            for feature in features
        ]
        fused = torch.cat(aligned, dim=1)
        dense_logits = self.head(fused)
        class_logits = multi_topk_pool(dense_logits, self.topk_fractions)
        tumor_logit = torch.logsumexp(class_logits[:, 1:], dim=1) - class_logits[:, 0]
        tumor_map = torch.sigmoid(torch.logsumexp(dense_logits[:, 1:], dim=1) - dense_logits[:, 0])
        return HRNetMILOutput(dense_logits, class_logits, tumor_logit, tumor_map, fused)


def normal_suppression_loss(
    dense_logits: torch.Tensor, class_targets: torch.Tensor
) -> torch.Tensor:
    normal = class_targets == 0
    if not normal.any():
        return dense_logits.sum() * 0
    tumor_logits = torch.logsumexp(dense_logits[normal, 1:], dim=1)
    normal_logits = dense_logits[normal, 0]
    return F.softplus(tumor_logits - normal_logits).mean()


def aligned_map_consistency(
    first: torch.Tensor | None,
    second: torch.Tensor | None,
) -> torch.Tensor:
    if first is None or second is None:
        reference = first if first is not None else second
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0
    if first.shape != second.shape:
        second = F.interpolate(
            second[:, None], size=first.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    return F.smooth_l1_loss(first, second)


def hrnet_mil_loss(
    output: HRNetMILOutput,
    class_targets: torch.Tensor,
    *,
    binary_targets: torch.Tensor | None = None,
    multi_hot_targets: torch.Tensor | None = None,
    aligned_tumor_map: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    normal_weight: float = 0.5,
    consistency_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    binary_targets = (
        (class_targets > 0).float() if binary_targets is None else binary_targets.float()
    )
    if multi_hot_targets is None:
        classification = F.cross_entropy(output.class_logits, class_targets, weight=class_weights)
    else:
        if multi_hot_targets.shape != output.class_logits.shape:
            raise ValueError("multi_hot_targets must align with class logits")
        classification = F.binary_cross_entropy_with_logits(
            output.class_logits, multi_hot_targets.float(), weight=class_weights
        )
    binary = F.binary_cross_entropy_with_logits(output.tumor_logit, binary_targets)
    normal = normal_suppression_loss(output.dense_logits, class_targets)
    consistency = aligned_map_consistency(output.tumor_map, aligned_tumor_map)
    total = (
        classification + 0.5 * binary + normal_weight * normal + consistency_weight * consistency
    )
    return total, {
        "loss": total.detach(),
        "classification": classification.detach(),
        "binary": binary.detach(),
        "normal_suppression": normal.detach(),
        "map_consistency": consistency.detach(),
    }


def hrnet_tile_bag_loss(
    output: HRNetMILOutput,
    class_targets: torch.Tensor,
    multi_hot_targets: torch.Tensor,
    full_references: list[torch.Tensor],
    *,
    normal_weight: float = 0.5,
    consistency_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MIL over tiles; a positive image does not make every tile positive."""
    tile_count = output.class_logits.shape[0]
    if tile_count != len(full_references):
        raise ValueError("Each tile requires an aligned full-view reference")
    normalizer = torch.log(output.class_logits.new_tensor(float(tile_count)))
    bag_class_logits = torch.logsumexp(output.class_logits, dim=0, keepdim=True) - normalizer
    bag_tumor_logit = torch.logsumexp(output.tumor_logit, dim=0, keepdim=True) - normalizer
    classification = F.binary_cross_entropy_with_logits(bag_class_logits, multi_hot_targets.float())
    binary = F.binary_cross_entropy_with_logits(bag_tumor_logit, (class_targets > 0).float())
    normal = normal_suppression_loss(output.dense_logits, class_targets.repeat(tile_count))
    consistency = torch.stack(
        [
            F.smooth_l1_loss(output.tumor_map[index : index + 1], reference)
            for index, reference in enumerate(full_references)
        ]
    ).mean()
    total = classification + 0.5 * binary + normal_weight * normal
    total = total + consistency_weight * consistency
    return total, {
        "loss": total.detach(),
        "classification": classification.detach(),
        "binary": binary.detach(),
        "normal_suppression": normal.detach(),
        "map_consistency": consistency.detach(),
    }
