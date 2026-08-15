from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def classic_cam(feature_map: torch.Tensor, classifier: nn.Linear, class_index: torch.Tensor) -> torch.Tensor:
    weights = classifier.weight[class_index]  # [B, C]
    return torch.einsum("bchw,bc->bhw", feature_map, weights)


def tile_2x2(images: torch.Tensor) -> torch.Tensor:
    b, c, h, w = images.shape
    assert h % 2 == 0 and w % 2 == 0, f"tile_2x2 requires even H, W, got {h}x{w}"
    half_h, half_w = h // 2, w // 2
    top, bottom = images[:, :, :half_h, :], images[:, :, half_h:, :]
    tiles = [
        top[:, :, :, :half_w], top[:, :, :, half_w:],
        bottom[:, :, :, :half_w], bottom[:, :, :, half_w:],
    ]
    return torch.cat(tiles, dim=0)


def merge_2x2(tiled_cam: torch.Tensor, batch_size: int) -> torch.Tensor:
    top_left, top_right, bottom_left, bottom_right = (
        tiled_cam[0 * batch_size : 1 * batch_size],
        tiled_cam[1 * batch_size : 2 * batch_size],
        tiled_cam[2 * batch_size : 3 * batch_size],
        tiled_cam[3 * batch_size : 4 * batch_size],
    )
    top = torch.cat([top_left, top_right], dim=-1)
    bottom = torch.cat([bottom_left, bottom_right], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def puzzle_alpha(epoch: int, total_epochs: int, alpha_max: float = 4.0) -> float:
    half_life = max(1, total_epochs // 2)
    return min(alpha_max * epoch / half_life, alpha_max)


def _normalize_cam(cam: torch.Tensor, degenerate_threshold: float = 1e-4) -> torch.Tensor:
    flat = cam.view(cam.shape[0], -1)
    mn = flat.min(dim=1).values.view(-1, 1, 1)
    mx = flat.max(dim=1).values.view(-1, 1, 1)
    cam_range = mx - mn
    normalized = (cam - mn) / (cam_range + 1e-8)
    return torch.where(cam_range > degenerate_threshold, normalized, torch.zeros_like(normalized))


def puzzle_cam_consistency_loss(
    model: nn.Module,
    images: torch.Tensor,
    target_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = images.shape[0]

    with torch.cuda.amp.autocast(enabled=False):
        full_features = model.forward_features(images).float()
        full_cam = classic_cam(full_features, model.classifier, target_class)

        tiled_images = tile_2x2(images)
        tiled_target_class = target_class.repeat(4)
        tiled_features = model.forward_features(tiled_images).float()
        tiled_cam = classic_cam(tiled_features, model.classifier, tiled_target_class)
        reconstructed = merge_2x2(tiled_cam, batch_size)
        reconstructed = F.interpolate(
            reconstructed.unsqueeze(1), size=full_cam.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        full_cam_norm = _normalize_cam(full_cam)
        reconstructed_norm = _normalize_cam(reconstructed)
        re_loss = (full_cam_norm - reconstructed_norm).abs().mean()

        merged_features = merge_2x2(tiled_features, batch_size)
        merged_pooled = model.avgpool(merged_features).flatten(1)
        merged_logits = model.classifier(merged_pooled)
        p_cls_loss = F.cross_entropy(merged_logits, target_class)

    return full_cam_norm, reconstructed_norm, re_loss, p_cls_loss
