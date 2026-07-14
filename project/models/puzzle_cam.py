from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def classic_cam(feature_map: torch.Tensor, classifier: nn.Linear, class_index: torch.Tensor) -> torch.Tensor:
    """CAM (Zhou et al.) for a per-sample target class, batched and fully
    differentiable -- no gradient hooks or separate backward() call needed,
    unlike LayerCAM. Requires a GAP + single nn.Linear head (this project's
    DenseNet121AnatomyClassifier), since CAM_c = sum_k(W[c,k] * feature_map[k]).

    feature_map:  [B, C, H, W] (pre-GAP feature map, e.g. from forward_features)
    classifier:   the model's final nn.Linear(C, num_classes) head
    class_index:  [B] long tensor, one target class per sample

    Returns: [B, H, W], NOT normalized (caller decides normalization).
    """
    weights = classifier.weight[class_index]  # [B, C]
    return torch.einsum("bchw,bc->bhw", feature_map, weights)


def tile_2x2(images: torch.Tensor) -> torch.Tensor:
    """Split a batch of images into 4 non-overlapping quadrants and stack
    them along the batch dimension, PuzzleCAM-style (Jo & Yu, ICIP 2021):
    tiling happens on the raw input before the backbone, and all 4 tiles for
    the whole batch are concatenated into one enlarged batch so the tiled
    forward pass is a single model() call, not 4 separate ones.

    images: [B, C, H, W] with H, W even.
    Returns: [4*B, C, H/2, W/2], ordered [top-left]*B, [top-right]*B,
             [bottom-left]*B, [bottom-right]*B.
    """
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
    """Inverse of tile_2x2 for a CAM/feature map: reassemble 4*B tiles back
    into B full-size maps in their original spatial layout.

    tiled_cam: [4*B, H, W] or [4*B, C, H, W]
    Returns:   [B, 2H, 2W] or [B, C, 2H, 2W]
    """
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
    """PuzzleCAM's warmup schedule: linearly ramp the consistency loss weight
    from 0 to alpha_max over the first half of training, then hold at
    alpha_max. Early CAMs are noisy/inconsistent, so penalizing reconstruction
    too early can suppress classification learning before the model has
    learned anything meaningful to be consistent about.
    """
    half_life = max(1, total_epochs // 2)
    return min(alpha_max * epoch / half_life, alpha_max)


def _normalize_cam(cam: torch.Tensor, degenerate_threshold: float = 1e-4) -> torch.Tensor:
    """Per-sample min-max normalize a CAM to [0, 1].

    If a sample's CAM is degenerate (max-min below degenerate_threshold --
    e.g. a near-flat/non-discriminative feature map, which is exactly the
    diffuse-CAM failure mode this whole PuzzleCAM effort targets), dividing
    by a near-zero range would blow tiny floating-point noise up into a
    spurious near-arbitrary [0,1] map, injecting a large but meaningless
    gradient into re_loss. Those samples are zeroed out instead of divided,
    so a genuinely flat CAM contributes zero consistency signal rather than
    amplified noise.
    """
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
    """Single-label (CrossEntropy) adaptation of PuzzleCAM's full objective:
    L_re (CAM consistency) + L_p-cls (tile-classification loss), per Jo & Yu,
    ICIP 2021 -- L_cls itself (on the full image) is computed by the caller.

    Runs ONE extra forward pass on the 4x-batched tiles (not 4 separate
    passes). From the tiled features, two things are derived:
      - the merged/reconstructed CAM, compared against the full-image CAM
        (L_re) -- both restricted to each sample's target class, per this
        project's choice to scope L_re to the relevant class only rather
        than all 10 channels (mirrors the paper's "masking" variant).
      - the merged/reconstructed FEATURE MAP, pooled (GAP) and classified,
        then scored against target_class with CrossEntropy (L_p-cls). This
        is the term the initial implementation omitted: L_re alone only
        forces the tiled CAM to spatially agree with the full-image CAM,
        even in the degenerate case where both are wrong/uninformative in
        the same way (e.g. both uniformly diffuse) -- nothing requires the
        tiled view to still carry enough signal to classify correctly.
        L_p-cls directly supervises that: the model must still recognize
        the right tumor_type from each 2x2 reconstruction of quadrant
        features, which is what forces genuinely localized (not just
        mutually-consistent) evidence per the paper's ablations.

    Returns:
        full_cam:      [B, H, W] normalized CAM from the full image, target class only
        reconstructed: [B, H, W] normalized CAM merged from the 4 tiles, target class only
        re_loss:       scalar L1 loss between full_cam and reconstructed
        p_cls_loss:    scalar CrossEntropy loss on the merged tiled features
    """
    batch_size = images.shape[0]

    # CAM math (classic_cam's einsum) must stay outside any fp16 autocast
    # region -- forward_features() already forces fp32 through the backbone
    # for exactly this reason (a RadImageNet backbone was found to overflow
    # fp16 mid-forward-pass on certain inputs), and re-entering fp16 here for
    # the einsum would partially undo that protection for this code path.
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

        # L_p-cls: GAP the merged tiled feature map (same shape as a normal
        # feature map, [B, C, H, W]) and classify it, exactly as forward()
        # does for the full image -- but skip forward()'s dropout, matching
        # classic_cam's determinism choice (CAM/consistency signals should
        # not depend on dropout noise).
        merged_features = merge_2x2(tiled_features, batch_size)
        merged_pooled = model.avgpool(merged_features).flatten(1)
        merged_logits = model.classifier(merged_pooled)
        p_cls_loss = F.cross_entropy(merged_logits, target_class)

    return full_cam_norm, reconstructed_norm, re_loss, p_cls_loss
