from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """pos_weight upweights the foreground (lesion) pixel's contribution to
    BCE, countering the collapse-to-all-background failure mode found
    empirically on BTXRD: lesions average only ~2.6% of image area, so with
    plain (unweighted) BCE, predicting "no lesion anywhere" already gets
    ~97.4% of pixels right and drives loss low before the model has learned
    anything -- observed directly as val_dice sitting frozen at exactly the
    dataset's normal-image fraction (0.505) for several epochs, matching
    Dice=1 on every empty-mask normal image and Dice~0 on every tumor image
    the model wrongly predicts as empty. A pos_weight around
    (background_pixels / foreground_pixels), estimated from the actual
    train-set masks, rebalances this so missing a lesion pixel costs as much
    as false-alarming on a background pixel.
    """
    pos_weight_tensor = (
        torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None else None
    )
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor)
    dice = dice_loss_from_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def weighted_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pixel_weight: torch.Tensor,
    bce_weight: float = 0.5,
    pos_weight: torch.Tensor | float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """bce_dice_loss variant where pixel_weight (same shape as logits/targets,
    values in [0, 1]) additionally scales each pixel's contribution to BOTH
    terms -- used by train_segmentation.py's --boundary-ignore-loss (weight=0
    on boundary-uncertain pixels, see pseudo/mask_selection.py's CONFIDENCE_*
    labels) and --confidence-weighted-loss (weight<1 on foreground-uncertain
    pixels). A weight of 0 everywhere pixel_weight is 0 is exact exclusion,
    not just down-weighting -- those pixels contribute nothing to either loss
    term, matching what "ignore this pixel" should mean.

    Unlike bce_dice_loss, this does not use
    F.binary_cross_entropy_with_logits' pos_weight-only reduction path: BCE is
    computed per-pixel (reduction="none") so pixel_weight can be applied
    before averaging, then Dice is computed on pixel_weight-masked
    probabilities/targets so weighted-out pixels contribute zero intersection
    AND zero denominator mass (not simply zero numerator, which would still
    let boundary pixels drag Dice's denominator around).
    """
    pos_weight_tensor = (
        torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None else None
    )
    per_pixel_bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight_tensor, reduction="none"
    )
    weighted_bce_sum = (per_pixel_bce * pixel_weight).sum()
    weight_sum = pixel_weight.sum().clamp(min=eps)
    bce = weighted_bce_sum / weight_sum

    probs = torch.sigmoid(logits) * pixel_weight
    weighted_targets = targets * pixel_weight
    probs_flat = probs.flatten(start_dim=1)
    targets_flat = weighted_targets.flatten(start_dim=1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = 1.0 - dice.mean()

    return bce_weight * bce + (1.0 - bce_weight) * dice_loss


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    preds = preds.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    preds = preds.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()
