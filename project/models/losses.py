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
