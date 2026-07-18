from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_boundary_weight_map(
    targets: torch.Tensor,
    radius: int = 1,
    boundary_weight: float = 0.25,
) -> torch.Tensor:
    """Down-weight uncertain mask boundaries without deleting tiny lesions.

    A hard erosion-based ignore band can remove every positive pixel from a
    small BTXRD pseudo mask.  This map keeps every pixel supervised while
    assigning the dilation/erosion disagreement band a smaller weight.
    """
    if radius < 0:
        raise ValueError("boundary radius must be >= 0")
    if not 0.0 <= boundary_weight <= 1.0:
        raise ValueError("boundary_weight must be in [0, 1]")
    if radius == 0:
        return torch.ones_like(targets)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(targets, kernel_size=kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-targets, kernel_size=kernel, stride=1, padding=radius)
    boundary = dilated.ne(eroded)
    return torch.where(
        boundary,
        torch.full_like(targets, float(boundary_weight)),
        torch.ones_like(targets),
    )


def grouped_pseudo_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    tumor_status: torch.Tensor,
    *,
    pos_weight: torch.Tensor | float | None = None,
    pixel_weights: torch.Tensor | None = None,
    bce_weight: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Noise-aware loss for BTXRD pseudo masks.

    Tumor and normal images are averaged as two equally weighted groups.
    Dice is defined only for tumor images with a non-empty pseudo mask; an
    empty pseudo mask on a known tumor image is *unknown supervision*, not a
    trustworthy all-background label.  Normal images retain reliable
    all-background BCE supervision from the image-level ``tumor_type=0``
    label.  No segmentation ground truth is consumed here.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {logits.shape} vs {targets.shape}")
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError("bce_weight must be in [0, 1]")
    status = tumor_status.to(device=logits.device, dtype=torch.bool).flatten()
    if status.numel() != logits.shape[0]:
        raise ValueError("tumor_status must contain one value per image")
    weights = torch.ones_like(targets) if pixel_weights is None else pixel_weights.to(logits)
    if weights.shape != targets.shape:
        raise ValueError("pixel_weights must match targets")

    flat_targets = targets.flatten(start_dim=1)
    target_nonempty = flat_targets.gt(0.5).any(dim=1)
    reliable_tumor = status & target_nonempty
    reliable_normal = ~status
    supervised = reliable_tumor | reliable_normal

    pos_weight_tensor = (
        torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None else None
    )
    pixel_bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight_tensor, reduction="none"
    )
    per_image_bce = (
        (pixel_bce * weights).flatten(start_dim=1).sum(dim=1)
        / weights.flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    )
    group_bce: list[torch.Tensor] = []
    if reliable_tumor.any():
        group_bce.append(per_image_bce[reliable_tumor].mean())
    if reliable_normal.any():
        group_bce.append(per_image_bce[reliable_normal].mean())
    bce = torch.stack(group_bce).mean() if group_bce else logits.sum() * 0.0

    probs = torch.sigmoid(logits)
    intersection = (probs * targets * weights).flatten(start_dim=1).sum(dim=1)
    denominator = ((probs + targets) * weights).flatten(start_dim=1).sum(dim=1)
    soft_dice = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = (
        1.0 - soft_dice[reliable_tumor].mean()
        if reliable_tumor.any() else logits.sum() * 0.0
    )
    if reliable_tumor.any():
        loss = bce_weight * bce + (1.0 - bce_weight) * dice_loss
    else:
        # A normal-only batch must still learn background rather than having
        # its BCE halved by an absent tumor-Dice term.
        loss = bce
    diagnostics = {
        "reliable_tumor_images": float(reliable_tumor.sum().item()),
        "blank_tumor_images": float((status & ~target_nonempty).sum().item()),
        "normal_images": float(reliable_normal.sum().item()),
        "supervised_images": float(supervised.sum().item()),
    }
    return loss, diagnostics


def binary_segmentation_metric_sums(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    tumor_status: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Return additive, group-explicit pseudo-validation metric terms.

    Overlap is computed only on known-tumor images with non-empty pseudo
    targets. Empty normal references are evaluated separately as image-level
    specificity and false-positive pixel burden, avoiding an undefined/mixed
    Dice convention for empty masks.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    pred = probabilities.ge(float(threshold)).flatten(start_dim=1)
    target = targets.gt(0.5).flatten(start_dim=1)
    status = tumor_status.to(device=pred.device, dtype=torch.bool).flatten()
    target_nonempty = target.any(dim=1)
    tumor = status & target_nonempty
    blank_tumor = status & ~target_nonempty
    normal = ~status

    intersection = (pred & target).sum(dim=1).float()
    pred_area = pred.sum(dim=1).float()
    target_area = target.sum(dim=1).float()
    union = pred_area + target_area - intersection
    # These overlap terms are consumed only for non-empty tumor references.
    # A missing prediction therefore receives zero (including precision), not
    # the artificial value 1 that additive epsilon smoothing would produce.
    dice = 2.0 * intersection / (pred_area + target_area).clamp_min(eps)
    iou = intersection / union.clamp_min(eps)
    precision = intersection / pred_area.clamp_min(eps)
    recall = intersection / target_area.clamp_min(eps)
    normal_empty = ~pred.any(dim=1)
    pixels_per_image = float(pred.shape[1])

    return {
        "tumor_dice_sum": float(dice[tumor].sum().item()),
        "tumor_iou_sum": float(iou[tumor].sum().item()),
        "tumor_precision_sum": float(precision[tumor].sum().item()),
        "tumor_recall_sum": float(recall[tumor].sum().item()),
        "tumor_count": float(tumor.sum().item()),
        "blank_tumor_count": float(blank_tumor.sum().item()),
        "normal_empty_sum": float(normal_empty[normal].float().sum().item()),
        "normal_fp_pixels": float(pred_area[normal].sum().item()),
        "normal_pixels": float(normal.sum().item()) * pixels_per_image,
        "normal_count": float(normal.sum().item()),
    }


def finalize_binary_segmentation_metrics(sums: dict[str, float]) -> dict[str, float]:
    tumor_count = max(float(sums.get("tumor_count", 0.0)), 1.0)
    normal_count = max(float(sums.get("normal_count", 0.0)), 1.0)
    normal_pixels = max(float(sums.get("normal_pixels", 0.0)), 1.0)
    tumor_dice = float(sums.get("tumor_dice_sum", 0.0)) / tumor_count
    normal_specificity = float(sums.get("normal_empty_sum", 0.0)) / normal_count
    denominator = tumor_dice + normal_specificity
    hmean = 0.0 if denominator <= 0.0 else 2.0 * tumor_dice * normal_specificity / denominator
    return {
        "tumor_dice": tumor_dice,
        "tumor_iou": float(sums.get("tumor_iou_sum", 0.0)) / tumor_count,
        "tumor_precision": float(sums.get("tumor_precision_sum", 0.0)) / tumor_count,
        "tumor_recall": float(sums.get("tumor_recall_sum", 0.0)) / tumor_count,
        "tumor_images": float(sums.get("tumor_count", 0.0)),
        "blank_tumor_images": float(sums.get("blank_tumor_count", 0.0)),
        "normal_specificity": normal_specificity,
        "normal_fp_pixel_rate": float(sums.get("normal_fp_pixels", 0.0)) / normal_pixels,
        "normal_images": float(sums.get("normal_count", 0.0)),
        "hmean": hmean,
    }


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
    """Combine Dice loss with optionally foreground-weighted BCE.

    BTXRD tumor pixels occupy only a small fraction of an image. Plain BCE can
    therefore reward an all-background prediction before the model learns the
    lesion. ``pos_weight`` increases the contribution of positive pixels; the
    fully supervised trainer can estimate it as background/foreground pixels
    from the actual training masks.
    """
    pos_weight_tensor = (
        torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None
        else None
    )
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight_tensor,
    )
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
