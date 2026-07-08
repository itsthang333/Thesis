from __future__ import annotations

"""Metrics for judging LayerCAM/prompt quality *before* SAM runs.

evaluate_ramh1200_masks.py only measures the final pseudo mask, which has
already passed through morphology + SAM + mask selection — a low Dice there
does not say whether the failure came from a bad CAM, a badly placed prompt
point, or a bad SAM/mask-selection choice. These metrics isolate the CAM and
point-prompt stage so that failure can be attributed correctly.
"""

import numpy as np


def binary_mask_localization_metrics(
    foreground_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, float]:
    """IoU/recall/precision between an already-binary foreground mask and GT.

    Use this when the pipeline already computed a concrete binary support/
    foreground mask (e.g. bone_support/tumor_support from
    build_class_conditioned_components, which involves seed+support
    thresholds, morphological reconstruction, and CAM-component filtering —
    not a single percentile cut). Recomputing a percentile threshold on the
    fused CAM in that case would not reflect the mask SAM actually receives.

    Returns:
        iou:       IoU between foreground_mask and gt_mask.
        recall:    fraction of GT mask pixels covered by foreground_mask
                   (low value means the foreground is missing part of the lesion).
        precision: fraction of foreground_mask that falls inside GT
                   (low value means the foreground extends into irrelevant regions).
    """
    gt_bool = gt_mask.astype(bool)
    fg_bool = foreground_mask.astype(bool)
    if not gt_bool.any():
        return {"iou": float("nan"), "recall": float("nan"), "precision": float("nan")}

    intersection = float((fg_bool & gt_bool).sum())
    union = float((fg_bool | gt_bool).sum())
    iou = intersection / union if union > 0 else 0.0
    recall = intersection / float(gt_bool.sum())
    precision = intersection / float(fg_bool.sum()) if fg_bool.any() else 0.0

    return {"iou": iou, "recall": recall, "precision": precision}


def cam_localization_metrics(
    cam: np.ndarray,
    gt_mask: np.ndarray,
    percentile: float = 85.0,
) -> dict[str, float]:
    """How well does {cam >= percentile} line up with GT?

    Only meaningful when the pipeline's actual foreground is a plain percentile
    cut on this CAM/prompt_map — i.e. the --disable-bone-morphology or
    morphology-fusion-mode=weighted path, which calls extract_point_prompts
    with this same percentile. In morphology-fusion-mode=components (the
    default), use binary_mask_localization_metrics on bone_support/
    tumor_support instead, since the real foreground there is not a single
    percentile cut on this array.

    Returns:
        cam_iou:       IoU between {cam >= percentile} and gt_mask.
        cam_recall:    fraction of GT mask pixels covered by the CAM foreground
                       (low value means the CAM is missing part of the lesion).
        cam_precision: fraction of the CAM foreground that falls inside GT
                       (low value means the CAM is activating on irrelevant regions).
    """
    gt_bool = gt_mask.astype(bool)
    if not gt_bool.any():
        return {"cam_iou": float("nan"), "cam_recall": float("nan"), "cam_precision": float("nan")}

    threshold = float(np.percentile(cam, percentile))
    cam_fg = cam > threshold

    intersection = float((cam_fg & gt_bool).sum())
    union = float((cam_fg | gt_bool).sum())
    cam_iou = intersection / union if union > 0 else 0.0
    cam_recall = intersection / float(gt_bool.sum())
    cam_precision = intersection / float(cam_fg.sum()) if cam_fg.any() else 0.0

    return {"cam_iou": cam_iou, "cam_recall": cam_recall, "cam_precision": cam_precision}


def point_prompt_hit_rate(
    points: list[tuple[int, int]],
    gt_mask: np.ndarray,
) -> dict[str, float]:
    """Fraction of (row, col) prompt points that land inside the GT mask.

    A low hit rate means SAM is being pointed at soft tissue / background
    instead of the lesion, regardless of how good the overall CAM heatmap looks.
    """
    gt_bool = gt_mask.astype(bool)
    if not points:
        return {"point_hit_rate": float("nan"), "num_points": 0, "num_hits": 0}
    if not gt_bool.any():
        return {"point_hit_rate": float("nan"), "num_points": len(points), "num_hits": 0}

    height, width = gt_bool.shape
    hits = 0
    for row, col in points:
        if 0 <= row < height and 0 <= col < width and gt_bool[row, col]:
            hits += 1
    return {
        "point_hit_rate": hits / len(points),
        "num_points": len(points),
        "num_hits": hits,
    }
