from __future__ import annotations

"""Diagnostics for isolating SAM-candidate quality from mask-selection quality.

If a pseudo mask's Dice against ground truth is low, that alone doesn't say
*why* — it could be that SAM never proposed a good candidate for this lesion,
or that a good candidate existed but select_and_fuse_masks (mask_selection.py)
picked a worse one. This module answers that by comparing, against the real
GT mask (never used elsewhere in the WSSS pipeline — this is diagnostics
only, not something the pipeline is allowed to see):

  - oracle_dice/iou:   best possible Dice from the raw SAM candidates, using
                       the best single candidate together with the best
                       above-threshold union — an upper bound on what
                       mask_selection.py could have produced from this exact
                       candidate set.
  - selected_dice/iou: Dice of the mask select_and_fuse_masks actually chose
                       (pre-morphology).

A large oracle-vs-selected gap means mask_selection.py is discarding a good
candidate that was already there (fix scoring/thresholds). A small gap with
a low oracle score means SAM itself never produced a usable candidate for
this lesion (fix prompts/support, not selection).
"""

import numpy as np


def _dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = float((pred & target).sum())
    denom = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (denom + eps)


def _iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = float((pred & target).sum())
    union = float((pred | target).sum())
    return (intersection + eps) / (union + eps)


def oracle_vs_selected_metrics(
    sam_masks: np.ndarray,
    selected_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, float]:
    """Compare the best-possible candidate mask against the one actually selected.

    Args:
        sam_masks:     [N, H, W] bool/uint8 — every raw candidate SAM proposed
                       for this image, before select_and_fuse_masks runs.
        selected_mask: [H, W] — the mask select_and_fuse_masks actually chose
                       (before Stage 6 morphological refinement, so this
                       isolates mask_selection.py specifically).
        gt_mask:       [H, W] ground-truth binary mask.

    Returns:
        best_single_dice/iou: best Dice/IoU from any single raw candidate —
            the ceiling on what mask_selection.py could have produced by
            picking one candidate outright from this exact set (it also
            supports unioning several, so this is a lower bound on its true
            ceiling, but is by far the dominant term for a single lesion).
        selected_dice/iou: Dice/IoU of the mask mask_selection.py picked.
        gap_dice: best_single_dice - selected_dice. Large gap => selection
            problem. Small gap + low best_single => SAM/prompt problem.
    """
    gt_bool = gt_mask.astype(bool)
    if not gt_bool.any() or sam_masks.shape[0] == 0:
        return {
            "best_single_dice": float("nan"),
            "best_single_iou": float("nan"),
            "selected_dice": float("nan"),
            "selected_iou": float("nan"),
            "gap_dice": float("nan"),
        }

    per_candidate_dice = np.array([_dice(sam_masks[i], gt_bool) for i in range(sam_masks.shape[0])])
    best_index = int(np.argmax(per_candidate_dice))
    best_single_dice = float(per_candidate_dice[best_index])
    best_single_iou = _iou(sam_masks[best_index], gt_bool)

    selected_dice = _dice(selected_mask, gt_bool)
    selected_iou = _iou(selected_mask, gt_bool)

    return {
        "best_single_dice": best_single_dice,
        "best_single_iou": best_single_iou,
        "selected_dice": selected_dice,
        "selected_iou": selected_iou,
        "gap_dice": best_single_dice - selected_dice,
    }
