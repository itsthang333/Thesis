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
    bone_support: np.ndarray | None = None,
    selection_method: str = "bone_hybrid",
    support_clip_kernel: int = 5,
) -> dict[str, float]:
    """Compare the best-possible candidate mask against the one actually selected.

    select_and_fuse_masks' bone_hybrid path always intersects its fused mask
    with bone_support (constrain_to_bone_support) before returning, but
    selected_mask is the ONLY one of the two masks compared here that goes
    through that clip -- the raw sam_masks used for the oracle never do. A
    large oracle_gap_dice can therefore come from two unrelated places that
    were previously conflated into one number:
      - selection loss:  mask_selection.py's bone_hybrid scoring picked a
                          worse candidate than the best one available.
      - support loss:    bone_support (from the pre-SAM morphology stage)
                          under-covers the true lesion, so even the best
                          candidate gets clipped down to near-nothing by
                          constrain_to_bone_support -- independent of which
                          candidate mask_selection.py picked.
    Passing bone_support (and the selection_method/support_clip_kernel used
    for the real run) lets this function clip each raw candidate the same
    way select_and_fuse_masks does, giving an oracle that is directly
    comparable to selected_mask and splitting the total loss into the two
    pieces above.

    Args:
        sam_masks:     [N, H, W] bool/uint8 — every raw candidate SAM proposed
                       for this image, before select_and_fuse_masks runs.
        selected_mask: [H, W] — the mask select_and_fuse_masks actually chose
                       (before Stage 6 morphological refinement, so this
                       isolates mask_selection.py specifically). Already
                       clipped to bone_support internally.
        gt_mask:       [H, W] ground-truth binary mask.
        bone_support:  [H, W] bool/uint8 support mask from the morphology
                       stage, or None to skip the clipped oracle (best_single_
                       dice_clipped/support_loss_dice/selection_loss_dice will
                       be NaN, matching pre-decomposition behavior).
        selection_method, support_clip_kernel: passed through to
            pseudo.mask_selection.constrain_to_bone_support so the clip
            matches exactly what the real run applied.

    Returns:
        best_single_dice/iou: best Dice/IoU from any single RAW (unclipped)
            candidate -- measures how good SAM's candidates are before any
            support constraint, independent of mask_selection.py or
            bone_support quality.
        best_single_dice_clipped: best Dice from any candidate after being
            clipped to bone_support -- the true ceiling on what
            mask_selection.py could have produced from this exact candidate
            set once the support clip is applied, directly comparable to
            selected_dice. NaN if bone_support is None.
        selected_dice/iou: Dice/IoU of the mask mask_selection.py picked
            (already clipped, since select_and_fuse_masks clips internally).
        gap_dice: best_single_dice - selected_dice. The OLD, undecomposed
            total gap -- kept for backward compatibility with existing CSVs.
        support_loss_dice: best_single_dice - best_single_dice_clipped. Dice
            lost purely from clipping the best candidate to bone_support,
            with mask_selection.py's scoring not involved at all. Large value
            => bone_support under-covers the lesion; fix the morphology
            stage (seed/support percentiles), not mask_selection.py.
        selection_loss_dice: best_single_dice_clipped - selected_dice. Dice
            lost specifically because mask_selection.py picked a worse
            candidate than the best clipped one available. Large value =>
            fix bone_hybrid scoring/thresholds.
    """
    gt_bool = gt_mask.astype(bool)
    if not gt_bool.any() or sam_masks.shape[0] == 0:
        return {
            "best_single_dice": float("nan"),
            "best_single_iou": float("nan"),
            "best_single_dice_clipped": float("nan"),
            "selected_dice": float("nan"),
            "selected_iou": float("nan"),
            "gap_dice": float("nan"),
            "support_loss_dice": float("nan"),
            "selection_loss_dice": float("nan"),
        }

    per_candidate_dice = np.array([_dice(sam_masks[i], gt_bool) for i in range(sam_masks.shape[0])])
    best_index = int(np.argmax(per_candidate_dice))
    best_single_dice = float(per_candidate_dice[best_index])
    best_single_iou = _iou(sam_masks[best_index], gt_bool)

    selected_dice = _dice(selected_mask, gt_bool)
    selected_iou = _iou(selected_mask, gt_bool)

    best_single_dice_clipped = float("nan")
    support_loss_dice = float("nan")
    selection_loss_dice = float("nan")
    if bone_support is not None:
        from .mask_selection import constrain_to_bone_support

        per_candidate_dice_clipped = np.array([
            _dice(
                constrain_to_bone_support(sam_masks[i], bone_support, selection_method, support_clip_kernel),
                gt_bool,
            )
            for i in range(sam_masks.shape[0])
        ])
        best_single_dice_clipped = float(per_candidate_dice_clipped.max())
        support_loss_dice = best_single_dice - best_single_dice_clipped
        selection_loss_dice = best_single_dice_clipped - selected_dice

    return {
        "best_single_dice": best_single_dice,
        "best_single_iou": best_single_iou,
        "best_single_dice_clipped": best_single_dice_clipped,
        "selected_dice": selected_dice,
        "selected_iou": selected_iou,
        "gap_dice": best_single_dice - selected_dice,
        "support_loss_dice": support_loss_dice,
        "selection_loss_dice": selection_loss_dice,
    }
