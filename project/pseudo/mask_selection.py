from __future__ import annotations

"""CAM-guided SAM mask scoring and selection (pipeline.md Stage 5)."""

import numpy as np


def score_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
) -> np.ndarray:
    """Score each SAM mask by mean CAM activation inside the mask.

    score(mask) = mean(bone_cam[mask == 1])

    Args:
        masks:    [N, H, W] bool or uint8.
        bone_cam: [H, W] float32 in [0, 1].

    Returns:
        scores: [N] float32 array.
    """
    n = masks.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        m = masks[i].astype(bool)
        if m.any():
            scores[i] = float(bone_cam[m].mean())
    return scores


def select_and_fuse_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    mask_score_threshold: float = 0.4,
) -> np.ndarray:
    """Select masks whose CAM score exceeds threshold, then logical-OR them.

    If no mask passes the threshold, fall back to the single best-scoring mask
    to guarantee a non-empty pseudo mask.

    Args:
        masks:               [N, H, W] bool/uint8 from SAM.
        bone_cam:            [H, W] float32 in [0, 1].
        mask_score_threshold: Masks below this are discarded.

    Returns:
        pseudo_mask: [H, W] uint8 binary mask (0 / 1).
    """
    if masks.shape[0] == 0:
        h, w = bone_cam.shape
        return np.zeros((h, w), dtype=np.uint8)

    scores = score_masks(masks, bone_cam)
    selected = masks[scores >= mask_score_threshold]

    # fallback: keep best mask if nothing passes threshold
    if selected.shape[0] == 0:
        best_idx = int(np.argmax(scores))
        selected = masks[best_idx : best_idx + 1]

    # logical OR fusion
    fused = selected.any(axis=0).astype(np.uint8)
    return fused
