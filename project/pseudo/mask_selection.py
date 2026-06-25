from __future__ import annotations

"""CAM-guided SAM mask scoring and selection (pipeline.md Stage 5)."""

import numpy as np

# Supported scoring methods:
#   "mean"      : score = mean(cam inside mask)
#   "sum"       : score = sum(cam inside mask)                    — favors large masks
#   "mean_area" : score = mean(cam) * sqrt(area)                  — balanced size+quality
#   "coverage"  : score = fraction of mask pixels where cam > 0.5 — rewards full coverage
#   "hybrid"    : score = 0.7*mean(cam) + 0.3*log1p(area)/log1p(H*W) — mean + area bonus
SELECTION_METHODS = ("mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid")


def score_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    method: str = "mean",
    bone_likelihood: np.ndarray | None = None,
    bone_support: np.ndarray | None = None,
    sam_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Score each SAM mask by CAM activation inside the mask.

    Args:
        masks:    [N, H, W] bool or uint8.
        bone_cam: [H, W] float32 in [0, 1].
        method:   One of "mean", "sum", "mean_area", "coverage", "hybrid".

    Returns:
        scores: [N] float32 array.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection_method '{method}'. Choose from {SELECTION_METHODS}.")

    n = masks.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        cam_vals = bone_cam[m]
        area = float(m.sum())
        if method == "mean":
            scores[i] = float(cam_vals.mean())
        elif method == "sum":
            scores[i] = float(cam_vals.sum())
        elif method == "mean_area":
            scores[i] = float(cam_vals.mean()) * float(np.sqrt(area))
        elif method == "coverage":
            # fraction of mask pixels that are "activated" (cam > 0.5)
            scores[i] = float((cam_vals > 0.5).sum()) / area
        elif method == "hybrid":
            # mean CAM quality + log-normalised area bonus
            total_pixels = float(bone_cam.size)
            area_bonus = float(np.log1p(area) / np.log1p(total_pixels))
            scores[i] = 0.7 * float(cam_vals.mean()) + 0.3 * area_bonus
        elif method == "bone_hybrid":
            if bone_likelihood is None:
                scores[i] = float(cam_vals.mean())
                continue
            bone_mean = float(bone_likelihood[m].mean())
            cam_mean = float(cam_vals.mean())
            support_recall = 0.0
            support_precision = 0.0
            if bone_support is not None and bone_support.any():
                overlap = float((m & bone_support.astype(bool)).sum())
                support_recall = overlap / float(bone_support.sum())
                support_precision = overlap / area
            area_ratio = area / float(bone_cam.size)
            large_mask_penalty = max(0.0, area_ratio - 0.45)
            sam_quality = float(sam_scores[i]) if sam_scores is not None else 0.0
            scores[i] = (
                0.25 * bone_mean
                + 0.25 * cam_mean
                + 0.20 * support_recall
                + 0.20 * support_precision
                + 0.10 * sam_quality
                - 0.40 * large_mask_penalty
            )
    return scores


def select_and_fuse_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    mask_score_threshold: float = 0.4,
    selection_method: str = "mean",
    fusion_topk: int = 0,
    bone_likelihood: np.ndarray | None = None,
    bone_support: np.ndarray | None = None,
    sam_scores: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
    component_masks: np.ndarray | None = None,
    best_per_component: bool = False,
) -> np.ndarray:
    """Select and fuse masks using CAM and bone morphology evidence.

    fusion_topk controls how the top-scored masks are combined:
      0 or 1 : logical-OR of all above-threshold masks (original behaviour)
      k > 1  : union (logical-OR) of the top-k above-threshold masks
      k < 0  : intersection (logical-AND) of the top-|k| above-threshold masks

    With best_per_component enabled, the best SAM candidate from each complete
    morphology proposal is selected before union. Otherwise the original
    global top-k behavior is preserved for ablation.

    Args:
        masks:               [N, H, W] bool/uint8 from SAM.
        bone_cam:            [H, W] float32 in [0, 1].
        mask_score_threshold: Masks below this are discarded.
        selection_method:    "mean" | "sum" | "mean_area" (see SELECTION_METHODS).
        fusion_topk:         Fusion mode (0=default OR, k>1=top-k union, k<0=top-|k| intersection).

    Returns:
        pseudo_mask: [H, W] uint8 binary mask (0 / 1).
    """
    if masks.shape[0] == 0:
        h, w = bone_cam.shape
        return np.zeros((h, w), dtype=np.uint8)

    scores = score_masks(
        masks,
        bone_cam,
        method=selection_method,
        bone_likelihood=bone_likelihood,
        bone_support=bone_support,
        sam_scores=sam_scores,
    )

    if best_per_component and component_ids is not None and component_ids.size == masks.shape[0]:
        selected_indices: list[int] = []
        for component_id in np.unique(component_ids):
            candidates = np.where(component_ids == component_id)[0]
            if candidates.size == 0:
                continue
            component_scores = scores[candidates]
            if (
                component_masks is not None
                and 0 <= int(component_id) < component_masks.shape[0]
            ):
                component_scores = score_masks(
                    masks[candidates],
                    bone_cam,
                    method=selection_method,
                    bone_likelihood=bone_likelihood,
                    bone_support=component_masks[int(component_id)],
                    sam_scores=sam_scores[candidates] if sam_scores is not None else None,
                )
            best_index = int(candidates[np.argmax(component_scores)])
            selected_indices.append(best_index)
        if selected_indices:
            return masks[selected_indices].any(axis=0).astype(np.uint8)

    order = np.argsort(scores)[::-1]
    above = [i for i in order if scores[i] >= mask_score_threshold]

    # fallback: keep best mask if nothing passes threshold
    if not above:
        above = [int(order[0])]

    if fusion_topk == 1:
        # top-1 only — return the single best-scoring mask
        return masks[above[0]].copy().astype(np.uint8)
    elif fusion_topk == 0:
        # default: logical-OR of all above-threshold masks
        selected = masks[above]
        return selected.any(axis=0).astype(np.uint8)
    elif fusion_topk > 1:
        # union of top-k
        topk = above[:fusion_topk]
        fused = masks[topk[0]].copy().astype(bool)
        for i in topk[1:]:
            fused = fused | masks[i].astype(bool)
        return fused.astype(np.uint8)
    else:
        # fusion_topk < 0 → intersection of top-|k|
        k = abs(fusion_topk)
        topk = above[:k]
        fused = masks[topk[0]].copy().astype(bool)
        for i in topk[1:]:
            fused = fused & masks[i].astype(bool)
        return fused.astype(np.uint8)
