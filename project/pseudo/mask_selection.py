from __future__ import annotations

"""CAM-guided SAM mask scoring and selection (pipeline.md Stage 5)."""

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

# Supported scoring methods:
#   "mean"      : score = mean(cam inside mask)
#   "sum"       : score = sum(cam inside mask)                    — favors large masks
#   "mean_area" : score = mean(cam) * sqrt(area)                  — balanced size+quality
#   "coverage"  : score = fraction of mask pixels where cam > 0.5 — rewards full coverage
#   "hybrid"    : score = 0.7*mean(cam) + 0.3*log1p(area)/log1p(H*W) — mean + area bonus
SELECTION_METHODS = ("mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid")


def _binary_dilation(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(np.uint8)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    pad = kernel_size // 2
    padded = np.pad(mask.astype(bool), pad, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            output |= padded[
                row_offset : row_offset + mask.shape[0],
                col_offset : col_offset + mask.shape[1],
            ]
    return output.astype(np.uint8)


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
            # bone_support is derived from a CAM percentile cut in the pre-SAM
            # morphology stage, not ground truth -- it typically UNDER-covers
            # the true lesion (support subset-of lesion), not the other way
            # around. support_recall (does the candidate contain the support
            # region?) is therefore a meaningful bonus: a good candidate
            # should contain the seed region SAM was prompted from. But the
            # inverse -- support_precision / outside_support_ratio, "does the
            # candidate stay INSIDE the support region?" -- assumes the
            # opposite (lesion subset-of support) and heavily penalizes any
            # candidate that correctly extends beyond a too-narrow support to
            # cover the rest of the real lesion. That assumption was verified
            # wrong in practice (support_loss_dice ~ 0 while selection_loss_dice
            # was the dominant term in this project's own oracle diagnostic),
            # so precision-vs-support is intentionally dropped here in favor
            # of weighting CAM/bone_likelihood more heavily as the primary
            # "is this actually the lesion" signal.
            support_recall = 0.0
            if bone_support is not None and bone_support.any():
                support_bool = bone_support.astype(bool)
                overlap = float((m & support_bool).sum())
                support_recall = overlap / float(bone_support.sum())
            area_ratio = area / float(bone_cam.size)
            support_area_ratio = (
                float(bone_support.sum()) / float(bone_cam.size)
                if bone_support is not None and bone_support.any()
                else 0.0
            )
            # expected_area is a soft ceiling on plausible lesion size, not a
            # hard support-shape constraint -- kept mild (lower weight below)
            # so it only discourages implausibly large candidates (e.g. the
            # whole hand) rather than penalizing any candidate larger than a
            # narrow support region.
            expected_area = max(0.08, min(0.35, support_area_ratio * 2.0 + 0.05))
            large_mask_penalty = max(0.0, area_ratio - expected_area)
            border_touch_count = int(m[0, :].any()) + int(m[-1, :].any()) + int(m[:, 0].any()) + int(m[:, -1].any())
            border_touch_penalty = border_touch_count / 4.0
            sam_quality = float(sam_scores[i]) if sam_scores is not None else 0.0
            scores[i] = (
                0.45 * bone_mean
                + 0.30 * cam_mean
                + 0.15 * support_recall
                + 0.10 * sam_quality
                - 0.40 * large_mask_penalty
                - 0.20 * border_touch_penalty
            )
    return scores


def constrain_to_bone_support(
    fused_mask: np.ndarray,
    bone_support: np.ndarray | None,
    selection_method: str = "bone_hybrid",
    support_clip_kernel: int = 5,
) -> np.ndarray:
    """Intersect a candidate/fused mask with the (optionally dilated) bone/tumor
    support region, falling back to the unclipped mask if the intersection is
    empty. Shared by select_and_fuse_masks (applied to its final fused mask)
    and oracle_diagnostics (applied per-candidate, to measure how much Dice
    the clip step itself costs, independent of mask-selection scoring).
    """
    fused_mask = fused_mask.astype(np.uint8)
    if (
        selection_method != "bone_hybrid"
        or bone_support is None
        or not bone_support.any()
        or support_clip_kernel < 0
    ):
        return fused_mask
    support_constraint = (
        bone_support.astype(np.uint8)
        if support_clip_kernel <= 1
        else _binary_dilation(bone_support, kernel_size=support_clip_kernel)
    )
    clipped = fused_mask & support_constraint
    return clipped.astype(np.uint8) if clipped.any() else fused_mask


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
    support_clip_kernel: int = 5,
) -> np.ndarray:
    """Select and fuse masks using CAM and bone morphology evidence.

    fusion_topk controls how the top-scored masks are combined:
      0 or 1 : logical-OR of all above-threshold masks (original behaviour)
      k > 1  : union (logical-OR) of the top-k above-threshold masks
      k < 0  : intersection (logical-AND) of the top-|k| above-threshold masks

    With best_per_component enabled, the best SAM candidate from each complete
    morphology proposal is selected before union. Otherwise the original
    global top-k behavior is preserved for ablation.

    component_masks is accepted for call-site backward compatibility but is
    unused: bone_hybrid's support_area_ratio/expected_area/large_mask_penalty
    terms are only meaningful relative to the whole-image bone_support map, so
    per-component re-scoring must not substitute a single component's mask
    for it.

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

    def _clip(fused_mask: np.ndarray) -> np.ndarray:
        return constrain_to_bone_support(fused_mask, bone_support, selection_method, support_clip_kernel)

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
            # Reuse the globally-scored candidates for this component. score_masks
            # already received the global bone_support above — bone_hybrid's
            # support_area_ratio/expected_area/large_mask_penalty terms are only
            # meaningful relative to the whole-image support map, so re-scoring
            # with a single component's mask here would corrupt those ratios.
            component_scores = scores[candidates]
            best_local = int(np.argmax(component_scores))
            best_index = int(candidates[best_local])
            if float(component_scores[best_local]) >= mask_score_threshold:
                selected_indices.append(best_index)
        if selected_indices:
            return _clip(masks[selected_indices].any(axis=0))

    order = np.argsort(scores)[::-1]
    above = [i for i in order if scores[i] >= mask_score_threshold]

    # fallback: keep best mask if nothing passes threshold
    if not above:
        above = [int(order[0])]

    if fusion_topk == 1:
        # top-1 only — return the single best-scoring mask
        fused = masks[above[0]].copy().astype(np.uint8)
    elif fusion_topk == 0:
        # default: logical-OR of all above-threshold masks
        selected = masks[above]
        fused = selected.any(axis=0).astype(np.uint8)
    elif fusion_topk > 1:
        # union of top-k
        topk = above[:fusion_topk]
        fused = masks[topk[0]].copy().astype(bool)
        for i in topk[1:]:
            fused = fused | masks[i].astype(bool)
        fused = fused.astype(np.uint8)
    else:
        # fusion_topk < 0 → intersection of top-|k|
        k = abs(fusion_topk)
        topk = above[:k]
        fused = masks[topk[0]].copy().astype(bool)
        for i in topk[1:]:
            fused = fused & masks[i].astype(bool)
        fused = fused.astype(np.uint8)
    return _clip(fused)
