from __future__ import annotations

"""CAM-guided SAM mask scoring and selection (pipeline.md Stage 5)."""

import csv
from pathlib import Path

import numpy as np

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


def _elongation_score(mask: np.ndarray) -> float:
    """Normalized elongation score in [0, 1].

    Bone masks are typically elongated; soft-tissue masks tend to be compact.
    Computed from bounding box aspect ratio: ratio = max(w,h) / min(w,h).
    Mapped to [0,1] with a soft cap at ratio=10 (very elongated → 1.0).
    Empty masks return 0.
    """
    rows, cols = np.where(mask)
    if rows.size == 0:
        return 0.0
    h = int(rows.max() - rows.min()) + 1
    w = int(cols.max() - cols.min()) + 1
    if min(h, w) == 0:
        return 0.0
    ratio = float(max(h, w)) / float(min(h, w))
    return float(min(1.0, (ratio - 1.0) / 9.0))


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
        method:   One of "mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid".

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
            scores[i] = float((cam_vals > 0.5).sum()) / area
        elif method == "hybrid":
            total_pixels = float(bone_cam.size)
            area_bonus = float(np.log1p(area) / np.log1p(total_pixels))
            scores[i] = 0.7 * float(cam_vals.mean()) + 0.3 * area_bonus
        elif method == "bone_hybrid":
            bone_mean = float(bone_likelihood[m].mean()) if bone_likelihood is not None else 0.0
            cam_mean = float(cam_vals.mean())
            support_recall = 0.0
            support_precision = 0.0
            if bone_support is not None and bone_support.any():
                overlap = float((m & bone_support.astype(bool)).sum())
                support_recall = overlap / float(bone_support.sum())
                support_precision = overlap / area
            area_ratio = area / float(bone_cam.size)
            support_area_ratio = (
                float(bone_support.sum()) / float(bone_cam.size)
                if bone_support is not None and bone_support.any()
                else 0.0
            )
            expected_area = max(0.08, min(0.35, support_area_ratio * 2.6 + 0.03))
            large_mask_penalty = max(0.0, area_ratio - expected_area)
            soft_tissue_penalty = max(0.0, 0.55 - support_precision)
            sam_quality = float(sam_scores[i]) if sam_scores is not None else 0.0
            elongation = _elongation_score(m)
            scores[i] = (
                0.20 * bone_mean
                + 0.15 * cam_mean
                + 0.20 * support_precision
                + 0.20 * support_recall
                + 0.15 * sam_quality
                + 0.10 * elongation
                - 0.90 * large_mask_penalty
                - 0.35 * soft_tissue_penalty
            )
    return scores


def score_masks_detailed(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    method: str = "bone_hybrid",
    bone_likelihood: np.ndarray | None = None,
    bone_support: np.ndarray | None = None,
    sam_scores: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
) -> list[dict]:
    """Return per-candidate scoring breakdown for ranking debug CSV (Section 10)."""
    n = masks.shape[0]
    rows = []
    for i in range(n):
        m = masks[i].astype(bool)
        area = int(m.sum())
        cam_vals = bone_cam[m] if area > 0 else np.array([0.0])
        bone_mean = float(bone_likelihood[m].mean()) if bone_likelihood is not None and area > 0 else 0.0
        cam_mean = float(cam_vals.mean())
        support_recall = 0.0
        support_precision = 0.0
        if bone_support is not None and bone_support.any() and area > 0:
            overlap = float((m & bone_support.astype(bool)).sum())
            support_recall = overlap / float(bone_support.sum())
            support_precision = overlap / float(area)
        sam_quality = float(sam_scores[i]) if sam_scores is not None else 0.0
        elongation = _elongation_score(m)
        rows.append({
            "candidate_id": i,
            "component_id": int(component_ids[i]) if component_ids is not None else -1,
            "area": area,
            "bone_mean": round(bone_mean, 4),
            "cam_mean": round(cam_mean, 4),
            "support_precision": round(support_precision, 4),
            "support_recall": round(support_recall, 4),
            "sam_score": round(sam_quality, 4),
            "elongation_score": round(elongation, 4),
            "final_score": 0.0,  # filled below
        })

    final_scores = score_masks(masks, bone_cam, method=method,
                               bone_likelihood=bone_likelihood,
                               bone_support=bone_support, sam_scores=sam_scores)
    for i, row in enumerate(rows):
        row["final_score"] = round(float(final_scores[i]), 4)
    return rows


def save_ranking_csv(rows: list[dict], path: Path) -> None:
    """Save per-candidate ranking details to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id", "component_id", "area",
        "bone_mean", "cam_mean", "support_precision", "support_recall",
        "sam_score", "elongation_score", "final_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    def constrain_to_bone_support(fused_mask: np.ndarray) -> np.ndarray:
        fused_mask = fused_mask.astype(np.uint8)
        if selection_method != "bone_hybrid" or bone_support is None or not bone_support.any():
            return fused_mask
        clipped = fused_mask & _binary_dilation(bone_support, kernel_size=11)
        return clipped.astype(np.uint8) if clipped.any() else fused_mask

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
            best_local = int(np.argmax(component_scores))
            best_index = int(candidates[best_local])
            if float(component_scores[best_local]) >= mask_score_threshold:
                selected_indices.append(best_index)
        if selected_indices:
            return constrain_to_bone_support(masks[selected_indices].any(axis=0))

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
    return constrain_to_bone_support(fused)
