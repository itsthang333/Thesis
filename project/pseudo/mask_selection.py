from __future__ import annotations

"""CAM-guided SAM mask scoring and selection (pipeline.md Stage 5)."""

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

# Supported scoring methods:
#   "mean"        : score = mean(cam inside mask)
#   "sum"         : score = sum(cam inside mask)                    — favors large masks
#   "mean_area"   : score = mean(cam) * sqrt(area)                  — balanced size+quality
#   "coverage"    : score = fraction of mask pixels where cam > 0.5 — rewards full coverage
#   "hybrid"      : score = 0.7*mean(cam) + 0.3*log1p(area)/log1p(H*W) — mean + area bonus
#   "simple_hybrid": score = 0.6*mean(cam) + 0.3*sam_quality + 0.1*log1p(area)/log1p(H*W)
#                   — a deliberately minimal alternative to bone_hybrid's ~10-term
#                   formula, added after this project's own experiments repeatedly
#                   retuning bone_hybrid's support-based terms (support_precision,
#                   outside_support_ratio, soft_tissue_penalty) produced only
#                   marginal Dice changes -- worth comparing against a much
#                   simpler CAM+SAM-confidence+area score before assuming more
#                   terms are needed.
#   "prompt_hybrid": combines CAM-mass coverage, CAM density, within-prompt
#                    SAM-quality rank, a support-relative area prior, and
#                    positive/negative prompt consistency. Unlike
#                    simple_hybrid, it does not treat raw SAM predicted IoU
#                    as calibrated across images and does not let mean CAM
#                    alone reward an arbitrarily tiny mask around the peak.
SELECTION_METHODS = (
    "mean", "sum", "mean_area", "coverage", "coverage_mass", "coverage_mass_sam", "hybrid", "bone_hybrid",
    "simple_hybrid", "prompt_hybrid", "consistency_hybrid",
)

DEFAULT_PROMPT_HYBRID_WEIGHTS = (0.30, 0.20, 0.15, 0.15, 0.20)


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


def _within_group_percentile_ranks(
    values: np.ndarray | None,
    group_ids: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Convert SAM scores to ordinal ranks within each prompt/component.

    SAM's predicted IoU is useful for ordering candidates from the same
    prompt, but it is not assumed to be calibrated on X-rays or comparable
    across images. Percentile ranks preserve only that within-prompt order.
    """
    ranks = np.zeros(size, dtype=np.float32)
    if values is None or len(values) != size:
        return ranks
    values = np.asarray(values, dtype=np.float32)
    groups = (
        np.asarray(group_ids, dtype=np.int32)
        if group_ids is not None and len(group_ids) == size
        else np.zeros(size, dtype=np.int32)
    )
    for group_id in np.unique(groups):
        indices = np.where(groups == group_id)[0]
        if indices.size == 1:
            ranks[indices[0]] = 1.0
            continue
        group_values = values[indices]
        denominator = float(indices.size - 1)
        for local_index, value in enumerate(group_values):
            lower = float((group_values < value).sum())
            equal = float((group_values == value).sum())
            ranks[indices[local_index]] = (lower + 0.5 * (equal - 1.0)) / denominator
    return ranks


def _point_consistency(
    mask: np.ndarray,
    positive_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    negative_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> float:
    """Fraction of positive prompts included and negative prompts excluded."""
    h, w = mask.shape

    def _inside(point: tuple[int, int]) -> bool:
        row, col = point
        return 0 <= row < h and 0 <= col < w and bool(mask[row, col])

    terms: list[float] = []
    if positive_points:
        terms.append(sum(_inside(point) for point in positive_points) / len(positive_points))
    if negative_points:
        terms.append(sum(not _inside(point) for point in negative_points) / len(negative_points))
    return float(sum(terms) / len(terms)) if terms else 0.0


def score_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    method: str = "mean",
    bone_likelihood: np.ndarray | None = None,
    bone_support: np.ndarray | None = None,
    sam_scores: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
    component_masks: np.ndarray | None = None,
    positive_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    negative_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    prompt_hybrid_weights: tuple[float, float, float, float, float] = DEFAULT_PROMPT_HYBRID_WEIGHTS,
    prompt_area_target: float = 2.0,
    prompt_area_log_sigma: float = 1.0,
    anatomy_cam: np.ndarray | None = None,
    anatomy_consistency_weight: float = 0.0,
) -> np.ndarray:
    """Score each SAM mask by CAM activation inside the mask.

    Args:
        masks:    [N, H, W] bool or uint8.
        bone_cam: [H, W] float32 in [0, 1]. The global (or already-selected-
                  aggregation) CAM every scoring method above is computed
                  from.
        method:   One of "mean", "sum", "mean_area", "coverage", "hybrid".
        anatomy_cam: Optional [H, W] float32 in [0, 1], the region-conditioned
                  CAM from generate_pseudo_masks.py's --cam-anatomy-conditioned
                  (models/layercam.py's cam_for_anatomy_conditioned_score).
                  When given (together with anatomy_consistency_weight > 0),
                  each candidate's score above is nudged by how much its mean
                  CAM activation agrees between bone_cam and anatomy_cam --
                  this rewards masks that BOTH the global tumor-type evidence
                  AND the region-conditioned "distinguishable from a same-
                  region normal" evidence support, without ever letting
                  anatomy_cam alone drive the score (a candidate that only
                  anatomy_cam likes, but bone_cam does not, gets no bonus).
                  None (default, matching every other call site unaware of
                  anatomy-conditioned CAM) disables this entirely -- scores
                  are then identical to before this parameter existed.
        anatomy_consistency_weight: How much of each candidate's own
                  bone_cam-only score to hold back and replace with the
                  agreement-based term above (0 disables it even if
                  anatomy_cam is given; typical range 0.1-0.3 so the primary
                  scoring method above still dominates).

    Returns:
        scores: [N] float32 array.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection_method '{method}'. Choose from {SELECTION_METHODS}.")

    n = masks.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    sam_ranks = _within_group_percentile_ranks(sam_scores, component_ids, n)

    component_mask_by_id: dict[int, np.ndarray] = {}
    if component_masks is not None and len(component_masks) > 0:
        if component_ids is not None and len(component_ids) == n:
            unique_ids = list(np.unique(component_ids))
            component_mask_by_id = {
                int(component_id): component_masks[position].astype(bool)
                for position, component_id in enumerate(unique_ids)
                if position < len(component_masks)
            }
        elif len(component_masks) == 1:
            component_mask_by_id[0] = component_masks[0].astype(bool)

    weights = np.asarray(prompt_hybrid_weights, dtype=np.float32)
    if weights.shape != (5,) or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError(
            "prompt_hybrid_weights must contain five non-negative values with a positive sum "
            "(cam_coverage, cam_density, sam_rank, area, prompt_consistency)."
        )
    weights = weights / weights.sum()
    area_target = max(float(prompt_area_target), 1e-6)
    area_sigma = max(float(prompt_area_log_sigma), 1e-6)

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
        elif method == "coverage_mass":
            # Retain the interpretable binary coverage term but break its
            # frequent 1.0 ties with the fraction of total CAM mass captured.
            # This remains image-only and is less biased toward tiny masks
            # than raw meanCAM.
            binary_coverage = float((cam_vals > 0.5).sum()) / area
            mass_coverage = float(cam_vals.sum()) / max(float(bone_cam.sum()), 1e-8)
            scores[i] = 0.70 * binary_coverage + 0.30 * mass_coverage
        elif method == "coverage_mass_sam":
            binary_coverage = float((cam_vals > 0.5).sum()) / area
            mass_coverage = float(cam_vals.sum()) / max(float(bone_cam.sum()), 1e-8)
            # SAM score is used only as a within-component rank, never as a
            # calibrated cross-image IoU probability.
            scores[i] = (
                0.60 * binary_coverage
                + 0.25 * mass_coverage
                + 0.15 * float(sam_ranks[i])
            )
        elif method == "hybrid":
            # mean CAM quality + log-normalised area bonus
            total_pixels = float(bone_cam.size)
            area_bonus = float(np.log1p(area) / np.log1p(total_pixels))
            scores[i] = 0.7 * float(cam_vals.mean()) + 0.3 * area_bonus
        elif method == "simple_hybrid":
            total_pixels = float(bone_cam.size)
            area_bonus = float(np.log1p(area) / np.log1p(total_pixels))
            sam_quality = float(sam_scores[i]) if sam_scores is not None else 0.0
            scores[i] = 0.6 * float(cam_vals.mean()) + 0.3 * sam_quality + 0.1 * area_bonus
        elif method in {"prompt_hybrid", "consistency_hybrid"}:
            component_id = (
                int(component_ids[i])
                if component_ids is not None and len(component_ids) == n
                else 0
            )
            support = component_mask_by_id.get(component_id)
            if support is None:
                support = bone_support.astype(bool) if bone_support is not None else None

            # Coverage measures how much of the prompt component's CAM mass
            # the candidate captures. Density measures how concentrated CAM
            # evidence remains inside the candidate. Their combination avoids
            # both extremes: tiny peak-only masks and huge high-recall masks.
            if support is not None and support.any():
                support_mass = float(bone_cam[support].sum())
                captured_mass = float(bone_cam[m & support].sum())
                cam_coverage = captured_mass / max(support_mass, 1e-8)
                reference_area = float(support.sum())
            else:
                total_mass = float(bone_cam.sum())
                cam_coverage = float(cam_vals.sum()) / max(total_mass, 1e-8)
                reference_area = max(1.0, 0.02 * float(bone_cam.size))
            cam_density = float(cam_vals.mean())

            expansion = max(area / max(reference_area, 1.0), 1e-8)
            log_distance = np.log(expansion / area_target) / area_sigma
            area_score = float(np.exp(-0.5 * log_distance * log_distance))

            positive_points = (
                positive_points_by_component.get(component_id, ())
                if positive_points_by_component is not None else ()
            )
            negative_points = (
                negative_points_by_component.get(component_id, ())
                if negative_points_by_component is not None else ()
            )
            prompt_consistency = _point_consistency(m, positive_points, negative_points)
            if not positive_points and not negative_points:
                # Component coverage is the closest available prompt
                # consistency signal for older call sites that do not expose
                # the actual points.
                prompt_consistency = cam_coverage

            if method == "consistency_hybrid":
                # Image-only stability across independent CAM components.
                # Do not compare candidates from the same component: those
                # are alternate SAM outputs for one prompt and can tie
                # trivially.  The best cross-component IoU rewards a mask
                # reproduced by more than one CAM proposal.
                stability = 0.0
                for other_index in range(n):
                    if other_index == i:
                        continue
                    if component_ids is not None and len(component_ids) == n:
                        if int(component_ids[other_index]) == int(component_ids[i]):
                            continue
                    other = masks[other_index].astype(bool)
                    union = float(np.logical_or(m, other).sum())
                    if union > 0:
                        stability = max(
                            stability,
                            float(np.logical_and(m, other).sum()) / union,
                        )
                # Use soft CAM mass rather than the binary CAM>0.5 fraction;
                # the latter saturates at 1.0 for many candidates and was
                # observed to select tiny masks under component_topk=1.
                soft_mass_coverage = float(cam_vals.sum()) / max(float(bone_cam.sum()), 1e-8)
                prompt_consistency = 0.5 * prompt_consistency + 0.5 * stability
                terms = np.asarray(
                    [soft_mass_coverage, cam_density, sam_ranks[i], area_score, prompt_consistency],
                    dtype=np.float32,
                )
                scores[i] = float(np.dot(
                    np.asarray([0.25, 0.15, 0.15, 0.15, 0.30], dtype=np.float32), terms
                ))
            else:
                terms = np.asarray(
                    [cam_coverage, cam_density, sam_ranks[i], area_score, prompt_consistency],
                    dtype=np.float32,
                )
                scores[i] = float(np.dot(weights, terms))
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

    if anatomy_cam is not None and anatomy_consistency_weight > 0:
        if anatomy_cam.shape != bone_cam.shape:
            raise ValueError(
                f"anatomy_cam shape {anatomy_cam.shape} must match bone_cam shape {bone_cam.shape}"
            )
        weight = min(float(anatomy_consistency_weight), 1.0)
        agreement = np.zeros(n, dtype=np.float32)
        for i in range(n):
            m = masks[i].astype(bool)
            if not m.any():
                continue
            # Geometric mean (not arithmetic) of the two CAMs' mean
            # activation inside the candidate -- this is 0 whenever EITHER
            # CAM has ~0 mean activation there, so a candidate anatomy_cam
            # likes but bone_cam does not (or vice versa) gets no bonus;
            # only candidates BOTH signals independently support are boosted.
            bone_agreement = float(bone_cam[m].mean())
            anatomy_agreement = float(anatomy_cam[m].mean())
            agreement[i] = float(np.sqrt(max(bone_agreement, 0.0) * max(anatomy_agreement, 0.0)))
        scores = (1.0 - weight) * scores + weight * agreement
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
    positive_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    negative_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    prompt_hybrid_weights: tuple[float, float, float, float, float] = DEFAULT_PROMPT_HYBRID_WEIGHTS,
    prompt_area_target: float = 2.0,
    prompt_area_log_sigma: float = 1.0,
    best_per_component: bool = False,
    component_topk: int = 0,
    support_clip_kernel: int = 5,
    anatomy_cam: np.ndarray | None = None,
    anatomy_consistency_weight: float = 0.0,
) -> np.ndarray:
    """Select and fuse masks using CAM and bone morphology evidence.

    fusion_topk controls how the top-scored masks are combined:
      0 or 1 : logical-OR of all above-threshold masks (original behaviour)
      k > 1  : union (logical-OR) of the top-k above-threshold masks
      k < 0  : intersection (logical-AND) of the top-|k| above-threshold masks

    With best_per_component enabled, the best SAM candidate from each complete
    morphology proposal is selected before union. Otherwise the original
    global top-k behavior is preserved for ablation.

    component_masks and the per-component point dictionaries are used only by
    prompt_hybrid. Existing scoring methods preserve their prior behavior;
    in particular, bone_hybrid continues to use the whole-image bone_support
    rather than substituting a single component mask.

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
        component_ids=component_ids,
        component_masks=component_masks,
        positive_points_by_component=positive_points_by_component,
        negative_points_by_component=negative_points_by_component,
        prompt_hybrid_weights=prompt_hybrid_weights,
        prompt_area_target=prompt_area_target,
        prompt_area_log_sigma=prompt_area_log_sigma,
        anatomy_cam=anatomy_cam,
        anatomy_consistency_weight=anatomy_consistency_weight,
    )

    if best_per_component and component_ids is not None and component_ids.size == masks.shape[0]:
        selected_components: list[tuple[float, int]] = []
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
                selected_components.append((float(component_scores[best_local]), best_index))
        if component_topk > 0 and len(selected_components) > component_topk:
            selected_components.sort(key=lambda item: item[0], reverse=True)
            selected_components = selected_components[:component_topk]
        selected_indices = [index for _, index in selected_components]
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


# Confidence map pixel labels (see compute_confidence_map).
CONFIDENCE_BACKGROUND = 0
CONFIDENCE_FOREGROUND_CONFIDENT = 1
CONFIDENCE_FOREGROUND_UNCERTAIN = 2
CONFIDENCE_BOUNDARY_UNCERTAIN = 3


def compute_confidence_map(
    final_mask: np.ndarray,
    fused_cam: np.ndarray,
    cam_confident_percentile: float = 90.0,
    boundary_width_px: int = 3,
) -> np.ndarray:
    """Per-pixel confidence label for a pseudo-mask, alongside the binary mask.

    Four labels (see the CONFIDENCE_* constants above):
      0 background:            outside final_mask AND outside CAM's
                                confident-activation region -- both signals
                                agree this is not the lesion.
      1 foreground confident:  inside final_mask AND inside CAM's confident
                                region -- SAM's selected mask and the CAM
                                evidence that drove it agree.
      2 foreground uncertain:  inside final_mask XOR inside CAM's confident
                                region (exactly one signal supports it) --
                                e.g. SAM extended the mask beyond the CAM's
                                strongest activation, or a CAM-confident pixel
                                was not covered by the selected SAM mask.
      3 boundary uncertain:    within boundary_width_px of final_mask's edge
                                (dilation minus erosion) -- SAM's mask
                                boundary itself is typically the least
                                reliable region (see this project's own
                                oracle diagnostics), overriding whichever of
                                the above three labels a boundary pixel would
                                otherwise get.

    Args:
        final_mask:   [H, W] bool/uint8, the selected/fused pseudo-mask
                      (select_and_fuse_masks's output).
        fused_cam:    [H, W] float32 in [0, 1], the same CAM select_and_fuse_
                      masks scored candidates against.
        cam_confident_percentile: Percentile of fused_cam (over the whole
                      image) above which a pixel counts as "CAM-confident".
        boundary_width_px: How many pixels out from final_mask's boundary
                      (via dilation/erosion) count as "boundary uncertain".

    Returns:
        confidence: [H, W] uint8, one of the CONFIDENCE_* labels above.
    """
    mask_bool = final_mask.astype(bool)
    if fused_cam.shape != mask_bool.shape:
        raise ValueError(f"fused_cam shape {fused_cam.shape} must match final_mask shape {mask_bool.shape}")

    cam_threshold = float(np.percentile(fused_cam, cam_confident_percentile))
    cam_confident = fused_cam >= cam_threshold

    confidence = np.where(
        mask_bool & cam_confident,
        CONFIDENCE_FOREGROUND_CONFIDENT,
        np.where(
            mask_bool ^ cam_confident,
            CONFIDENCE_FOREGROUND_UNCERTAIN,
            CONFIDENCE_BACKGROUND,
        ),
    ).astype(np.uint8)

    if boundary_width_px > 0 and mask_bool.any():
        dilated = _binary_dilation(mask_bool, kernel_size=2 * boundary_width_px + 1).astype(bool)
        eroded = ~_binary_dilation(~mask_bool, kernel_size=2 * boundary_width_px + 1).astype(bool)
        boundary = dilated & ~eroded
        confidence[boundary] = CONFIDENCE_BOUNDARY_UNCERTAIN

    return confidence
