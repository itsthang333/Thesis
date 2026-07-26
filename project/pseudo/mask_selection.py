from __future__ import annotations

"""CAM-guided SAM mask scoring and selection (pipeline.md Stage 5)."""

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

SELECTION_METHODS = (
    "mean", "sum", "mean_area", "coverage", "coverage_mass", "coverage_mass_sam",
    "coverage_mass_sam_causal", "hybrid", "bone_hybrid", "simple_hybrid",
    "prompt_hybrid", "consistency_hybrid", "source_consensus",
    "prompt_source_graph",
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


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first.astype(bool)
    second_bool = second.astype(bool)
    union = float(np.logical_or(first_bool, second_bool).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(first_bool, second_bool).sum()) / union


def prompt_source_graph_selection(
    masks: np.ndarray,
    sam_scores: np.ndarray,
    component_ids: np.ndarray,
    prompt_modes: np.ndarray,
    proposal_source_ids: np.ndarray,
    *,
    component_topk: int,
    cluster_iou_threshold: float = 0.5,
) -> tuple[np.ndarray, list[int], dict[str, int]]:
    """Select prompt-stable medoids from source-consensus proposal clusters.

    This selector deliberately avoids a fitted weighted sum.  It first chooses
    one robust medoid per morphology component by lexicographically comparing
    agreement across distinct SAM prompt modes.  Component medoids are then
    grouped by mask IoU; clusters supported by more independent proposal
    sources rank first, while single-source clusters remain eligible as a
    fallback for lesions missed by one source.
    """
    count = int(masks.shape[0])
    scores = np.zeros(count, dtype=np.float32)
    components = np.asarray(component_ids, dtype=np.int32).reshape(-1)
    modes = np.asarray(prompt_modes, dtype="U32").reshape(-1)
    sources = np.asarray(proposal_source_ids, dtype="U32").reshape(-1)
    qualities = np.asarray(sam_scores, dtype=np.float32).reshape(-1)
    if any(len(values) != count for values in (components, modes, sources, qualities)):
        raise ValueError(
            "prompt_source_graph requires aligned masks, SAM scores, component IDs, "
            "prompt modes and proposal source IDs"
        )
    if count == 0:
        return scores, [], {"proposal_clusters": 0, "cross_source_clusters": 0}
    if not 0.0 <= float(cluster_iou_threshold) <= 1.0:
        raise ValueError("cluster_iou_threshold must be in [0, 1]")

    component_representatives: list[dict[str, object]] = []
    for component_id in np.unique(components):
        indices = np.where(components == component_id)[0]
        available_modes = sorted(set(str(modes[index]) for index in indices))
        ranked: list[tuple[tuple[float, ...], int, float, float]] = []
        for index in indices:
            other_mode_agreements: list[float] = []
            for other_mode in available_modes:
                if other_mode == str(modes[index]):
                    continue
                peers = [
                    int(peer)
                    for peer in indices
                    if str(modes[peer]) == other_mode
                ]
                if peers:
                    other_mode_agreements.append(
                        max(_mask_iou(masks[index], masks[peer]) for peer in peers)
                    )
            robust = min(other_mode_agreements) if other_mode_agreements else 0.0
            central = (
                float(np.median(other_mode_agreements))
                if other_mode_agreements else 0.0
            )
            key = (
                float(len(other_mode_agreements)),
                robust,
                central,
                float(qualities[index]),
                float(-int(index)),
            )
            ranked.append((key, int(index), robust, central))
            scores[index] = np.float32(robust)
        _, representative, robust, central = max(ranked, key=lambda item: item[0])
        component_representatives.append(
            {
                "component_id": int(component_id),
                "candidate_index": representative,
                "source": str(sources[representative]),
                "prompt_robust": robust,
                "prompt_central": central,
                "sam_score": float(qualities[representative]),
            }
        )

    representative_count = len(component_representatives)
    adjacency: list[set[int]] = [set() for _ in range(representative_count)]
    for left in range(representative_count):
        left_index = int(component_representatives[left]["candidate_index"])
        for right in range(left + 1, representative_count):
            right_index = int(component_representatives[right]["candidate_index"])
            if _mask_iou(masks[left_index], masks[right_index]) >= cluster_iou_threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    clusters: list[list[int]] = []
    unseen = set(range(representative_count))
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        cluster: list[int] = []
        while stack:
            node = stack.pop()
            cluster.append(node)
            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        clusters.append(sorted(cluster))

    ranked_clusters: list[tuple[tuple[float, ...], int]] = []
    cross_source_cluster_count = 0
    for cluster in clusters:
        cluster_sources = {
            str(component_representatives[node]["source"]) for node in cluster
        }
        if len(cluster_sources) > 1:
            cross_source_cluster_count += 1
        candidate_nodes: list[tuple[tuple[float, ...], int]] = []
        pairwise_values: list[float] = []
        for position, left in enumerate(cluster):
            left_index = int(component_representatives[left]["candidate_index"])
            for right in cluster[position + 1:]:
                right_index = int(component_representatives[right]["candidate_index"])
                pairwise_values.append(_mask_iou(masks[left_index], masks[right_index]))
        cluster_cohesion = (
            float(np.median(pairwise_values)) if pairwise_values else 0.0
        )
        for node in cluster:
            record = component_representatives[node]
            index = int(record["candidate_index"])
            cross_source_iou = 0.0
            medoid_agreements: list[float] = []
            for other in cluster:
                if other == node:
                    continue
                other_record = component_representatives[other]
                other_index = int(other_record["candidate_index"])
                agreement = _mask_iou(masks[index], masks[other_index])
                medoid_agreements.append(agreement)
                if str(other_record["source"]) != str(record["source"]):
                    cross_source_iou = max(cross_source_iou, agreement)
            medoid_centrality = (
                float(np.mean(medoid_agreements)) if medoid_agreements else 0.0
            )
            candidate_nodes.append(
                (
                    (
                        cross_source_iou,
                        medoid_centrality,
                        float(record["prompt_robust"]),
                        float(record["prompt_central"]),
                        float(record["sam_score"]),
                        float(-index),
                    ),
                    index,
                )
            )
        representative_index = max(candidate_nodes, key=lambda item: item[0])[1]
        representative_record = next(
            record
            for record in component_representatives
            if int(record["candidate_index"]) == representative_index
        )
        cluster_key = (
            float(len(cluster_sources)),
            float(len(cluster)),
            cluster_cohesion,
            float(representative_record["prompt_robust"]),
            float(representative_record["prompt_central"]),
            float(representative_record["sam_score"]),
            float(-representative_index),
        )
        ranked_clusters.append((cluster_key, representative_index))

    ranked_clusters.sort(key=lambda item: item[0], reverse=True)
    limit = len(ranked_clusters) if component_topk <= 0 else component_topk
    selected = [index for _, index in ranked_clusters[:limit]]
    for rank, index in enumerate(selected):
        # A diagnostic-only ordering code. Selection itself uses the complete
        # lexicographic keys above, not this scalar.
        scores[index] = np.float32(2.0 + (len(selected) - rank) / max(1, len(selected)))
    return scores, selected, {
        "proposal_clusters": len(clusters),
        "cross_source_clusters": cross_source_cluster_count,
    }


def score_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    method: str = "mean",
    bone_likelihood: np.ndarray | None = None,
    bone_support: np.ndarray | None = None,
    sam_scores: np.ndarray | None = None,
    classifier_causal_scores: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
    component_masks: np.ndarray | None = None,
    positive_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    negative_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    proposal_teacher_probability: np.ndarray | None = None,
    proposal_teacher_component_start: int | None = None,
    prompt_modes: np.ndarray | None = None,
    proposal_source_ids: np.ndarray | None = None,
    graph_component_topk: int = 0,
    prompt_hybrid_weights: tuple[float, float, float, float, float] = DEFAULT_PROMPT_HYBRID_WEIGHTS,
    prompt_area_target: float = 2.0,
    prompt_area_log_sigma: float = 1.0,
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
    if method == "prompt_source_graph":
        if (
            sam_scores is None
            or component_ids is None
            or prompt_modes is None
            or proposal_source_ids is None
        ):
            raise ValueError(
                "prompt_source_graph requires SAM scores, component IDs, prompt "
                "modes and proposal source IDs"
            )
        graph_scores, _, _ = prompt_source_graph_selection(
            masks,
            sam_scores,
            component_ids,
            prompt_modes,
            proposal_source_ids,
            component_topk=graph_component_topk,
        )
        return graph_scores
    scores = np.zeros(n, dtype=np.float32)
    sam_ranks = _within_group_percentile_ranks(sam_scores, component_ids, n)
    causal_ranks = _within_group_percentile_ranks(
        classifier_causal_scores, component_ids, n
    )

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
            scores[i] = float((cam_vals > 0.5).sum()) / area
        elif method == "coverage_mass":
            cam_density = float((cam_vals > 0.5).sum()) / area
            mass_coverage = float(cam_vals.sum()) / max(float(bone_cam.sum()), 1e-8)
            scores[i] = 0.70 * cam_density + 0.30 * mass_coverage
        elif method == "coverage_mass_sam":
            cam_density = float((cam_vals > 0.5).sum()) / area
            mass_coverage = float(cam_vals.sum()) / max(float(bone_cam.sum()), 1e-8)
            scores[i] = (
                0.60 * cam_density
                + 0.25 * mass_coverage
                + 0.15 * float(sam_ranks[i])
            )
        elif method == "coverage_mass_sam_causal":
            cam_density = float((cam_vals > 0.5).sum()) / area
            mass_coverage = float(cam_vals.sum()) / max(float(bone_cam.sum()), 1e-8)
            scores[i] = (
                0.45 * cam_density
                + 0.20 * mass_coverage
                + 0.15 * float(sam_ranks[i])
                + 0.20 * float(causal_ranks[i])
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
                prompt_consistency = cam_coverage

            if method == "consistency_hybrid":
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
        elif method == "source_consensus":
            if (
                proposal_teacher_probability is None
                or proposal_teacher_probability.shape != bone_cam.shape
                or proposal_teacher_component_start is None
                or component_ids is None
                or len(component_ids) != n
            ):
                raise ValueError(
                    "source_consensus requires an aligned proposal-teacher "
                    "probability map, component boundary and candidate component IDs"
                )
            component_id = int(component_ids[i])
            is_teacher = component_id >= int(proposal_teacher_component_start)
            source_map = (
                proposal_teacher_probability.astype(np.float32)
                if is_teacher
                else bone_cam
            )
            support = component_mask_by_id.get(component_id)
            if support is not None and support.any():
                source_mass = float(source_map[support].sum())
                source_coverage = float(source_map[m & support].sum()) / max(
                    source_mass, 1e-8
                )
            else:
                source_coverage = float(source_map[m].sum()) / max(
                    float(source_map.sum()), 1e-8
                )
            source_density = float(source_map[m].mean())
            cam_density = float((cam_vals > 0.5).sum()) / area
            cam_mass_coverage = float(cam_vals.sum()) / max(
                float(bone_cam.sum()), 1e-8
            )
            cross_source_iou = 0.0
            for other_index in range(n):
                if other_index == i:
                    continue
                other_component = int(component_ids[other_index])
                other_is_teacher = (
                    other_component >= int(proposal_teacher_component_start)
                )
                if other_is_teacher == is_teacher:
                    continue
                other = masks[other_index].astype(bool)
                union = float(np.logical_or(m, other).sum())
                if union > 0:
                    cross_source_iou = max(
                        cross_source_iou,
                        float(np.logical_and(m, other).sum()) / union,
                    )
            scores[i] = (
                0.25 * cam_density
                + 0.15 * cam_mass_coverage
                + 0.15 * float(sam_ranks[i])
                + 0.25 * source_coverage
                + 0.10 * source_density
                + 0.10 * cross_source_iou
            )
        elif method == "bone_hybrid":
            if bone_likelihood is None:
                scores[i] = float(cam_vals.mean())
                continue
            bone_mean = float(bone_likelihood[m].mean())
            cam_mean = float(cam_vals.mean())
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
    """Intersect a candidate/fused mask with the (optionally dilated) tumor
    support region. An empty intersection stays empty: restoring the original
    mask would silently bypass the image-derived support constraint. Shared by
    selection and oracle diagnostics so both measure the same fail-closed rule.
    """
    fused_mask = fused_mask.astype(np.uint8)
    if (
        selection_method not in {
            "bone_hybrid",
            "coverage_mass_sam",
            "coverage_mass_sam_causal",
            "source_consensus",
            "prompt_source_graph",
        }
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
    return clipped.astype(np.uint8)


def select_and_fuse_masks(
    masks: np.ndarray,
    bone_cam: np.ndarray,
    mask_score_threshold: float = 0.4,
    selection_method: str = "mean",
    fusion_topk: int = 0,
    bone_likelihood: np.ndarray | None = None,
    bone_support: np.ndarray | None = None,
    sam_scores: np.ndarray | None = None,
    classifier_causal_scores: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
    component_masks: np.ndarray | None = None,
    positive_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    negative_points_by_component: dict[int, tuple[tuple[int, int], ...]] | None = None,
    proposal_teacher_probability: np.ndarray | None = None,
    proposal_teacher_component_start: int | None = None,
    prompt_modes: np.ndarray | None = None,
    proposal_source_ids: np.ndarray | None = None,
    prompt_hybrid_weights: tuple[float, float, float, float, float] = DEFAULT_PROMPT_HYBRID_WEIGHTS,
    prompt_area_target: float = 2.0,
    prompt_area_log_sigma: float = 1.0,
    best_per_component: bool = False,
    component_topk: int = 0,
    support_clip_kernel: int = 5,
    low_score_policy: str = "empty",
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """Select and fuse masks using CAM and bone morphology evidence.

    fusion_topk controls how the top-scored masks are combined:
      0       : logical-OR of all above-threshold masks
      1       : keep only the highest-scoring above-threshold mask
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
        low_score_policy:    "empty" rejects all candidates below the threshold;
                             "keep-best" retains the best candidate for debug ablations.

    Returns:
        pseudo_mask: [H, W] uint8 binary mask (0 / 1).
    """
    def result(mask: np.ndarray, selected_indices: list[int], above_count: int):
        selected_component_count = 0
        if component_ids is not None and component_ids.size == masks.shape[0] and selected_indices:
            selected_component_count = int(len(np.unique(component_ids[selected_indices])))
        details = {
            "above_threshold_candidates": int(above_count),
            "selected_candidates": int(len(selected_indices)),
            "selected_components": selected_component_count,
        }
        return (mask, details) if return_details else mask

    if masks.shape[0] == 0:
        h, w = bone_cam.shape
        return result(np.zeros((h, w), dtype=np.uint8), [], 0)
    if low_score_policy not in {"empty", "keep-best"}:
        raise ValueError("low_score_policy must be 'empty' or 'keep-best'")

    def _clip(fused_mask: np.ndarray) -> np.ndarray:
        return constrain_to_bone_support(fused_mask, bone_support, selection_method, support_clip_kernel)

    if selection_method == "prompt_source_graph":
        if (
            sam_scores is None
            or component_ids is None
            or prompt_modes is None
            or proposal_source_ids is None
        ):
            raise ValueError(
                "prompt_source_graph requires SAM scores, component IDs, prompt "
                "modes and proposal source IDs"
            )
        graph_scores, selected_indices, graph_details = prompt_source_graph_selection(
            masks,
            sam_scores,
            component_ids,
            prompt_modes,
            proposal_source_ids,
            component_topk=component_topk,
        )
        if not selected_indices:
            return result(np.zeros_like(bone_cam, dtype=np.uint8), [], 0)
        selected_mask = masks[selected_indices].any(axis=0).astype(np.uint8)
        output = result(
            _clip(selected_mask),
            selected_indices,
            int(np.count_nonzero(graph_scores > 0)),
        )
        if return_details:
            mask, details = output
            details.update(graph_details)
            return mask, details
        return output

    scores = score_masks(
        masks,
        bone_cam,
        method=selection_method,
        bone_likelihood=bone_likelihood,
        bone_support=bone_support,
        sam_scores=sam_scores,
        classifier_causal_scores=classifier_causal_scores,
        component_ids=component_ids,
        component_masks=component_masks,
        positive_points_by_component=positive_points_by_component,
        negative_points_by_component=negative_points_by_component,
        proposal_teacher_probability=proposal_teacher_probability,
        proposal_teacher_component_start=proposal_teacher_component_start,
        prompt_modes=prompt_modes,
        proposal_source_ids=proposal_source_ids,
        prompt_hybrid_weights=prompt_hybrid_weights,
        prompt_area_target=prompt_area_target,
        prompt_area_log_sigma=prompt_area_log_sigma,
    )
    above_threshold_count = int(np.count_nonzero(scores >= mask_score_threshold))

    if best_per_component and component_ids is not None and component_ids.size == masks.shape[0]:
        selected_components: list[tuple[float, int]] = []
        for component_id in np.unique(component_ids):
            candidates = np.where(component_ids == component_id)[0]
            if candidates.size == 0:
                continue
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
            return result(
                _clip(masks[selected_indices].any(axis=0)),
                selected_indices,
                above_threshold_count,
            )

    order = np.argsort(scores)[::-1]
    above = [i for i in order if scores[i] >= mask_score_threshold]

    if not above:
        if low_score_policy == "empty":
            return result(np.zeros_like(bone_cam, dtype=np.uint8), [], above_threshold_count)
        above = [int(order[0])]

    if fusion_topk == 1:
        # top-1 only — return the single best-scoring mask
        fused = masks[above[0]].copy().astype(np.uint8)
        selected_indices = above[:1]
    elif fusion_topk == 0:
        # default: logical-OR of all above-threshold masks
        selected = masks[above]
        fused = selected.any(axis=0).astype(np.uint8)
        selected_indices = above
    elif fusion_topk > 1:
        # union of top-k
        topk = above[:fusion_topk]
        fused = masks[topk[0]].copy().astype(bool)
        for i in topk[1:]:
            fused = fused | masks[i].astype(bool)
        fused = fused.astype(np.uint8)
        selected_indices = topk
    else:
        # fusion_topk < 0 → intersection of top-|k|
        k = abs(fusion_topk)
        topk = above[:k]
        fused = masks[topk[0]].copy().astype(bool)
        for i in topk[1:]:
            fused = fused & masks[i].astype(bool)
        fused = fused.astype(np.uint8)
        selected_indices = topk
    return result(_clip(fused), selected_indices, above_threshold_count)
