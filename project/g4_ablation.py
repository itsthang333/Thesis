from __future__ import annotations

"""Deterministic, annotation-free score ablations for the G4 study.

All functions operate only on frozen candidate metadata.  Spatial ground truth
is deliberately absent from this module; oracle choices belong exclusively to
the downstream evaluator.
"""

from dataclasses import dataclass
import hashlib
from itertools import combinations

import numpy as np

from final_selector import average_percentile_rank, stable_select


SOURCE_L320 = "layercam"
SOURCE_C448 = "classifier448:layercam"
SOURCE_EXTERNAL = "external_saliency"
ALL_SOURCES = (SOURCE_L320, SOURCE_C448, SOURCE_EXTERNAL)
SOURCE_SUBSETS = {
    "+".join(subset): subset
    for size in range(1, len(ALL_SOURCES) + 1)
    for subset in combinations(ALL_SOURCES, size)
}
UPSTREAM_ARMS = ("U0", "U1", "U2", "U3", "U4", "U5", "U6")
FUSION_ARMS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")


@dataclass(frozen=True)
class UpstreamComponents:
    sam_score: np.ndarray
    cam_density: np.ndarray
    cam_mass_coverage: np.ndarray
    sam_component_rank: np.ndarray
    sam_global_rank: np.ndarray


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite nonempty vector")
    return result


def _validate_aligned(**arrays: np.ndarray) -> int:
    lengths = {name: int(np.asarray(value).shape[0]) for name, value in arrays.items()}
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"candidate arrays are not aligned: {lengths}")
    count = next(iter(lengths.values()))
    if count < 1:
        raise ValueError("candidate arrays must be nonempty")
    return count


def within_group_percentile_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = _finite_vector(values, name="values")
    groups = np.asarray(groups).reshape(-1)
    _validate_aligned(values=values, groups=groups)
    result = np.empty_like(values)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        result[indices] = average_percentile_rank(values[indices])
    return result


def upstream_components(
    masks: np.ndarray,
    prompt_map: np.ndarray,
    sam_scores: np.ndarray,
    component_ids: np.ndarray,
) -> UpstreamComponents:
    masks = np.asarray(masks, dtype=bool)
    prompt_map = np.asarray(prompt_map, dtype=np.float64)
    sam_scores = _finite_vector(sam_scores, name="sam_scores")
    component_ids = np.asarray(component_ids).reshape(-1)
    if masks.ndim != 3 or prompt_map.shape != masks.shape[1:]:
        raise ValueError("masks must be NxHxW and prompt_map must be HxW")
    _validate_aligned(masks=masks, sam_scores=sam_scores, component_ids=component_ids)
    if not np.isfinite(prompt_map).all() or np.any(prompt_map < 0):
        raise ValueError("prompt_map must be finite and nonnegative")
    flat_masks = masks.reshape(len(masks), -1)
    flat_prompt = prompt_map.reshape(-1)
    areas = flat_masks.sum(axis=1)
    cam_mass = flat_masks @ flat_prompt
    cam_positive = flat_masks @ (flat_prompt > 0.5).astype(np.float64)
    density = np.divide(
        cam_positive,
        areas,
        out=np.zeros(len(masks), dtype=np.float64),
        where=areas > 0,
    )
    total_mass = float(flat_prompt.sum())
    mass_coverage = (
        cam_mass / total_mass if total_mass > 0 else np.zeros(len(masks), dtype=np.float64)
    )
    return UpstreamComponents(
        sam_score=sam_scores,
        cam_density=density,
        cam_mass_coverage=mass_coverage,
        sam_component_rank=within_group_percentile_rank(sam_scores, component_ids),
        sam_global_rank=average_percentile_rank(sam_scores),
    )


def source_correct_upstream_components(
    masks: np.ndarray,
    prompt_maps: np.ndarray,
    sam_scores: np.ndarray,
    component_groups: np.ndarray,
) -> UpstreamComponents:
    """Compute upstream evidence with the generating map of each candidate.

    A merged rich gallery contains candidates produced by several localization
    sources. Reusing one map for every candidate makes density and captured-mass
    terms source-incorrect. ``prompt_maps[i]`` must therefore be the normalized
    map that generated candidate ``i``. ``component_groups`` must also be
    source-qualified (for example ``"layercam:3"``), preventing SAM ranks from
    leaking across unrelated components whose integer IDs happen to collide.

    This is algebraically identical to :func:`upstream_components` when all
    candidates use the same map. Spatial annotations are neither accepted nor
    used.
    """

    masks = np.asarray(masks, dtype=bool)
    prompt_maps = np.asarray(prompt_maps, dtype=np.float64)
    sam_scores = _finite_vector(sam_scores, name="sam_scores")
    component_groups = np.asarray(component_groups).reshape(-1)
    if masks.ndim != 3 or prompt_maps.shape != masks.shape:
        raise ValueError("masks and prompt_maps must both be NxHxW")
    _validate_aligned(
        masks=masks,
        prompt_maps=prompt_maps,
        sam_scores=sam_scores,
        component_groups=component_groups,
    )
    if not np.isfinite(prompt_maps).all() or np.any(prompt_maps < 0):
        raise ValueError("prompt_maps must be finite and nonnegative")

    flat_masks = masks.reshape(len(masks), -1)
    flat_prompt = prompt_maps.reshape(len(masks), -1)
    areas = flat_masks.sum(axis=1)
    cam_mass = np.einsum("ij,ij->i", flat_masks, flat_prompt, optimize=True)
    cam_positive = np.einsum(
        "ij,ij->i",
        flat_masks,
        (flat_prompt > 0.5).astype(np.float64),
        optimize=True,
    )
    density = np.divide(
        cam_positive,
        areas,
        out=np.zeros(len(masks), dtype=np.float64),
        where=areas > 0,
    )
    total_mass = flat_prompt.sum(axis=1)
    mass_coverage = np.divide(
        cam_mass,
        total_mass,
        out=np.zeros(len(masks), dtype=np.float64),
        where=total_mass > 0,
    )
    return UpstreamComponents(
        sam_score=sam_scores,
        cam_density=density,
        cam_mass_coverage=mass_coverage,
        sam_component_rank=within_group_percentile_rank(
            sam_scores, component_groups
        ),
        sam_global_rank=average_percentile_rank(sam_scores),
    )


def upstream_score(components: UpstreamComponents, arm: str) -> np.ndarray:
    arm = str(arm).upper()
    d = components.cam_density
    m = components.cam_mass_coverage
    local = components.sam_component_rank
    if arm == "U0":
        return components.sam_score.copy()
    if arm == "U1":
        return d.copy()
    if arm == "U2":
        return m.copy()
    if arm == "U3":
        return 0.5 * d + 0.5 * m
    if arm == "U4":
        return (d + m + local) / 3.0
    if arm == "U5":
        return 0.60 * d + 0.25 * m + 0.15 * local
    if arm == "U6":
        return 0.60 * d + 0.25 * m + 0.15 * components.sam_global_rank
    raise ValueError(f"unknown upstream arm: {arm}")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = _finite_vector(values, name="values")
    standard_deviation = float(values.std())
    return np.zeros_like(values) if standard_deviation == 0 else (values - values.mean()) / standard_deviation


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = _finite_vector(values, name="values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # 1.4826 makes MAD consistent for a Gaussian distribution.  A constant
    # scorer carries no ranking information and therefore contributes zero.
    return np.zeros_like(values) if mad == 0 else (values - median) / (1.4826 * mad)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = _finite_vector(values, name="values")
    span = float(values.max() - values.min())
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def _descending_average_ranks(values: np.ndarray) -> np.ndarray:
    percentile = average_percentile_rank(_finite_vector(values, name="values"))
    # Best item has rank one. Average-tie percentile ranks preserve ties.
    return 1.0 + (1.0 - percentile) * max(len(percentile) - 1, 1)


def fusion_score(g1_logits: np.ndarray, upstream_scores: np.ndarray, arm: str) -> np.ndarray:
    g1 = _finite_vector(g1_logits, name="g1_logits")
    upstream = _finite_vector(upstream_scores, name="upstream_scores")
    _validate_aligned(g1=g1, upstream=upstream)
    arm = str(arm).upper()
    if arm == "R0":
        return upstream.copy()
    if arm == "R1":
        return g1.copy()
    if arm == "R2":
        return _zscore(g1) + _zscore(upstream)
    if arm == "R3":
        return _robust_zscore(g1) + _robust_zscore(upstream)
    if arm == "R4":
        return _minmax(g1) + _minmax(upstream)
    if arm == "R5":
        # Cormack et al.'s RRF uses reciprocal ranks and k=60.  Higher is better.
        return 1.0 / (60.0 + _descending_average_ranks(g1)) + 1.0 / (
            60.0 + _descending_average_ranks(upstream)
        )
    weights = {"R6": 0.25, "R7": 0.50, "R8": 0.75}
    if arm in weights:
        g1_weight = weights[arm]
        return g1_weight * average_percentile_rank(g1) + (1.0 - g1_weight) * average_percentile_rank(upstream)
    raise ValueError(f"unknown fusion arm: {arm}")


def candidate_filter(
    sources: np.ndarray,
    upstream_scores: np.ndarray,
    *,
    allowed_sources: tuple[str, ...] = ALL_SOURCES,
    per_source_cap: int | None = None,
    prompt_mode: str | None = None,
    prompt_modes: np.ndarray | None = None,
) -> np.ndarray:
    sources = np.asarray(sources).astype(str).reshape(-1)
    upstream_scores = _finite_vector(upstream_scores, name="upstream_scores")
    _validate_aligned(sources=sources, upstream=upstream_scores)
    keep = np.isin(sources, np.asarray(allowed_sources, dtype=str))
    if prompt_mode is not None:
        if prompt_modes is None:
            raise ValueError("prompt_modes are required for a prompt-mode filter")
        modes = np.asarray(prompt_modes).astype(str).reshape(-1)
        _validate_aligned(sources=sources, prompt_modes=modes)
        keep &= modes == str(prompt_mode)
    if per_source_cap is not None:
        if per_source_cap < 1:
            raise ValueError("per_source_cap must be positive")
        capped = np.zeros(len(sources), dtype=bool)
        for source in allowed_sources:
            indices = np.flatnonzero(keep & (sources == source))
            ranked = sorted(indices.tolist(), key=lambda index: (-upstream_scores[index], index))
            capped[ranked[:per_source_cap]] = True
        keep &= capped
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError("candidate filter produced an empty gallery")
    return indices


def select_from_scores(scores: np.ndarray, g1_logits: np.ndarray, candidate_indices: np.ndarray) -> int:
    scores = _finite_vector(scores, name="scores")
    g1 = _finite_vector(g1_logits, name="g1_logits")
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    _validate_aligned(scores=scores, g1=g1, indices=indices)
    local = stable_select(scores, g1)
    return int(indices[local])


def select_score_only(scores: np.ndarray, candidate_indices: np.ndarray) -> int:
    """Select a scorer's maximum with only the frozen lower-index tie-break."""

    values = _finite_vector(scores, name="scores")
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    _validate_aligned(scores=values, indices=indices)
    local = max(range(len(values)), key=lambda index: (values[index], -int(indices[index])))
    return int(indices[local])


def deterministic_random_candidate(image_id: str, candidate_indices: np.ndarray, seed: int) -> int:
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("random selector requires candidates")
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % len(indices)
    return int(indices[offset])


__all__ = [
    "ALL_SOURCES",
    "FUSION_ARMS",
    "SOURCE_SUBSETS",
    "UPSTREAM_ARMS",
    "UpstreamComponents",
    "candidate_filter",
    "deterministic_random_candidate",
    "fusion_score",
    "select_from_scores",
    "select_score_only",
    "source_correct_upstream_components",
    "upstream_components",
    "upstream_score",
    "within_group_percentile_rank",
]
