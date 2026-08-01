"""GT-blind BAS evidence and exact rich-gallery rank-pair primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from models.bas_candidate_localizer import candidate_activation_evidence


SOURCE_TO_ID = {
    "classifier448": 0,
    "layercam320": 1,
    "external_saliency": 2,
}


def canonical_source(value: object) -> str:
    """Map frozen collaborator source labels without using them as a score."""

    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown rich-gallery source: {value!r}")


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Exact ascending average-tie percentile ranks used by frozen G1 fusion."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("rank input must be one finite nonempty vector")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / max(len(array) - 1, 1)


def stable_select(scores: np.ndarray, raw_g1_logits: np.ndarray) -> int:
    """Select by aggregate, then frozen G1 logit, then lower local index."""

    values = np.asarray(scores, dtype=np.float64)
    raw = np.asarray(raw_g1_logits, dtype=np.float64)
    if values.ndim != 1 or values.shape != raw.shape or not len(values):
        raise ValueError("selector arrays must be aligned nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(raw).all():
        raise ValueError("selector arrays must be finite")
    return int(
        max(
            range(len(values)),
            key=lambda index: (values[index], raw[index], -index),
        )
    )


def _finite_vector(value: object, *, name: str, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite nonempty vector")
    return result


@dataclass(frozen=True)
class RichGalleryAlignedPayload:
    candidate_indices: np.ndarray
    candidate_masks: np.ndarray
    source_ids: np.ndarray
    g1_logits: np.ndarray
    upstream_scores: np.ndarray


def align_transport_payloads(
    candidate_payload: Mapping[str, object],
    stage_a_payload: Mapping[str, object],
) -> RichGalleryAlignedPayload:
    """Fail closed unless candidate masks and collaborator Stage-A scores align."""

    required_candidate = {"sam_masks", "selection_scores", "proposal_source_ids"}
    required_stage_a = {
        "candidate_indices",
        "source_ids",
        "upstream_scores",
        "g1_frozen_candidate_logits",
    }
    if not required_candidate.issubset(candidate_payload):
        raise ValueError("rich-gallery candidate payload is incomplete")
    if not required_stage_a.issubset(stage_a_payload):
        raise ValueError("rich-gallery Stage-A score payload is incomplete")

    masks = np.asarray(candidate_payload["sam_masks"])
    full_upstream = _finite_vector(
        candidate_payload["selection_scores"],
        name="candidate selection_scores",
        dtype=np.float32,
    )
    source_names = np.asarray(candidate_payload["proposal_source_ids"]).reshape(-1)
    if masks.ndim != 3 or not len(masks):
        raise ValueError("sam_masks must have shape [N,H,W]")
    if len(full_upstream) != len(masks) or len(source_names) != len(masks):
        raise ValueError("candidate masks, upstream scores and sources must align")

    indices_raw = np.asarray(stage_a_payload["candidate_indices"])
    if indices_raw.ndim != 1 or not np.issubdtype(indices_raw.dtype, np.integer):
        raise ValueError("candidate_indices must be one integer vector")
    indices = indices_raw.astype(np.int64, copy=False)
    if (
        not len(indices)
        or len(np.unique(indices)) != len(indices)
        or int(indices.min()) < 0
        or int(indices.max()) >= len(masks)
    ):
        raise ValueError("candidate_indices are empty, duplicated or out of range")

    stage_upstream = _finite_vector(
        stage_a_payload["upstream_scores"],
        name="Stage-A upstream_scores",
        dtype=np.float32,
    )
    g1_logits = _finite_vector(
        stage_a_payload["g1_frozen_candidate_logits"],
        name="G1 candidate logits",
        dtype=np.float32,
    )
    stage_sources_raw = np.asarray(stage_a_payload["source_ids"])
    if stage_sources_raw.ndim != 1 or not np.issubdtype(
        stage_sources_raw.dtype, np.integer
    ):
        raise ValueError("Stage-A source_ids must be one integer vector")
    stage_sources = stage_sources_raw.astype(np.int64, copy=False)
    if not (
        len(indices)
        == len(stage_upstream)
        == len(g1_logits)
        == len(stage_sources)
    ):
        raise ValueError("Stage-A candidate arrays do not align")
    if not np.array_equal(stage_upstream, full_upstream[indices]):
        raise ValueError("Stage-A upstream scores changed from candidate payload")

    expected_sources = np.asarray(
        [SOURCE_TO_ID[canonical_source(source_names[index])] for index in indices],
        dtype=np.int64,
    )
    if not np.array_equal(stage_sources, expected_sources):
        raise ValueError("Stage-A source ids changed from candidate payload")

    return RichGalleryAlignedPayload(
        candidate_indices=indices,
        candidate_masks=masks[indices].astype(bool, copy=False),
        source_ids=stage_sources,
        g1_logits=g1_logits,
        upstream_scores=stage_upstream,
    )


def bas_candidate_scores(
    activation: np.ndarray,
    candidate_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute class-aware BAS coverage/purity/harmonic evidence without GT."""

    activation_array = np.asarray(activation, dtype=np.float32)
    masks = np.asarray(candidate_masks)
    if activation_array.ndim != 2 or masks.ndim != 3 or not len(masks):
        raise ValueError("activation/masks must have shapes [H,W] and [N,H,W]")
    if not np.isfinite(activation_array).all():
        raise ValueError("BAS activation must be finite")
    valid = torch.ones((1, len(masks)), dtype=torch.bool)
    coverage, purity, harmonic = candidate_activation_evidence(
        torch.from_numpy(activation_array)[None, None],
        torch.from_numpy(masks.astype(np.float32, copy=False))[None],
        valid,
    )
    return tuple(
        value[0].detach().cpu().numpy().astype(np.float64)
        for value in (coverage, purity, harmonic)
    )


@dataclass(frozen=True)
class RichGalleryBasPair:
    baseline_rank: np.ndarray
    bas_residual_rank: np.ndarray
    baseline_local_index: int
    bas_residual_local_index: int


def score_rich_gallery_bas_pair(
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
    bas_scores: np.ndarray,
) -> RichGalleryBasPair:
    """Return frozen two-rank control and the sole equal three-rank BAS arm."""

    g1 = _finite_vector(g1_logits, name="G1 logits", dtype=np.float64)
    upstream = _finite_vector(
        upstream_scores, name="upstream scores", dtype=np.float64
    )
    bas = _finite_vector(bas_scores, name="BAS scores", dtype=np.float64)
    if not (g1.shape == upstream.shape == bas.shape):
        raise ValueError("G1, upstream and BAS score vectors must align")
    g1_rank = average_percentile_rank(g1)
    upstream_rank = average_percentile_rank(upstream)
    bas_rank = average_percentile_rank(bas)
    baseline = 0.5 * (g1_rank + upstream_rank)
    residual = (g1_rank + upstream_rank + bas_rank) / 3.0
    return RichGalleryBasPair(
        baseline_rank=baseline,
        bas_residual_rank=residual,
        baseline_local_index=stable_select(baseline, g1),
        bas_residual_local_index=stable_select(residual, g1),
    )


__all__ = [
    "RichGalleryAlignedPayload",
    "RichGalleryBasPair",
    "align_transport_payloads",
    "average_percentile_rank",
    "bas_candidate_scores",
    "canonical_source",
    "score_rich_gallery_bas_pair",
    "stable_select",
]
