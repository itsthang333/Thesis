from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

from btxrd_wsss.config import SAMConfig, SelectionConfig
from btxrd_wsss.types import CandidateMask, Selection


def _ring(mask: np.ndarray) -> np.ndarray:
    radius = max(2, round(np.sqrt(mask.sum()) * 0.1))
    return ndimage.binary_dilation(mask, iterations=radius) & ~mask


def candidate_evidence(candidate: CandidateMask, evidence: np.ndarray) -> dict[str, float]:
    mask = np.asarray(candidate.mask, bool)
    component = np.asarray(candidate.metadata["source_component"], bool)
    ring = _ring(mask)
    inside = float(evidence[mask].mean()) if mask.any() else 0.0
    outside = float(evidence[ring].mean()) if ring.any() else 0.0
    intersection = np.logical_and(mask, component).sum()
    peak = (int(candidate.metadata["peak_y"]), int(candidate.metadata["peak_x"]))
    return {
        "purity": float(np.clip(inside, 0, 1)),
        "contrast": float(np.clip(inside - outside, 0, 1)),
        "coverage": float(intersection / max(1, component.sum())),
        "peak": float(mask[peak]),
        "predicted_iou": float(np.clip(candidate.predicted_iou, 0, 1)),
        "stability": float(np.clip(candidate.stability, 0, 1)),
        "sam_quality_raw": float(
            np.clip((candidate.predicted_iou + candidate.stability) / 2, 0, 1)
        ),
    }


def mask_size_bucket(candidate: CandidateMask, sam_config: SAMConfig) -> str:
    area_ratio = float(candidate.mask.sum() / candidate.mask.size)
    if area_ratio < sam_config.tiny_area_ratio:
        return "tiny"
    if area_ratio < sam_config.small_area_ratio:
        return "small"
    return "large"


def stability_floor(
    candidate: CandidateMask, sam_config: SAMConfig, selection_config: SelectionConfig
) -> float:
    bucket = mask_size_bucket(candidate, sam_config)
    if bucket == "tiny":
        return selection_config.minimum_tiny_stability
    if bucket == "small":
        return selection_config.minimum_small_stability
    return selection_config.minimum_stability


def gate_reasons(
    candidate: CandidateMask,
    stats: dict[str, float],
    sam_config: SAMConfig,
    selection_config: SelectionConfig,
) -> list[str]:
    area = int(candidate.mask.sum())
    reasons: list[str] = []
    if area < selection_config.minimum_mask_area:
        reasons.append("area_too_small")
    if area / candidate.mask.size > selection_config.maximum_mask_area_ratio:
        reasons.append("area_too_large")
    if candidate.stability < stability_floor(candidate, sam_config, selection_config):
        reasons.append("low_stability")
    if stats["coverage"] < selection_config.minimum_component_coverage:
        reasons.append("low_component_coverage")
    if stats["peak"] != 1.0:
        reasons.append("misses_source_peak")
    return reasons


def passes_gates(
    candidate: CandidateMask,
    stats: dict[str, float],
    sam_config: SAMConfig,
    selection_config: SelectionConfig,
) -> bool:
    return not gate_reasons(candidate, stats, sam_config, selection_config)


def upstream_score(source: str, stats: dict[str, float], config: SelectionConfig) -> float:
    weights = config.biomedclip_weights if source == "biomedclip" else config.hrnet_weights
    return float(sum(weights[key] * stats[key] for key in weights))


def percentile_ranks(values: list[float]) -> np.ndarray:
    if not values:
        return np.empty(0, np.float32)
    if len(values) == 1:
        return np.asarray([0.5], np.float32)
    return ((rankdata(values, method="average") - 0.5) / len(values)).astype(np.float32)


def add_multifocal_unions(
    candidates: list[CandidateMask], config: SelectionConfig
) -> list[CandidateMask]:
    if not config.add_multifocal_unions:
        return candidates
    additions: list[CandidateMask] = []
    by_source: dict[str, list[CandidateMask]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.proposal_source, []).append(candidate)
    for _source, members in by_source.items():
        members = sorted(
            members, key=lambda item: float(item.metadata.get("upstream_score", 0)), reverse=True
        )[:6]
        for size in range(2, min(config.maximum_components_per_union, len(members)) + 1):
            for group in combinations(members, size):
                union = np.logical_or.reduce([item.mask for item in group])
                # Only combine spatially distinct candidates; otherwise this is just a duplicate.
                if any(
                    np.logical_and(a.mask, b.mask).sum() / max(1, min(a.mask.sum(), b.mask.sum()))
                    > 0.25
                    for a, b in combinations(group, 2)
                ):
                    continue
                identifier = "union:" + "+".join(item.candidate_id for item in group)
                base = group[0]
                metadata = dict(base.metadata)
                metadata["union_members"] = [item.candidate_id for item in group]
                metadata["upstream_score"] = float(
                    np.mean([item.metadata["upstream_score"] for item in group])
                )
                additions.append(
                    replace(
                        base,
                        candidate_id=identifier,
                        mask=union,
                        proposal_id=identifier,
                        prompt_type="multifocal_union",
                        predicted_iou=float(np.mean([item.predicted_iou for item in group])),
                        stability=float(min(item.stability for item in group)),
                        metadata=metadata,
                    )
                )
                if len(additions) >= config.maximum_union_masks:
                    return candidates + additions
    return candidates + additions


def score_and_gate(
    candidates: list[CandidateMask],
    source_maps: dict[str, np.ndarray],
    selection_config: SelectionConfig,
    sam_config: SAMConfig,
) -> list[CandidateMask]:
    preliminary: list[tuple[CandidateMask, dict[str, float]]] = []
    for candidate in candidates:
        stats = candidate_evidence(candidate, source_maps[candidate.proposal_source])
        if not passes_gates(candidate, stats, sam_config, selection_config):
            continue
        preliminary.append((candidate, stats))

    accepted: list[CandidateMask] = []
    sources = dict.fromkeys(candidate.proposal_source for candidate, _stats in preliminary)
    for source in sources:
        members = [
            (candidate, stats)
            for candidate, stats in preliminary
            if candidate.proposal_source == source
        ]
        iou_ranks = percentile_ranks([stats["predicted_iou"] for _candidate, stats in members])
        stability_ranks = percentile_ranks([stats["stability"] for _candidate, stats in members])
        for (candidate, stats), iou_rank, stability_rank in zip(
            members, iou_ranks, stability_ranks, strict=True
        ):
            # The SAM-Med2D IoU head is not calibrated on BTXRD. Source-local ranks
            # retain useful ordering without treating its absolute values as probabilities.
            stats["sam_quality"] = float((iou_rank + stability_rank) / 2)
            metadata = dict(candidate.metadata)
            metadata.update(stats)
            metadata["size_bucket"] = mask_size_bucket(candidate, sam_config)
            metadata["stability_floor"] = stability_floor(
                candidate, sam_config, selection_config
            )
            metadata["upstream_score"] = upstream_score(
                candidate.proposal_source, stats, selection_config
            )
            accepted.append(replace(candidate, metadata=metadata))
    return accepted


def gate_audit(
    candidates: list[CandidateMask],
    source_maps: dict[str, np.ndarray],
    sam_config: SAMConfig,
    selection_config: SelectionConfig,
) -> dict[str, object]:
    rejected: dict[str, int] = {}
    by_size: dict[str, dict[str, int]] = {}
    accepted = 0
    for candidate in candidates:
        stats = candidate_evidence(candidate, source_maps[candidate.proposal_source])
        reasons = gate_reasons(candidate, stats, sam_config, selection_config)
        bucket = mask_size_bucket(candidate, sam_config)
        counts = by_size.setdefault(bucket, {"total": 0, "accepted": 0})
        counts["total"] += 1
        if not reasons:
            accepted += 1
            counts["accepted"] += 1
        for reason in reasons:
            rejected[reason] = rejected.get(reason, 0) + 1
    return {
        "total": len(candidates),
        "accepted": accepted,
        "rejected": len(candidates) - accepted,
        "rejection_reasons": rejected,
        "by_size": by_size,
    }


def unions_with_logits(
    candidates: list[CandidateMask],
    logits: np.ndarray,
    config: SelectionConfig,
) -> tuple[list[CandidateMask], np.ndarray]:
    """Add unions after G1; union logits are conservative member aggregates."""
    expanded = add_multifocal_unions(candidates, config)
    if len(expanded) == len(candidates):
        return expanded, np.asarray(logits, np.float32)
    by_id = {
        item.candidate_id: float(logit) for item, logit in zip(candidates, logits, strict=True)
    }
    union_logits: list[float] = []
    for candidate in expanded[len(candidates) :]:
        members = candidate.metadata["union_members"]
        values = np.asarray([by_id[member] for member in members], np.float32)
        # All components must be credible: mean with a penalty from the weakest member.
        union_logits.append(float(0.7 * values.mean() + 0.3 * values.min()))
    return expanded, np.concatenate((np.asarray(logits, np.float32), union_logits))


def select_final(
    image_id: str,
    candidates: list[CandidateMask],
    g1_logits: np.ndarray,
    config: SelectionConfig,
    *,
    bag_temperature: float = 0.2,
) -> Selection:
    if not candidates:
        raise ValueError("No candidates passed the selector gates")
    logits = np.asarray(g1_logits, np.float32)
    if logits.shape != (len(candidates),):
        raise ValueError("G1 logits must align with candidates")
    g1_ranks = percentile_ranks(logits.tolist())
    upstream_ranks = np.zeros(len(candidates), np.float32)
    for source in {item.proposal_source for item in candidates}:
        indices = [index for index, item in enumerate(candidates) if item.proposal_source == source]
        ranks = percentile_ranks(
            [float(candidates[index].metadata["upstream_score"]) for index in indices]
        )
        for index, rank in zip(indices, ranks, strict=True):
            confidence = float(candidates[index].metadata.get("source_confidence", 1.0))
            adjusted_confidence = (
                config.source_confidence_floor + (1 - config.source_confidence_floor) * confidence
            )
            upstream_ranks[index] = rank * adjusted_confidence
    final = config.g1_rank_weight * g1_ranks + config.upstream_rank_weight * upstream_ranks
    winner = int(np.argmax(final))
    probability = float(1 / (1 + np.exp(-logits[winner])))
    bag_logit = bag_temperature * (
        np.logaddexp.reduce(logits / bag_temperature) - np.log(len(logits))
    )
    bag_probability = float(1 / (1 + np.exp(-bag_logit)))
    order = np.sort(final)
    uncertainty = float(1 - (order[-1] - order[-2])) if len(order) > 1 else 0.5
    return Selection(
        image_id=image_id,
        candidate_id=candidates[winner].candidate_id,
        mask=candidates[winner].mask,
        probability=probability,
        bag_probability=bag_probability,
        uncertainty=uncertainty,
        evidence={
            "final_score": float(final[winner]),
            "g1_rank": float(g1_ranks[winner]),
            "upstream_rank": float(upstream_ranks[winner]),
            "upstream_score": float(candidates[winner].metadata["upstream_score"]),
        },
    )
