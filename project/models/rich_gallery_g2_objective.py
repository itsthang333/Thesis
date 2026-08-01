from __future__ import annotations

"""Source-safe objective primitives for the rich-gallery G2 selector.

This module is deliberately annotation- and dataset-agnostic.  Source IDs and
upstream scores are immutable proposal metadata; no spatial target is accepted
by any public function.
"""

import numpy as np
import torch
import torch.nn.functional as F


SHARED_SOURCE_IDS = (0, 1)
EXTERNAL_SOURCE_ID = 2


def shared_source_validity(
    candidate_valid: torch.Tensor,
    source_ids: torch.Tensor,
) -> torch.Tensor:
    """Keep only classifier448/layercam candidates for training losses."""

    if candidate_valid.shape != source_ids.shape or candidate_valid.ndim != 2:
        raise ValueError("candidate_valid/source_ids must share shape [B,N]")
    valid = candidate_valid.bool()
    shared = valid & ((source_ids == SHARED_SOURCE_IDS[0]) | (source_ids == SHARED_SOURCE_IDS[1]))
    if not shared.any(dim=1).all():
        raise ValueError("every bag must contain a shared-source candidate")
    return shared


def hierarchical_source_smooth_pool(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Normalized LogMeanExp within sources and then across sources.

    Candidate multiplicity is normalized inside each source.  The second
    normalization gives every source present in a bag equal prior mass.
    """

    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B,N]")
    if candidate_valid.shape != candidate_logits.shape or source_ids.shape != candidate_logits.shape:
        raise ValueError("validity/source IDs must align with candidate logits")
    if not np.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(candidate_logits).all():
        raise ValueError("candidate logits must be finite")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every bag must contain a valid candidate")

    bag_values: list[torch.Tensor] = []
    diagnostic_values: list[torch.Tensor] = []
    for row in range(candidate_logits.shape[0]):
        identifiers = torch.unique(source_ids[row, valid[row]], sorted=True)
        per_source: list[torch.Tensor] = []
        for identifier in identifiers:
            members = valid[row] & (source_ids[row] == identifier)
            values = candidate_logits[row, members]
            per_source.append(
                temperature
                * (
                    torch.logsumexp(values / temperature, dim=0)
                    - values.new_tensor(values.numel()).log()
                )
            )
        stacked = torch.stack(per_source)
        diagnostic_values.append(stacked)
        bag_values.append(
            temperature
            * (
                torch.logsumexp(stacked / temperature, dim=0)
                - stacked.new_tensor(stacked.numel()).log()
            )
        )
    return torch.stack(bag_values), diagnostic_values


def hierarchical_source_candidate_weights(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return the exact candidate gradients of hierarchical LogMeanExp.

    For each bag, a candidate receives its within-source softmax weight times
    its source's outer softmax weight.  These are the derivatives of
    :func:`hierarchical_source_smooth_pool` with respect to candidate logits,
    so diagnostics cannot accidentally report flat-pool concentration for the
    hierarchical arm.
    """

    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B,N]")
    if candidate_valid.shape != candidate_logits.shape or source_ids.shape != candidate_logits.shape:
        raise ValueError("validity/source IDs must align with candidate logits")
    if not np.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(candidate_logits).all():
        raise ValueError("candidate logits must be finite")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every bag must contain a valid candidate")

    weights = torch.zeros_like(candidate_logits)
    for row in range(candidate_logits.shape[0]):
        identifiers = torch.unique(source_ids[row, valid[row]], sorted=True)
        member_masks: list[torch.Tensor] = []
        per_source: list[torch.Tensor] = []
        for identifier in identifiers:
            members = valid[row] & (source_ids[row] == identifier)
            values = candidate_logits[row, members]
            member_masks.append(members)
            per_source.append(
                temperature
                * (
                    torch.logsumexp(values / temperature, dim=0)
                    - values.new_tensor(values.numel()).log()
                )
            )
        source_weights = torch.softmax(torch.stack(per_source) / temperature, dim=0)
        for source_weight, members in zip(source_weights, member_masks):
            weights[row, members] = source_weight * torch.softmax(
                candidate_logits[row, members] / temperature,
                dim=0,
            )
    return weights


def negative_bag_instance_loss(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    image_labels: torch.Tensor,
) -> torch.Tensor:
    """Use only the logically certain instance labels from negative bags."""

    if candidate_logits.ndim != 2 or candidate_valid.shape != candidate_logits.shape:
        raise ValueError("candidate logits/validity must share shape [B,N]")
    labels = image_labels.reshape(-1).bool()
    if labels.numel() != candidate_logits.shape[0]:
        raise ValueError("image-label batch size differs from candidate bags")
    valid = candidate_valid.bool()
    negative = (~labels)[:, None] & valid
    if not negative.any():
        return candidate_logits.sum() * 0.0
    values = candidate_logits[negative]
    return F.binary_cross_entropy_with_logits(values, torch.zeros_like(values))


def geometric_continuation_temperature(
    epoch: int,
    total_epochs: int,
    *,
    start: float = 1.0,
    end: float = 0.2,
) -> float:
    if total_epochs < 1 or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch lies outside the continuation schedule")
    if not (np.isfinite(start) and np.isfinite(end)) or start < end or end <= 0:
        raise ValueError("continuation temperatures are invalid")
    if total_epochs == 1:
        return float(end)
    progress = float(epoch - 1) / float(total_epochs - 1)
    return float(start * (end / start) ** progress)


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return ascending average-tie percentile ranks in [0,1]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be one finite nonempty vector")
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


def rank_fusion_scores(
    model_logits: np.ndarray,
    upstream_scores: np.ndarray,
) -> np.ndarray:
    model = np.asarray(model_logits, dtype=np.float64)
    upstream = np.asarray(upstream_scores, dtype=np.float64)
    if model.shape != upstream.shape:
        raise ValueError("model/upstream arrays must align")
    return 0.5 * (
        average_percentile_rank(model) + average_percentile_rank(upstream)
    )


def stable_select(scores: np.ndarray, raw_logits: np.ndarray) -> int:
    """Select by score, then raw logit, then lower frozen local index."""

    values = np.asarray(scores, dtype=np.float64)
    raw = np.asarray(raw_logits, dtype=np.float64)
    if values.ndim != 1 or values.shape != raw.shape or not len(values):
        raise ValueError("selector arrays must be aligned nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(raw).all():
        raise ValueError("selector arrays must be finite")
    return int(max(range(len(values)), key=lambda index: (values[index], raw[index], -index)))


__all__ = [
    "EXTERNAL_SOURCE_ID",
    "SHARED_SOURCE_IDS",
    "average_percentile_rank",
    "geometric_continuation_temperature",
    "hierarchical_source_candidate_weights",
    "hierarchical_source_smooth_pool",
    "negative_bag_instance_loss",
    "rank_fusion_scores",
    "shared_source_validity",
    "stable_select",
]
