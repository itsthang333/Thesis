from __future__ import annotations

"""Proposal-cluster primitives for cross-fitted image-label-only MIL.

Teacher logits must be produced out of fold; the caller owns and audits that
provenance. This module never reads data splits, annotations or subgroups.
"""

import torch


def build_teacher_proposal_clusters(
    teacher_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    overlap: torch.Tensor,
    *,
    maximum_clusters: int,
    minimum_overlap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily form disjoint clusters around high-scoring teacher seeds."""

    if teacher_logits.ndim != 2 or candidate_valid.shape != teacher_logits.shape:
        raise ValueError("teacher logits/validity must share shape [B,N]")
    batch, candidates = teacher_logits.shape
    if overlap.shape != (batch, candidates, candidates):
        raise ValueError("overlap must have shape [B,N,N]")
    if maximum_clusters < 1 or not 0.0 <= minimum_overlap <= 1.0:
        raise ValueError("cluster controls are invalid")
    if not torch.isfinite(teacher_logits).all() or not torch.isfinite(
        overlap
    ).all():
        raise ValueError("teacher logits and overlap must be finite")
    if (overlap < 0).any() or (overlap > 1).any():
        raise ValueError("overlap must lie in [0,1]")
    if not torch.allclose(overlap, overlap.transpose(1, 2), atol=1.0e-6):
        raise ValueError("overlap must be symmetric")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every bag must contain a valid candidate")

    clusters = torch.zeros(
        (batch, maximum_clusters, candidates),
        dtype=torch.bool,
        device=teacher_logits.device,
    )
    cluster_valid = torch.zeros(
        (batch, maximum_clusters),
        dtype=torch.bool,
        device=teacher_logits.device,
    )
    seed_indices = torch.full(
        (batch, maximum_clusters),
        -1,
        dtype=torch.long,
        device=teacher_logits.device,
    )
    for row in range(batch):
        order = torch.argsort(
            teacher_logits[row].masked_fill(~valid[row], -torch.inf),
            descending=True,
            stable=True,
        )
        assigned = torch.zeros(
            candidates,
            dtype=torch.bool,
            device=teacher_logits.device,
        )
        cluster_index = 0
        for seed_tensor in order:
            seed = int(seed_tensor.item())
            if not valid[row, seed] or assigned[seed]:
                continue
            members = (
                valid[row]
                & ~assigned
                & (overlap[row, seed] >= minimum_overlap)
            )
            members[seed] = True
            clusters[row, cluster_index] = members
            cluster_valid[row, cluster_index] = True
            seed_indices[row, cluster_index] = seed
            assigned |= members
            cluster_index += 1
            if cluster_index == maximum_clusters:
                break
    return clusters, cluster_valid, seed_indices


def proposal_cluster_smooth_pool(
    candidate_logits: torch.Tensor,
    clusters: torch.Tensor,
    cluster_valid: torch.Tensor,
    *,
    within_temperature: float,
    between_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool candidates within OOF-seeded clusters, then clusters within a bag."""

    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B,N]")
    batch, candidates = candidate_logits.shape
    if clusters.ndim != 3 or clusters.shape[0] != batch or clusters.shape[2] != candidates:
        raise ValueError("clusters must have shape [B,K,N]")
    if cluster_valid.shape != clusters.shape[:2]:
        raise ValueError("cluster_valid must have shape [B,K]")
    if within_temperature <= 0 or between_temperature <= 0:
        raise ValueError("pooling temperatures must be positive")
    if not torch.isfinite(candidate_logits).all():
        raise ValueError("candidate logits must be finite")
    members = clusters.bool()
    valid_clusters = cluster_valid.bool()
    if not valid_clusters.any(dim=1).all():
        raise ValueError("every bag must contain at least one cluster")
    if not (members.any(dim=2) == valid_clusters).all():
        raise ValueError("cluster membership and validity disagree")

    expanded_logits = candidate_logits[:, None, :].expand_as(
        members.to(candidate_logits.dtype)
    )
    scaled = (expanded_logits / within_temperature).masked_fill(
        ~members,
        -torch.inf,
    )
    counts = members.sum(dim=2).clamp_min(1).to(candidate_logits.dtype)
    cluster_logits = within_temperature * (
        torch.logsumexp(scaled, dim=2) - counts.log()
    )
    cluster_logits = cluster_logits.masked_fill(~valid_clusters, 0.0)
    outer_scaled = (cluster_logits / between_temperature).masked_fill(
        ~valid_clusters,
        -torch.inf,
    )
    cluster_counts = valid_clusters.sum(dim=1).to(candidate_logits.dtype)
    bag_logits = between_temperature * (
        torch.logsumexp(outer_scaled, dim=1) - cluster_counts.log()
    )
    return cluster_logits, bag_logits


def continuation_temperature(
    epoch: int,
    total_epochs: int,
    *,
    start_temperature: float,
    end_temperature: float,
) -> float:
    """Geometrically sharpen a cluster loss without a hard early threshold."""

    if total_epochs < 1 or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch lies outside the continuation schedule")
    if start_temperature <= 0 or end_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if start_temperature < end_temperature:
        raise ValueError("continuation must sharpen rather than soften")
    if total_epochs == 1:
        return float(end_temperature)
    progress = float(epoch - 1) / float(total_epochs - 1)
    return float(
        start_temperature
        * (end_temperature / start_temperature) ** progress
    )


__all__ = [
    "build_teacher_proposal_clusters",
    "continuation_temperature",
    "proposal_cluster_smooth_pool",
]
