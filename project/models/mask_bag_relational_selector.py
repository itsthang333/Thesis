from __future__ import annotations

"""GT-free relational selector primitives for immutable proposal bags.

The functions in this module accept only proposal descriptors, proposal masks,
proposal-family identities and image-label-trained logits.  They deliberately
have no dataset, annotation or lesion-size interface.
"""

import torch
from torch import nn


def _validate_bag_tensors(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    family_ids: torch.Tensor,
) -> torch.Tensor:
    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B,N]")
    if candidate_valid.shape != candidate_logits.shape:
        raise ValueError("candidate_valid must align with candidate_logits")
    if family_ids.shape != candidate_logits.shape:
        raise ValueError("family_ids must align with candidate_logits")
    if not torch.isfinite(candidate_logits).all():
        raise ValueError("candidate_logits must be finite")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every bag must contain at least one valid candidate")
    if (family_ids[valid] < 0).any():
        raise ValueError("Every valid candidate must have a nonnegative family ID")
    return valid


def family_balanced_smooth_mil_pool(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    family_ids: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Hierarchical normalized SmoothMax over candidates then families.

    Candidate multiplicity is normalized inside each immutable proposal family
    before families are normalized at bag level.  The returned family logits
    are diagnostic tensors in sorted family-ID order for each bag.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = _validate_bag_tensors(
        candidate_logits,
        candidate_valid,
        family_ids,
    )
    bag_logits: list[torch.Tensor] = []
    diagnostic_family_logits: list[torch.Tensor] = []
    for row in range(candidate_logits.shape[0]):
        row_valid = valid[row]
        identifiers = torch.unique(family_ids[row, row_valid], sorted=True)
        family_values: list[torch.Tensor] = []
        for identifier in identifiers:
            members = row_valid & (family_ids[row] == identifier)
            values = candidate_logits[row, members]
            family_values.append(
                temperature
                * (
                    torch.logsumexp(values / temperature, dim=0)
                    - values.new_tensor(values.numel()).log()
                )
            )
        stacked = torch.stack(family_values)
        diagnostic_family_logits.append(stacked)
        bag_logits.append(
            temperature
            * (
                torch.logsumexp(stacked / temperature, dim=0)
                - stacked.new_tensor(stacked.numel()).log()
            )
        )
    return torch.stack(bag_logits), diagnostic_family_logits


def build_family_overlap_graph(
    candidate_masks: torch.Tensor,
    candidate_valid: torch.Tensor,
    family_ids: torch.Tensor,
    *,
    minimum_iou: float = 0.25,
    minimum_containment: float = 0.50,
) -> torch.Tensor:
    """Build a symmetric proposal graph without absolute anatomy coordinates.

    Two distinct proposals are adjacent only when they belong to the same
    immutable family and have sufficient IoU or containment.  Valid isolated
    proposals receive a self-loop so subsequent smoothing preserves them.
    """

    if candidate_masks.ndim != 4:
        raise ValueError("candidate_masks must have shape [B,N,H,W]")
    if candidate_masks.shape[:2] != candidate_valid.shape:
        raise ValueError("candidate_valid must align with candidate_masks")
    if not torch.isfinite(candidate_masks).all():
        raise ValueError("candidate_masks must be finite")
    dummy_logits = torch.zeros(
        candidate_valid.shape,
        dtype=torch.float32,
        device=candidate_masks.device,
    )
    valid = _validate_bag_tensors(dummy_logits, candidate_valid, family_ids)
    if not (0.0 <= minimum_iou <= 1.0):
        raise ValueError("minimum_iou must lie in [0,1]")
    if not (0.0 <= minimum_containment <= 1.0):
        raise ValueError("minimum_containment must lie in [0,1]")

    binary = (candidate_masks > 0.5).flatten(start_dim=2).float()
    areas = binary.sum(dim=-1)
    if (areas[valid] <= 0).any():
        raise ValueError("Every valid candidate mask must be nonempty")
    intersections = binary @ binary.transpose(1, 2)
    unions = areas[:, :, None] + areas[:, None, :] - intersections
    iou = intersections / unions.clamp_min(1.0)
    minimum_area = torch.minimum(areas[:, :, None], areas[:, None, :])
    containment = intersections / minimum_area.clamp_min(1.0)
    same_family = family_ids[:, :, None] == family_ids[:, None, :]
    pair_valid = valid[:, :, None] & valid[:, None, :]
    adjacency = (
        pair_valid
        & same_family
        & ((iou >= minimum_iou) | (containment >= minimum_containment))
    )
    identity = torch.eye(
        candidate_masks.shape[1],
        dtype=torch.bool,
        device=candidate_masks.device,
    )[None]
    adjacency &= ~identity

    isolated = valid & ~adjacency.any(dim=-1)
    adjacency |= identity & isolated[:, :, None]
    return adjacency.to(dtype=candidate_masks.dtype)


def smooth_candidate_logits(
    candidate_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    iterations: int = 10,
) -> torch.Tensor:
    """Apply fidelity-preserving normalized-graph smoothing to logits."""

    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B,N]")
    if candidate_valid.shape != candidate_logits.shape:
        raise ValueError("candidate_valid must align with candidate_logits")
    if adjacency.shape != (
        candidate_logits.shape[0],
        candidate_logits.shape[1],
        candidate_logits.shape[1],
    ):
        raise ValueError("adjacency must have shape [B,N,N]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not torch.isfinite(candidate_logits).all() or not torch.isfinite(
        adjacency
    ).all():
        raise ValueError("logits and adjacency must be finite")
    if (adjacency < 0).any():
        raise ValueError("adjacency must be nonnegative")
    if not torch.allclose(adjacency, adjacency.transpose(1, 2)):
        raise ValueError("adjacency must be symmetric")
    valid = candidate_valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every bag must contain at least one valid candidate")

    coefficient = torch.as_tensor(
        alpha,
        dtype=candidate_logits.dtype,
        device=candidate_logits.device,
    )
    if coefficient.numel() != 1 or not torch.isfinite(coefficient):
        raise ValueError("alpha must be one finite scalar")
    if not (0.0 <= float(coefficient.detach()) < 1.0):
        raise ValueError("alpha must lie in [0,1)")

    graph = adjacency * valid[:, :, None] * valid[:, None, :]
    degrees = graph.sum(dim=-1)
    isolated = valid & (degrees <= 0)
    identity = torch.eye(
        candidate_logits.shape[1],
        dtype=graph.dtype,
        device=graph.device,
    )[None]
    graph = graph + identity * isolated[:, :, None]
    degrees = graph.sum(dim=-1).clamp_min(1.0)
    inverse_sqrt = degrees.rsqrt()
    propagation = inverse_sqrt[:, :, None] * graph * inverse_sqrt[:, None, :]

    original = candidate_logits * valid.to(candidate_logits.dtype)
    smoothed = original
    for _ in range(iterations):
        smoothed = coefficient * torch.bmm(
            propagation,
            smoothed[:, :, None],
        )[:, :, 0] + (1.0 - coefficient) * original
        smoothed = smoothed * valid.to(smoothed.dtype)
    return smoothed


class CriticalRelationResidual(nn.Module):
    """Zero-initialized DSMIL-style relation residual for candidate logits."""

    def __init__(self, descriptor_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if descriptor_dim <= 0 or hidden_dim <= 1:
            raise ValueError("descriptor_dim and hidden_dim must be positive")
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim + 1),
            nn.Linear(4 * self.hidden_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        final = self.residual[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        independent_logits: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError("descriptors have an invalid shape")
        if independent_logits.shape != descriptors.shape[:2]:
            raise ValueError("independent_logits must align with descriptors")
        if candidate_valid.shape != independent_logits.shape:
            raise ValueError("candidate_valid must align with logits")
        if not torch.isfinite(descriptors).all() or not torch.isfinite(
            independent_logits
        ).all():
            raise ValueError("descriptors and logits must be finite")
        valid = candidate_valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Every bag must contain at least one valid candidate")

        critical_indices = independent_logits.detach().masked_fill(
            ~valid,
            -torch.inf,
        ).argmax(dim=1)
        embedded = self.embedding(descriptors)
        batch = torch.arange(descriptors.shape[0], device=descriptors.device)
        critical = embedded[batch, critical_indices][:, None]
        expanded = critical.expand_as(embedded)
        cosine = torch.nn.functional.cosine_similarity(
            embedded,
            expanded,
            dim=-1,
            eps=1.0e-8,
        )[:, :, None]
        relation = torch.cat(
            (
                embedded,
                expanded,
                embedded - expanded,
                embedded * expanded,
                cosine,
            ),
            dim=-1,
        )
        residual = self.residual(relation).squeeze(-1)
        residual = residual * valid.to(residual.dtype)
        combined = (independent_logits + residual) * valid.to(
            independent_logits.dtype
        )
        return combined, critical_indices, residual


__all__ = [
    "CriticalRelationResidual",
    "build_family_overlap_graph",
    "family_balanced_smooth_mil_pool",
    "smooth_candidate_logits",
]
