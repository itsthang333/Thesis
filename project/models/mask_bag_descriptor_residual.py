from __future__ import annotations

"""Zero-initialized auxiliary-descriptor residual for proposal logits."""

import torch
from torch import nn


class AuxiliaryDescriptorResidual(nn.Module):
    """Add auxiliary proposal evidence while preserving frozen base logits."""

    def __init__(
        self,
        *,
        base_descriptor_dim: int,
        auxiliary_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if base_descriptor_dim <= 0 or auxiliary_dim <= 0 or hidden_dim < 2:
            raise ValueError("descriptor and hidden dimensions must be positive")
        self.base_descriptor_dim = int(base_descriptor_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.hidden_dim = int(hidden_dim)
        self.base_projection = nn.Sequential(
            nn.LayerNorm(self.base_descriptor_dim),
            nn.Linear(self.base_descriptor_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.auxiliary_projection = nn.Sequential(
            nn.LayerNorm(self.auxiliary_dim),
            nn.Linear(self.auxiliary_dim, self.hidden_dim),
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
        base_descriptors: torch.Tensor,
        auxiliary_features: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            base_descriptors.ndim != 3
            or base_descriptors.shape[-1] != self.base_descriptor_dim
        ):
            raise ValueError("base_descriptors have an invalid shape")
        if (
            auxiliary_features.ndim != 3
            or auxiliary_features.shape[:2] != base_descriptors.shape[:2]
            or auxiliary_features.shape[-1] != self.auxiliary_dim
        ):
            raise ValueError("auxiliary_features have an invalid shape")
        if base_logits.shape != base_descriptors.shape[:2]:
            raise ValueError("base_logits must align with descriptors")
        if candidate_valid.shape != base_logits.shape:
            raise ValueError("candidate_valid must align with base logits")
        if (
            not torch.isfinite(base_descriptors).all()
            or not torch.isfinite(auxiliary_features).all()
            or not torch.isfinite(base_logits).all()
        ):
            raise ValueError("descriptor residual inputs must be finite")
        valid = candidate_valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("every bag must contain at least one valid candidate")

        base = self.base_projection(base_descriptors)
        auxiliary = self.auxiliary_projection(auxiliary_features)
        cosine = torch.nn.functional.cosine_similarity(
            base,
            auxiliary,
            dim=-1,
            eps=1.0e-8,
        )[:, :, None]
        relation = torch.cat(
            (
                base,
                auxiliary,
                base - auxiliary,
                base * auxiliary,
                cosine,
            ),
            dim=-1,
        )
        residual = self.residual(relation).squeeze(-1)
        residual = residual * valid.to(residual.dtype)
        combined = (base_logits + residual) * valid.to(base_logits.dtype)
        return combined, residual


__all__ = ["AuxiliaryDescriptorResidual"]
