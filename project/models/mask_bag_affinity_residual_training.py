from __future__ import annotations

"""Image-label-only R2 residual fitting from cached RAD-DINO affinity."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_normal_residual_training import (
    NormalResidualTrainingConfig,
    score_normal_residual_records,
    train_normal_residual_adapter,
)
from models.mask_bag_residual_objective import ResidualObjectiveConfig


AFFINITY_DIM = 24


@dataclass(frozen=True)
class AffinityResidualTrainingConfig:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    adapter_hidden_dim: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer controls are invalid")
        if self.adapter_hidden_dim < 2:
            raise ValueError("adapter_hidden_dim must be at least two")


def attach_cached_affinity_features(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose immutable original/flip affinity under the shared adapter keys."""

    enriched: list[dict[str, Any]] = []
    for record in records:
        descriptors = np.asarray(record["descriptors"])
        flipped_descriptors = np.asarray(record["flipped_descriptors"])
        affinity = np.asarray(record["affinity_features"], dtype=np.float32)
        flipped_affinity = np.asarray(
            record["flipped_affinity_features"], dtype=np.float32
        )
        expected = (len(descriptors), AFFINITY_DIM)
        if (
            descriptors.ndim != 2
            or descriptors.shape != flipped_descriptors.shape
            or affinity.shape != expected
            or flipped_affinity.shape != expected
            or not np.isfinite(affinity).all()
            or not np.isfinite(flipped_affinity).all()
        ):
            raise ValueError("cached affinity does not align with descriptors")
        enriched.append(
            {
                **record,
                "auxiliary_features": affinity,
                "flipped_auxiliary_features": flipped_affinity,
            }
        )
    return enriched


def _shared_config(
    config: AffinityResidualTrainingConfig,
) -> NormalResidualTrainingConfig:
    # prototype_temperature is unused by the shared residual fitter. It is
    # assigned a finite sentinel solely because the R1 configuration object
    # also owns prototype construction.
    return NormalResidualTrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        prototype_temperature=1.0,
        adapter_hidden_dim=config.adapter_hidden_dim,
        seed=config.seed,
    )


def train_affinity_residual_adapter(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    *,
    descriptor_dim: int,
    objective_config: ResidualObjectiveConfig,
    training_config: AffinityResidualTrainingConfig,
    device: torch.device,
    initial_adapter_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fit the zero-initialized R2 adapter without positive-instance targets."""

    return train_normal_residual_adapter(
        attach_cached_affinity_features(records),
        frozen_base_scorer,
        descriptor_dim=descriptor_dim,
        objective_config=objective_config,
        training_config=_shared_config(training_config),
        device=device,
        initial_adapter_state=initial_adapter_state,
        auxiliary_dim=AFFINITY_DIM,
    )


def score_affinity_residual_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: nn.Module,
    *,
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score every immutable candidate with aligned original/flip affinity."""

    return score_normal_residual_records(
        attach_cached_affinity_features(records),
        frozen_base_scorer,
        adapter,
        bag_temperature=bag_temperature,
        batch_size=batch_size,
        device=device,
    )


__all__ = [
    "AFFINITY_DIM",
    "AffinityResidualTrainingConfig",
    "attach_cached_affinity_features",
    "score_affinity_residual_records",
    "train_affinity_residual_adapter",
]
