from __future__ import annotations

"""Train-label-only normal-prototype residual fitting for immutable bags."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_descriptor_residual import AuxiliaryDescriptorResidual
from models.mask_bag_normal_prototypes import (
    fit_weighted_spherical_prototypes,
    hierarchical_image_family_weights,
    normal_prototype_features,
)
from models.mask_bag_residual_objective import (
    ResidualObjectiveConfig,
    residual_arm_objective,
)
from models.rad_dino_mask_bag_mil import smooth_mil_pool


@dataclass(frozen=True)
class NormalResidualTrainingConfig:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    prototype_temperature: float = 0.10
    adapter_hidden_dim: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer controls are invalid")
        if self.prototype_temperature <= 0 or self.adapter_hidden_dim < 2:
            raise ValueError("prototype/adapter controls are invalid")


def fit_normal_prototype_bank(
    records: Sequence[Mapping[str, Any]],
    *,
    prototype_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one flip-symmetric bank from image-label-normal training bags."""

    normal = [record for record in records if int(record["label"]) == 0]
    if not normal:
        raise ValueError("normal prototype fitting requires normal image bags")
    descriptors: list[np.ndarray] = []
    image_ids: list[str] = []
    family_ids: list[np.ndarray] = []
    for record in normal:
        original = np.asarray(record["descriptors"], dtype=np.float32)
        flipped = np.asarray(record["flipped_descriptors"], dtype=np.float32)
        families = np.asarray(record["family_ids"], dtype=np.int32)
        if (
            original.ndim != 2
            or original.shape != flipped.shape
            or families.shape != (original.shape[0],)
            or not np.isfinite(original).all()
            or not np.isfinite(flipped).all()
        ):
            raise ValueError("normal prototype record has invalid aligned arrays")
        # Both views share the same image/family identity, so hierarchical
        # weighting gives an image equal total mass rather than double weight.
        descriptors.extend((original, flipped))
        image_ids.extend([str(record["image_id"])] * (2 * len(original)))
        family_ids.extend((families, families))
    values = np.concatenate(descriptors, axis=0)
    images = np.asarray(image_ids, dtype="U128")
    families = np.concatenate(family_ids, axis=0)
    weights = hierarchical_image_family_weights(images, families)
    prototypes, assignments = fit_weighted_spherical_prototypes(
        values,
        weights,
        prototype_count=prototype_count,
        seed=seed,
    )
    return prototypes, {
        "prototype_count": int(prototype_count),
        "normal_images": len(normal),
        "normal_candidate_views": int(len(values)),
        "descriptor_dim": int(values.shape[1]),
        "hierarchical_weight_sum": float(weights.sum()),
        "nonempty_clusters": int(len(np.unique(assignments))),
        "original_and_flip_share_image_family_weight": True,
        "segmentation_quality_used": False,
    }


def attach_normal_prototype_features(
    records: Sequence[Mapping[str, Any]],
    prototypes: np.ndarray,
    *,
    temperature: float,
) -> list[dict[str, Any]]:
    """Attach four normality features to both aligned descriptor views."""

    enriched: list[dict[str, Any]] = []
    for record in records:
        original = np.asarray(record["descriptors"], dtype=np.float32)
        flipped = np.asarray(record["flipped_descriptors"], dtype=np.float32)
        if original.shape != flipped.shape:
            raise ValueError("original/flip descriptor shapes differ")
        enriched.append(
            {
                **record,
                "auxiliary_features": normal_prototype_features(
                    original, prototypes, temperature=temperature
                ),
                "flipped_auxiliary_features": normal_prototype_features(
                    flipped, prototypes, temperature=temperature
                ),
            }
        )
    return enriched


def _padded_batch(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    selected = [records[int(index)] for index in indices]
    maximum = max(len(record["descriptors"]) for record in selected)
    descriptor_dim = int(np.asarray(selected[0]["descriptors"]).shape[1])
    auxiliary_dim = int(np.asarray(selected[0]["auxiliary_features"]).shape[1])
    original = np.zeros((len(selected), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(original)
    auxiliary = np.zeros(
        (len(selected), maximum, auxiliary_dim), dtype=np.float32
    )
    flipped_auxiliary = np.zeros_like(auxiliary)
    valid = np.zeros((len(selected), maximum), dtype=bool)
    labels = np.zeros(len(selected), dtype=np.float32)
    for row_index, record in enumerate(selected):
        count = len(record["descriptors"])
        arrays = (
            np.asarray(record["descriptors"], dtype=np.float32),
            np.asarray(record["flipped_descriptors"], dtype=np.float32),
            np.asarray(record["auxiliary_features"], dtype=np.float32),
            np.asarray(record["flipped_auxiliary_features"], dtype=np.float32),
        )
        if (
            arrays[0].shape != (count, descriptor_dim)
            or arrays[1].shape != arrays[0].shape
            or arrays[2].shape != (count, auxiliary_dim)
            or arrays[3].shape != arrays[2].shape
            or not all(np.isfinite(array).all() for array in arrays)
        ):
            raise ValueError("residual-training record arrays are invalid")
        original[row_index, :count] = arrays[0]
        flipped[row_index, :count] = arrays[1]
        auxiliary[row_index, :count] = arrays[2]
        flipped_auxiliary[row_index, :count] = arrays[3]
        valid[row_index, :count] = True
        labels[row_index] = float(record["label"])
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (
            original,
            flipped,
            auxiliary,
            flipped_auxiliary,
            valid,
            labels,
        )
    )


def train_normal_residual_adapter(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    *,
    descriptor_dim: int,
    objective_config: ResidualObjectiveConfig,
    training_config: NormalResidualTrainingConfig,
    device: torch.device,
    initial_adapter_state: Mapping[str, torch.Tensor] | None = None,
    auxiliary_dim: int = 4,
) -> tuple[AuxiliaryDescriptorResidual, list[dict[str, float]]]:
    """Fit only the residual adapter; the independent scorer stays frozen."""

    if not records:
        raise ValueError("adapter training records cannot be empty")
    frozen_base_scorer.requires_grad_(False).eval()
    if auxiliary_dim <= 0:
        raise ValueError("auxiliary_dim must be positive")
    if initial_adapter_state is None:
        torch.manual_seed(training_config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(training_config.seed)
    adapter = AuxiliaryDescriptorResidual(
        base_descriptor_dim=descriptor_dim,
        auxiliary_dim=auxiliary_dim,
        hidden_dim=training_config.adapter_hidden_dim,
    ).to(device)
    if initial_adapter_state is not None:
        adapter.load_state_dict(initial_adapter_state, strict=True)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        adapter.train()
        order = np.random.default_rng(training_config.seed + epoch).permutation(
            len(records)
        )
        sums = {
            "total": 0.0,
            "image": 0.0,
            "consistency": 0.0,
            "residual_drift": 0.0,
        }
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            batch_indices = order[start : start + training_config.batch_size]
            (
                original,
                flipped,
                auxiliary,
                flipped_auxiliary,
                valid,
                labels,
            ) = _padded_batch(records, batch_indices, device)
            with torch.inference_mode():
                original_base, _ = frozen_base_scorer.score_descriptors(
                    original, valid
                )
                flipped_base, _ = frozen_base_scorer.score_descriptors(
                    flipped, valid
                )
            original_logits, original_residual = adapter(
                original, auxiliary, original_base, valid
            )
            flipped_logits, flipped_residual = adapter(
                flipped, flipped_auxiliary, flipped_base, valid
            )
            total, details = residual_arm_objective(
                original_base,
                flipped_base,
                original_residual,
                flipped_residual,
                valid,
                labels,
                objective_config,
            )
            # The adapter outputs above must match the objective's detached
            # reconstruction exactly; this catches accidental base-logit drift.
            if not torch.equal(
                original_logits, original_base + original_residual
            ) or not torch.equal(
                flipped_logits, flipped_base + flipped_residual
            ):
                raise RuntimeError("residual adapter changed frozen-base semantics")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key in sums:
                value = total if key == "total" else details[key]
                sums[key] += float(value.detach().item())
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / batches for key, value in sums.items()},
            }
        )
    if any(parameter.requires_grad for parameter in frozen_base_scorer.parameters()):
        raise RuntimeError("base scorer became trainable")
    return adapter, history


def score_normal_residual_records(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    adapter: AuxiliaryDescriptorResidual,
    *,
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return aligned TTA candidate logits and bag scores without fitting."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frozen_base_scorer.requires_grad_(False).eval()
    adapter.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        (
            original,
            flipped,
            auxiliary,
            flipped_auxiliary,
            valid,
            _labels,
        ) = _padded_batch(records, indices, device)
        with torch.inference_mode():
            original_base, _ = frozen_base_scorer.score_descriptors(original, valid)
            flipped_base, _ = frozen_base_scorer.score_descriptors(flipped, valid)
            original_logits, _ = adapter(
                original, auxiliary, original_base, valid
            )
            flipped_logits, _ = adapter(
                flipped, flipped_auxiliary, flipped_base, valid
            )
            logits = 0.5 * (original_logits + flipped_logits)
            bag_logits = smooth_mil_pool(
                logits, valid, temperature=bag_temperature
            )
        for row, record_index in enumerate(indices):
            count = int(valid[row].sum().item())
            output.append(
                {
                    "image_id": records[int(record_index)]["image_id"],
                    "candidate_logits": logits[row, :count]
                    .float()
                    .cpu()
                    .numpy(),
                    "bag_logit": float(bag_logits[row].item()),
                    "bag_probability": float(torch.sigmoid(bag_logits[row]).item()),
                    "candidate_count": count,
                }
            )
    return output


__all__ = [
    "NormalResidualTrainingConfig",
    "attach_normal_prototype_features",
    "fit_normal_prototype_bank",
    "score_normal_residual_records",
    "train_normal_residual_adapter",
]
