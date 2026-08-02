from __future__ import annotations

"""Training/scoring orchestration for the matched S6 label-granularity pair."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_label_granularity import (
    LabelGranularityConfig,
    LabelGranularityResidual,
    coarse_candidate_logits,
    entropy_routed_candidate_logits,
    inverse_sqrt_subtype_weights,
    label_granularity_losses,
)
from models.rad_dino_mask_bag_mil import RadDinoMaskBagMIL, smooth_mil_pool


@dataclass(frozen=True)
class LabelGranularityTrainingConfig:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are invalid")


def initial_residual_state(
    config: LabelGranularityConfig,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = LabelGranularityResidual(config)
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def attach_tumor_type_labels(
    records: Sequence[dict[str, Any]],
    split_rows: Sequence[Mapping[str, str]],
) -> list[int]:
    by_image = {str(row["image_id"]): row for row in split_rows}
    counts = [0] * 9
    if len(by_image) != len(split_rows):
        raise ValueError("split rows contain duplicate image identifiers")
    for record in records:
        image_id = str(record["image_id"])
        row = by_image.get(image_id)
        if row is None:
            raise ValueError(f"split omits cache image {image_id}")
        tumor = int(row["tumor"])
        tumor_type = int(row["tumor_type"])
        if tumor != int(record["label"]) or bool(tumor) != bool(tumor_type):
            raise ValueError(f"image-label mismatch for {image_id}")
        if not 0 <= tumor_type <= 9:
            raise ValueError(f"tumor_type out of range for {image_id}")
        record["tumor_type"] = tumor_type
        if tumor_type:
            counts[tumor_type - 1] += 1
    return counts


def padded_label_granularity_batch(
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    selected = [records[int(index)] for index in indices]
    if not selected:
        raise ValueError("label-granularity batch cannot be empty")
    maximum = max(len(record["candidate_indices"]) for record in selected)
    descriptor_dim = int(np.asarray(selected[0]["descriptors"]).shape[1])
    descriptors = np.zeros((len(selected), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(descriptors)
    valid = np.zeros((len(selected), maximum), dtype=bool)
    base = np.zeros((len(selected), maximum), dtype=np.float32)
    base_flipped = np.zeros_like(base)
    tumor = np.zeros(len(selected), dtype=np.int64)
    tumor_type = np.zeros(len(selected), dtype=np.int64)
    for row_index, record in enumerate(selected):
        original = np.asarray(record["descriptors"], dtype=np.float32)
        mirror = np.asarray(record["flipped_descriptors"], dtype=np.float32)
        count = len(record["candidate_indices"])
        if (
            original.shape != (count, descriptor_dim)
            or mirror.shape != original.shape
            or "tumor_type" not in record
        ):
            raise ValueError("cache record differs from the S6 batch contract")
        descriptors[row_index, :count] = original
        flipped[row_index, :count] = mirror
        valid[row_index, :count] = True
        if "base_candidate_logits" in record:
            base_values = np.asarray(record["base_candidate_logits"], dtype=np.float32)
            base_mirror = np.asarray(
                record["base_flipped_candidate_logits"], dtype=np.float32
            )
            if base_values.shape != (count,) or base_mirror.shape != (count,):
                raise ValueError("frozen base score count differs from cache")
            base[row_index, :count] = base_values
            base_flipped[row_index, :count] = base_mirror
        tumor[row_index] = int(record["label"])
        tumor_type[row_index] = int(record["tumor_type"])
    return {
        "descriptors": torch.from_numpy(descriptors).to(device),
        "flipped_descriptors": torch.from_numpy(flipped).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "base_candidate_logits": torch.from_numpy(base).to(device),
        "base_flipped_candidate_logits": torch.from_numpy(base_flipped).to(device),
        "tumor_labels": torch.from_numpy(tumor).to(device),
        "tumor_type_labels": torch.from_numpy(tumor_type).to(device),
    }


@torch.inference_mode()
def attach_frozen_base_logits(
    records: Sequence[dict[str, Any]],
    frozen_base_scorer: RadDinoMaskBagMIL,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    frozen_base_scorer.requires_grad_(False).eval()
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = padded_label_granularity_batch(records, indices, device=device)
        original, _ = frozen_base_scorer.score_descriptors(
            batch["descriptors"], batch["valid"]
        )
        flipped, _ = frozen_base_scorer.score_descriptors(
            batch["flipped_descriptors"], batch["valid"]
        )
        for offset, record_index in enumerate(indices):
            count = int(batch["valid"][offset].sum().item())
            records[record_index]["base_candidate_logits"] = (
                original[offset, :count].float().cpu().numpy()
            )
            records[record_index]["base_flipped_candidate_logits"] = (
                flipped[offset, :count].float().cpu().numpy()
            )


def _mean_loss_pair(
    model: LabelGranularityResidual,
    batch: Mapping[str, torch.Tensor],
    *,
    class_weights: torch.Tensor,
    config: LabelGranularityConfig,
    hierarchical: bool,
) -> dict[str, torch.Tensor]:
    original = model(batch["descriptors"], batch["valid"])
    flipped = model(batch["flipped_descriptors"], batch["valid"])
    first = label_granularity_losses(
        base_candidate_logits=batch["base_candidate_logits"],
        residuals=original,
        flipped_residuals=flipped,
        candidate_valid=batch["valid"],
        tumor_labels=batch["tumor_labels"],
        tumor_type_labels=batch["tumor_type_labels"],
        subtype_class_weights=class_weights,
        config=config,
        hierarchical=hierarchical,
    )
    second = label_granularity_losses(
        base_candidate_logits=batch["base_flipped_candidate_logits"],
        residuals=flipped,
        flipped_residuals=original,
        candidate_valid=batch["valid"],
        tumor_labels=batch["tumor_labels"],
        tumor_type_labels=batch["tumor_type_labels"],
        subtype_class_weights=class_weights,
        config=config,
        hierarchical=hierarchical,
    )
    return {
        key: 0.5 * (first[key] + second[key])
        for key in (
            "total",
            "binary",
            "pathology",
            "subtype",
            "consistency",
            "drift",
        )
    }


def train_label_granularity_arm(
    records: Sequence[dict[str, Any]],
    *,
    model_config: LabelGranularityConfig,
    training_config: LabelGranularityTrainingConfig,
    subtype_counts: Sequence[int],
    hierarchical: bool,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
) -> tuple[LabelGranularityResidual, list[dict[str, float]]]:
    if len(records) == 0:
        raise ValueError("S6 training records cannot be empty")
    counts = torch.as_tensor(subtype_counts, dtype=torch.float32, device=device)
    class_weights = inverse_sqrt_subtype_weights(counts)
    torch.manual_seed(training_config.seed)
    model = LabelGranularityResidual(model_config).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        torch.manual_seed(training_config.seed * 1000 + epoch)
        generator = np.random.default_rng(training_config.seed + epoch)
        order = generator.permutation(len(records))
        sums = {
            key: 0.0
            for key in (
                "total",
                "binary",
                "pathology",
                "subtype",
                "consistency",
                "drift",
            )
        }
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            batch = padded_label_granularity_batch(records, indices, device=device)
            losses = _mean_loss_pair(
                model,
                batch,
                class_weights=class_weights,
                config=model_config,
                hierarchical=hierarchical,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach().item())
            batches += 1
        row = {"epoch": float(epoch), "batches": float(batches)}
        row.update({key: value / batches for key, value in sums.items()})
        history.append(row)
    return model.eval(), history


@torch.inference_mode()
def score_label_granularity_pair(
    records: Sequence[dict[str, Any]],
    control_model: LabelGranularityResidual,
    hierarchy_model: LabelGranularityResidual,
    *,
    model_config: LabelGranularityConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    control_model.eval()
    hierarchy_model.eval()
    arms: dict[str, list[dict[str, Any]]] = {
        "coarse_control": [],
        "hierarchical_entropy_routed": [],
    }
    diagnostics: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = padded_label_granularity_batch(records, indices, device=device)
        valid = batch["valid"]
        base = 0.5 * (
            batch["base_candidate_logits"]
            + batch["base_flipped_candidate_logits"]
        )
        control_original_residual = control_model(batch["descriptors"], valid)
        control_flipped_residual = control_model(batch["flipped_descriptors"], valid)
        hierarchy_original_residual = hierarchy_model(batch["descriptors"], valid)
        hierarchy_flipped_residual = hierarchy_model(
            batch["flipped_descriptors"], valid
        )
        control_residual = 0.5 * (
            control_original_residual + control_flipped_residual
        )
        hierarchy_residual = 0.5 * (
            hierarchy_original_residual + hierarchy_flipped_residual
        )
        control_logits = coarse_candidate_logits(base, control_residual, valid)
        hierarchy_logits, subtype_logits, predicted_subtype, route = (
            entropy_routed_candidate_logits(
                base,
                hierarchy_residual,
                valid,
                temperature=model_config.bag_temperature,
            )
        )
        hierarchy_coarse = coarse_candidate_logits(base, hierarchy_residual, valid)
        control_bag = smooth_mil_pool(
            control_logits, valid, temperature=model_config.bag_temperature
        )
        hierarchy_bag = smooth_mil_pool(
            hierarchy_coarse, valid, temperature=model_config.bag_temperature
        )

        control_original = coarse_candidate_logits(
            batch["base_candidate_logits"], control_original_residual, valid
        )
        control_flipped = coarse_candidate_logits(
            batch["base_flipped_candidate_logits"],
            control_flipped_residual,
            valid,
        )
        hierarchy_original, _so, _po, _ro = entropy_routed_candidate_logits(
            batch["base_candidate_logits"],
            hierarchy_original_residual,
            valid,
            temperature=model_config.bag_temperature,
        )
        hierarchy_flipped, _sf, _pf, _rf = entropy_routed_candidate_logits(
            batch["base_flipped_candidate_logits"],
            hierarchy_flipped_residual,
            valid,
            temperature=model_config.bag_temperature,
        )
        probabilities = torch.softmax(subtype_logits, dim=1)
        for offset, record_index in enumerate(indices):
            record = records[record_index]
            count = int(valid[offset].sum().item())
            control_values = control_logits[offset, :count].float().cpu().numpy()
            hierarchy_values = hierarchy_logits[offset, :count].float().cpu().numpy()
            control_probability = float(torch.sigmoid(control_bag[offset]).item())
            hierarchy_probability = float(torch.sigmoid(hierarchy_bag[offset]).item())
            arms["coarse_control"].append(
                {
                    "image_id": record["image_id"],
                    "candidate_logits": control_values,
                    "bag_logit": float(control_bag[offset].item()),
                    "bag_probability": control_probability,
                }
            )
            arms["hierarchical_entropy_routed"].append(
                {
                    "image_id": record["image_id"],
                    "candidate_logits": hierarchy_values,
                    "bag_logit": float(hierarchy_bag[offset].item()),
                    "bag_probability": hierarchy_probability,
                }
            )
            diagnostics.append(
                {
                    "image_id": record["image_id"],
                    "candidate_count": count,
                    "tumor": int(record["label"]),
                    "tumor_type": int(record["tumor_type"]),
                    "control_bag_probability": control_probability,
                    "hierarchy_bag_probability": hierarchy_probability,
                    "predicted_tumor_type": int(predicted_subtype[offset].item()) + 1,
                    "predicted_subtype_probability": float(
                        probabilities[offset, predicted_subtype[offset]].item()
                    ),
                    "entropy_route_strength": float(route[offset].item()),
                    "control_selected_local_index": int(control_values.argmax()),
                    "hierarchy_selected_local_index": int(hierarchy_values.argmax()),
                    "control_original_flip_agreement": int(
                        int(control_original[offset, :count].argmax().item())
                        == int(control_flipped[offset, :count].argmax().item())
                    ),
                    "hierarchy_original_flip_agreement": int(
                        int(hierarchy_original[offset, :count].argmax().item())
                        == int(hierarchy_flipped[offset, :count].argmax().item())
                    ),
                }
            )
    return arms, diagnostics


def audit_zero_initialization(
    records: Sequence[dict[str, Any]],
    *,
    model_config: LabelGranularityConfig,
    batch_size: int,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    model = LabelGranularityResidual(model_config).to(device).eval()
    model.load_state_dict(initial_state, strict=True)
    arms, diagnostics = score_label_granularity_pair(
        records,
        model,
        model,
        model_config=model_config,
        batch_size=batch_size,
        device=device,
    )
    maximum_delta = 0.0
    maximum_route = 0.0
    exact_control = 0
    exact_hierarchy = 0
    if not (
        len(records)
        == len(arms["coarse_control"])
        == len(arms["hierarchical_entropy_routed"])
        == len(diagnostics)
    ):
        raise RuntimeError("zero-initialization audit record counts differ")
    for record, control, hierarchy, diagnostic in zip(
        records,
        arms["coarse_control"],
        arms["hierarchical_entropy_routed"],
        diagnostics,
    ):
        baseline = 0.5 * (
            np.asarray(record["base_candidate_logits"], dtype=np.float32)
            + np.asarray(record["base_flipped_candidate_logits"], dtype=np.float32)
        )
        control_values = np.asarray(control["candidate_logits"], dtype=np.float32)
        hierarchy_values = np.asarray(hierarchy["candidate_logits"], dtype=np.float32)
        control_delta = float(np.max(np.abs(control_values - baseline)))
        hierarchy_delta = float(np.max(np.abs(hierarchy_values - baseline)))
        maximum_delta = max(maximum_delta, control_delta, hierarchy_delta)
        exact_control += int(np.array_equal(control_values, baseline))
        exact_hierarchy += int(np.array_equal(hierarchy_values, baseline))
        maximum_route = max(
            maximum_route, float(diagnostic["entropy_route_strength"])
        )
    return {
        "records": len(records),
        "exact_control_candidate_score_records": exact_control,
        "exact_hierarchy_candidate_score_records": exact_hierarchy,
        "maximum_candidate_score_delta": maximum_delta,
        "maximum_zero_init_entropy_route_strength": maximum_route,
    }


__all__ = [
    "LabelGranularityTrainingConfig",
    "attach_frozen_base_logits",
    "attach_tumor_type_labels",
    "audit_zero_initialization",
    "initial_residual_state",
    "padded_label_granularity_batch",
    "score_label_granularity_pair",
    "train_label_granularity_arm",
]
