from __future__ import annotations

"""Training and scoring for prospective S7 constrained instance learning."""

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from models.mask_bag_global_local_instance import (
    GlobalLocalInstanceConfig,
    GlobalLocalInstanceResidual,
    adaptive_positive_mass,
    build_global_local_soft_targets,
    combined_instance_logits,
    global_local_instance_losses,
)
from models.rad_dino_mask_bag_mil import smooth_mil_pool


@dataclass(frozen=True)
class GlobalLocalInstanceTrainingConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")


def initial_global_local_state(
    config: GlobalLocalInstanceConfig,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = GlobalLocalInstanceResidual(config)
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _validate_record(record: Mapping[str, Any]) -> int:
    required = {
        "image_id",
        "label",
        "descriptors",
        "flipped_descriptors",
        "candidate_indices",
        "family_ids",
        "base_candidate_logits",
        "base_flipped_candidate_logits",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"S7 record misses required keys: {sorted(missing)}")
    count = len(record["candidate_indices"])
    original = np.asarray(record["descriptors"])
    flipped = np.asarray(record["flipped_descriptors"])
    families = np.asarray(record["family_ids"]).reshape(-1)
    base = np.asarray(record["base_candidate_logits"]).reshape(-1)
    base_flipped = np.asarray(record["base_flipped_candidate_logits"]).reshape(-1)
    if count <= 0:
        raise ValueError("S7 candidate bags cannot be empty")
    if (
        original.ndim != 2
        or original.shape[0] != count
        or flipped.shape != original.shape
        or families.shape != (count,)
        or base.shape != (count,)
        or base_flipped.shape != (count,)
    ):
        raise ValueError("S7 cache record arrays do not align")
    if not (
        np.isfinite(original).all()
        and np.isfinite(flipped).all()
        and np.isfinite(base).all()
        and np.isfinite(base_flipped).all()
    ):
        raise ValueError("S7 cache record values must be finite")
    if int(record["label"]) not in (0, 1):
        raise ValueError("S7 image labels must be binary")
    return count


def padded_global_local_batch(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    device: torch.device,
    require_targets: bool,
) -> dict[str, torch.Tensor]:
    selected = [records[int(index)] for index in indices]
    if not selected:
        raise ValueError("S7 batch cannot be empty")
    counts = [_validate_record(record) for record in selected]
    maximum = max(counts)
    descriptor_dim = int(np.asarray(selected[0]["descriptors"]).shape[1])
    original = np.zeros((len(selected), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(original)
    valid = np.zeros((len(selected), maximum), dtype=bool)
    base = np.zeros((len(selected), maximum), dtype=np.float32)
    base_flipped = np.zeros_like(base)
    labels = np.zeros(len(selected), dtype=np.int64)
    targets = np.zeros((len(selected), maximum), dtype=np.float32)
    weights = np.zeros_like(targets)
    for row_index, (record, count) in enumerate(zip(selected, counts)):
        descriptors = np.asarray(record["descriptors"], dtype=np.float32)
        mirror = np.asarray(record["flipped_descriptors"], dtype=np.float32)
        if descriptors.shape[1] != descriptor_dim:
            raise ValueError("S7 descriptor dimension differs across records")
        original[row_index, :count] = descriptors
        flipped[row_index, :count] = mirror
        valid[row_index, :count] = True
        base[row_index, :count] = np.asarray(
            record["base_candidate_logits"], dtype=np.float32
        )
        base_flipped[row_index, :count] = np.asarray(
            record["base_flipped_candidate_logits"], dtype=np.float32
        )
        labels[row_index] = int(record["label"])
        if require_targets:
            if "s7_soft_targets" not in record or "s7_candidate_weights" not in record:
                raise ValueError("S7 training batch requires frozen soft targets")
            target = np.asarray(record["s7_soft_targets"], dtype=np.float32)
            weight = np.asarray(record["s7_candidate_weights"], dtype=np.float32)
            if target.shape != (count,) or weight.shape != (count,):
                raise ValueError("S7 target/weight arrays do not align with candidates")
            targets[row_index, :count] = target
            weights[row_index, :count] = weight
    result = {
        "descriptors": torch.from_numpy(original).to(device),
        "flipped_descriptors": torch.from_numpy(flipped).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "base_candidate_logits": torch.from_numpy(base).to(device),
        "base_flipped_candidate_logits": torch.from_numpy(base_flipped).to(device),
        "labels": torch.from_numpy(labels).to(device),
    }
    if require_targets:
        result["soft_targets"] = torch.from_numpy(targets).to(device)
        result["candidate_weights"] = torch.from_numpy(weights).to(device)
    return result


@torch.inference_mode()
def score_current_instance_logits(
    records: Sequence[Mapping[str, Any]],
    model: GlobalLocalInstanceResidual,
    *,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = padded_global_local_batch(
            records, indices, device=device, require_targets=False
        )
        original_residual = model(batch["descriptors"], batch["valid"])
        flipped_residual = model(batch["flipped_descriptors"], batch["valid"])
        original = combined_instance_logits(
            batch["base_candidate_logits"], original_residual, batch["valid"]
        )
        flipped = combined_instance_logits(
            batch["base_flipped_candidate_logits"],
            flipped_residual,
            batch["valid"],
        )
        averaged = 0.5 * (original + flipped)
        for offset in range(len(indices)):
            count = int(batch["valid"][offset].sum().item())
            result.append(averaged[offset, :count].float().cpu().numpy())
    if len(result) != len(records):
        raise RuntimeError("S7 scoring did not cover every record")
    return result


def _target_sha256(
    records: Sequence[Mapping[str, Any]],
    targets: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    if not (len(records) == len(targets) == len(weights)):
        raise ValueError("S7 target hash inputs must align")
    for record, target, weight in zip(records, targets, weights):
        image_id = str(record["image_id"]).encode("utf-8")
        target32 = np.asarray(target, dtype="<f4")
        weight64 = np.asarray(weight, dtype="<f8")
        digest.update(len(image_id).to_bytes(4, "little"))
        digest.update(image_id)
        digest.update(target32.shape[0].to_bytes(4, "little"))
        digest.update(target32.tobytes(order="C"))
        digest.update(weight64.tobytes(order="C"))
    return digest.hexdigest()


def assign_global_local_targets(
    records: Sequence[dict[str, Any]],
    model: GlobalLocalInstanceResidual,
    *,
    epoch_index: int,
    model_config: GlobalLocalInstanceConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    logits = score_current_instance_logits(
        records, model, batch_size=batch_size, device=device
    )
    target_mass = adaptive_positive_mass(epoch_index, model_config)
    targets, weights, diagnostics = build_global_local_soft_targets(
        logits,
        [int(record["label"]) for record in records],
        [np.asarray(record["family_ids"]).reshape(-1) for record in records],
        target_mass=target_mass,
    )
    for record, target, weight in zip(records, targets, weights):
        record["s7_soft_targets"] = np.asarray(target, dtype=np.float32)
        record["s7_candidate_weights"] = np.asarray(weight, dtype=np.float64)
    diagnostics = dict(diagnostics)
    diagnostics["epoch_index"] = int(epoch_index)
    diagnostics["target_sha256"] = _target_sha256(records, targets, weights)
    return diagnostics


def train_global_local_instance(
    records: Sequence[dict[str, Any]],
    *,
    model_config: GlobalLocalInstanceConfig,
    training_config: GlobalLocalInstanceTrainingConfig,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
) -> tuple[GlobalLocalInstanceResidual, list[dict[str, Any]]]:
    if len(records) == 0:
        raise ValueError("S7 training records cannot be empty")
    if training_config.epochs != model_config.total_epochs:
        raise ValueError("S7 training/model epoch contracts must match")
    torch.manual_seed(training_config.seed)
    model = GlobalLocalInstanceResidual(model_config).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, training_config.epochs + 1):
        target_diagnostics = assign_global_local_targets(
            records,
            model,
            epoch_index=epoch - 1,
            model_config=model_config,
            batch_size=training_config.batch_size,
            device=device,
        )
        torch.manual_seed(training_config.seed * 1000 + epoch)
        generator = np.random.default_rng(training_config.seed + epoch)
        order = generator.permutation(len(records))
        model.train()
        sums = {key: 0.0 for key in ("total", "instance", "consistency", "drift")}
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            batch = padded_global_local_batch(
                records, indices, device=device, require_targets=True
            )
            original_residual = model(batch["descriptors"], batch["valid"])
            flipped_residual = model(
                batch["flipped_descriptors"], batch["valid"]
            )
            losses = global_local_instance_losses(
                original_logits=combined_instance_logits(
                    batch["base_candidate_logits"],
                    original_residual,
                    batch["valid"],
                ),
                flipped_logits=combined_instance_logits(
                    batch["base_flipped_candidate_logits"],
                    flipped_residual,
                    batch["valid"],
                ),
                original_residuals=original_residual,
                flipped_residuals=flipped_residual,
                soft_targets=batch["soft_targets"],
                candidate_weights=batch["candidate_weights"],
                candidate_valid=batch["valid"],
                config=model_config,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach().item())
            batches += 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "batches": batches,
            "target": target_diagnostics,
        }
        row.update({key: value / batches for key, value in sums.items()})
        history.append(row)
    return model.eval(), history


@torch.inference_mode()
def score_global_local_instance(
    records: Sequence[Mapping[str, Any]],
    model: GlobalLocalInstanceResidual,
    *,
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if bag_temperature <= 0.0:
        raise ValueError("accepted bag temperature must be positive")
    model.eval()
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = padded_global_local_batch(
            records, indices, device=device, require_targets=False
        )
        original_residual = model(batch["descriptors"], batch["valid"])
        flipped_residual = model(batch["flipped_descriptors"], batch["valid"])
        original = combined_instance_logits(
            batch["base_candidate_logits"], original_residual, batch["valid"]
        )
        flipped = combined_instance_logits(
            batch["base_flipped_candidate_logits"],
            flipped_residual,
            batch["valid"],
        )
        combined = 0.5 * (original + flipped)
        accepted_base = 0.5 * (
            batch["base_candidate_logits"]
            + batch["base_flipped_candidate_logits"]
        )
        accepted_bag = smooth_mil_pool(
            accepted_base,
            batch["valid"],
            temperature=bag_temperature,
        )
        for offset, record_index in enumerate(indices):
            count = int(batch["valid"][offset].sum().item())
            values = combined[offset, :count].float().cpu().numpy()
            base_values = accepted_base[offset, :count].float().cpu().numpy()
            outputs.append(
                {
                    "image_id": str(records[record_index]["image_id"]),
                    "candidate_logits": values,
                    "base_candidate_logits": base_values,
                    "residual_logits": (values - base_values).astype(np.float32),
                    "selected_local_index": int(np.argmax(values)),
                    "base_selected_local_index": int(np.argmax(base_values)),
                    "bag_logit": float(accepted_bag[offset].item()),
                    "bag_probability": float(
                        torch.sigmoid(accepted_bag[offset]).item()
                    ),
                    "original_flip_agreement": int(
                        int(original[offset, :count].argmax().item())
                        == int(flipped[offset, :count].argmax().item())
                    ),
                }
            )
    if len(outputs) != len(records):
        raise RuntimeError("S7 final scoring did not cover every record")
    return outputs


def audit_zero_initialization(
    records: Sequence[Mapping[str, Any]],
    *,
    model_config: GlobalLocalInstanceConfig,
    initial_state: Mapping[str, torch.Tensor],
    bag_temperature: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model = GlobalLocalInstanceResidual(model_config).to(device).eval()
    model.load_state_dict(initial_state, strict=True)
    outputs = score_global_local_instance(
        records,
        model,
        bag_temperature=bag_temperature,
        batch_size=batch_size,
        device=device,
    )
    exact = 0
    selected = 0
    maximum_delta = 0.0
    for record, output in zip(records, outputs):
        expected = 0.5 * (
            np.asarray(record["base_candidate_logits"], dtype=np.float32)
            + np.asarray(record["base_flipped_candidate_logits"], dtype=np.float32)
        )
        actual = np.asarray(output["candidate_logits"], dtype=np.float32)
        exact += int(np.array_equal(actual, expected))
        selected += int(
            int(output["selected_local_index"])
            == int(output["base_selected_local_index"])
        )
        maximum_delta = max(maximum_delta, float(np.max(np.abs(actual - expected))))
    return {
        "records": len(records),
        "exact_candidate_score_records": exact,
        "exact_selected_index_records": selected,
        "maximum_candidate_score_delta": maximum_delta,
    }


__all__ = [
    "GlobalLocalInstanceTrainingConfig",
    "assign_global_local_targets",
    "audit_zero_initialization",
    "initial_global_local_state",
    "padded_global_local_batch",
    "score_current_instance_logits",
    "score_global_local_instance",
    "train_global_local_instance",
]
