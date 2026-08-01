from __future__ import annotations

"""GT-blind primitives for the count-controlled T1 confirmation arm.

The module consumes candidate descriptors, image labels, group IDs and frozen
OOF producer scores.  It deliberately exposes no segmentation target, lesion
size or validation-evaluation interface.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_crossfit import audit_crossfit_training_exclusion
from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    smooth_mil_pool,
)


@dataclass(frozen=True)
class CountControlledSelfPacedConfig:
    fold_count: int = 5
    producer_epochs: int = 16
    producer_batch_size: int = 16
    producer_learning_rate: float = 3.0e-4
    producer_weight_decay: float = 1.0e-4
    view_consistency_weight: float = 0.10
    count_independence_weight: float = 1.0
    maximum_count_spearman: float = 0.5013777759365411
    minimum_oof_auroc: float = 0.75
    minimum_view_agreement: float = 0.60
    pace_fractions: tuple[float, ...] = (0.20, 0.40, 0.60)
    consumer_epochs: int = 12
    consumer_learning_rate: float = 1.0e-4
    supervised_contrastive_weight: float = 0.25
    residual_hidden_dim: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.fold_count < 2 or self.producer_epochs < 1:
            raise ValueError("T1 producer fold/epoch controls are invalid")
        if self.producer_batch_size < 2 or self.consumer_epochs < 1:
            raise ValueError("T1 batch/consumer controls are invalid")
        if self.producer_learning_rate <= 0 or self.consumer_learning_rate <= 0:
            raise ValueError("T1 learning rates must be positive")
        if self.producer_weight_decay < 0:
            raise ValueError("T1 weight decay must be nonnegative")
        for name, value in (
            ("view consistency", self.view_consistency_weight),
            ("count independence", self.count_independence_weight),
            ("supervised contrastive", self.supervised_contrastive_weight),
        ):
            if value < 0:
                raise ValueError(f"T1 {name} weight must be nonnegative")
        if not 0.0 <= self.maximum_count_spearman <= 1.0:
            raise ValueError("T1 count-Spearman ceiling is invalid")
        if not 0.0 <= self.minimum_oof_auroc <= 1.0:
            raise ValueError("T1 AUROC floor is invalid")
        if not 0.0 <= self.minimum_view_agreement <= 1.0:
            raise ValueError("T1 view-agreement floor is invalid")
        if not self.pace_fractions or any(
            not 0.0 < value <= 1.0 for value in self.pace_fractions
        ):
            raise ValueError("T1 pace fractions must lie in (0, 1]")
        if tuple(sorted(set(self.pace_fractions))) != self.pace_fractions:
            raise ValueError("T1 pace fractions must be unique and increasing")
        if self.residual_hidden_dim < 2:
            raise ValueError("T1 residual hidden dimension is invalid")


class CountControlledResidual(nn.Module):
    """A zero-initialized candidate-logit residual over the accepted scorer."""

    def __init__(self, descriptor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if descriptor_dim < 1 or hidden_dim < 2:
            raise ValueError("T1 residual dimensions are invalid")
        self.network = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        base_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or base_logits.shape != descriptors.shape[:2]:
            raise ValueError("T1 descriptors/base logits must align")
        if valid.shape != base_logits.shape:
            raise ValueError("T1 validity mask must align with logits")
        residual = self.network(descriptors).squeeze(-1).masked_fill(~valid.bool(), 0.0)
        return base_logits + residual, residual


def default_producer_model_config() -> MaskBagMILConfig:
    """Match the accepted descriptor/scorer geometry with fresh weights."""

    return MaskBagMILConfig(
        token_dim=128,
        token_layers=3,
        hidden_dim=256,
        metadata_dim=4,
        bag_temperature=0.20,
        context_radius=2,
        minimum_grid_mass=0.25,
    )


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initial_producer_state(
    model_config: MaskBagMILConfig, *, seed: int
) -> dict[str, torch.Tensor]:
    _seed_torch(seed)
    model = RadDinoMaskBagMIL(model_config)
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def deterministic_label_group_balanced_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """Interleave image labels while cycling once across groups before reuse."""

    if batch_size < 2 or not records:
        raise ValueError("T1 balanced batches need records and batch size >= 2")
    rng = np.random.default_rng(seed)
    sequences: dict[int, list[int]] = {}
    for label in (0, 1):
        grouped: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if int(record["label"]) == label:
                grouped.setdefault(str(record["group_id"]), []).append(index)
        if not grouped:
            raise ValueError("T1 producer batches require both image labels")
        keys = sorted(grouped)
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(grouped[key])
        sequence: list[int] = []
        while any(grouped[key] for key in keys):
            for key in keys:
                if grouped[key]:
                    sequence.append(grouped[key].pop())
        sequences[label] = sequence
    positions = {0: 0, 1: 0}
    batches: list[np.ndarray] = []
    while any(positions[label] < len(sequences[label]) for label in (0, 1)):
        remaining = batch_size
        chosen: list[int] = []
        for label, quota in ((0, batch_size // 2), (1, batch_size - batch_size // 2)):
            stop = min(positions[label] + quota, len(sequences[label]))
            chosen.extend(sequences[label][positions[label] : stop])
            remaining -= stop - positions[label]
            positions[label] = stop
        for label in (0, 1):
            if remaining == 0:
                break
            stop = min(positions[label] + remaining, len(sequences[label]))
            chosen.extend(sequences[label][positions[label] : stop])
            remaining -= stop - positions[label]
            positions[label] = stop
        if not chosen:
            raise RuntimeError("T1 balanced batch construction made no progress")
        rng.shuffle(chosen)
        batches.append(np.asarray(chosen, dtype=np.int64))
    flattened = np.concatenate(batches)
    if sorted(flattened.tolist()) != list(range(len(records))):
        raise RuntimeError("T1 balanced batches do not cover each record once")
    return batches


def _padded_batch(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    chosen = [records[int(index)] for index in indices]
    if not chosen:
        raise ValueError("T1 producer batch cannot be empty")
    maximum = max(len(record["candidate_indices"]) for record in chosen)
    descriptor_dim = int(np.asarray(chosen[0]["descriptors"]).shape[1])
    original = np.zeros((len(chosen), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(original)
    valid = np.zeros((len(chosen), maximum), dtype=bool)
    labels = np.zeros(len(chosen), dtype=np.float32)
    counts = np.zeros(len(chosen), dtype=np.float32)
    for row, record in enumerate(chosen):
        count = len(record["candidate_indices"])
        arrays = (
            np.asarray(record["descriptors"], dtype=np.float32),
            np.asarray(record["flipped_descriptors"], dtype=np.float32),
        )
        if count < 1 or arrays[0].shape != (count, descriptor_dim):
            raise ValueError("T1 descriptor/candidate alignment mismatch")
        if arrays[1].shape != arrays[0].shape:
            raise ValueError("T1 original/flip descriptors differ in shape")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("T1 descriptors must be finite")
        original[row, :count] = arrays[0]
        flipped[row, :count] = arrays[1]
        valid[row, :count] = True
        labels[row] = float(record["label"])
        counts[row] = float(count)
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (original, flipped, valid, labels, counts)
    )


def train_count_controlled_producer(
    records: Sequence[Mapping[str, Any]],
    model_config: MaskBagMILConfig,
    training_config: CountControlledSelfPacedConfig,
    *,
    device: torch.device,
    seed_offset: int,
    initial_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[RadDinoMaskBagMIL, list[dict[str, float]]]:
    """Fit one fresh image-label producer with the fixed count penalty."""

    if not records:
        raise ValueError("T1 producer training records cannot be empty")
    derived_seed = int(training_config.seed) + int(seed_offset)
    if initial_state is None:
        initial_state = initial_producer_state(model_config, seed=derived_seed)
    elif device.type != "cuda":
        raise ValueError("T1 precomputed producer state requires a CUDA device")
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(derived_seed)
    else:
        torch.manual_seed(derived_seed)
    model = RadDinoMaskBagMIL(model_config).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.producer_learning_rate,
        weight_decay=training_config.producer_weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.producer_epochs + 1):
        model.train()
        batches = deterministic_label_group_balanced_batches(
            records,
            batch_size=training_config.producer_batch_size,
            seed=derived_seed + epoch,
        )
        sums = {"total": 0.0, "image": 0.0, "consistency": 0.0, "count": 0.0}
        for indices in batches:
            original, flipped, valid, labels, counts = _padded_batch(
                records, indices, device
            )
            original_logits, original_bag = model.score_descriptors(original, valid)
            flipped_logits, flipped_bag = model.score_descriptors(flipped, valid)
            image_loss = 0.5 * (
                image_bag_loss(original_bag, labels)
                + image_bag_loss(flipped_bag, labels)
            )
            consistency = aligned_candidate_consistency_loss(
                original_logits, flipped_logits, valid
            )
            count_loss = 0.5 * (
                count_independence_loss(original_bag, counts)
                + count_independence_loss(flipped_bag, counts)
            )
            total = (
                image_loss
                + training_config.view_consistency_weight * consistency
                + training_config.count_independence_weight * count_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total),
                ("image", image_loss),
                ("consistency", consistency),
                ("count", count_loss),
            ):
                sums[key] += float(value.detach().item())
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / len(batches) for key, value in sums.items()},
            }
        )
    return model, history


def score_count_controlled_producer(
    records: Sequence[Mapping[str, Any]],
    model: RadDinoMaskBagMIL,
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Freeze aligned-view candidate logits and averaged-view bag probability."""

    if batch_size < 1:
        raise ValueError("T1 producer scoring batch size must be positive")
    model.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        original, flipped, valid, _labels, _counts = _padded_batch(
            records, indices, device
        )
        with torch.inference_mode():
            original_logits, _ = model.score_descriptors(original, valid)
            flipped_logits, _ = model.score_descriptors(flipped, valid)
            averaged = 0.5 * (original_logits + flipped_logits)
            bag_logits = smooth_mil_pool(
                averaged, valid, temperature=model.config.bag_temperature
            )
        for row, record_index in enumerate(indices):
            record = records[int(record_index)]
            count = len(record["candidate_indices"])
            original_row = original_logits[row, :count].float().cpu().numpy()
            flipped_row = flipped_logits[row, :count].float().cpu().numpy()
            output.append(
                {
                    "image_id": str(record["image_id"]),
                    "group_id": str(record["group_id"]),
                    "image_label": int(record["label"]),
                    "original_logits": original_row,
                    "flipped_logits": flipped_row,
                    "candidate_count": count,
                    "bag_logit": float(bag_logits[row].item()),
                    "bag_probability": float(torch.sigmoid(bag_logits[row]).item()),
                }
            )
    return output


def fit_count_controlled_oof_fold(
    records: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    *,
    heldout_fold: int,
    model_config: MaskBagMILConfig,
    training_config: CountControlledSelfPacedConfig,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Fit and score one producer fold with complete group exclusion."""

    folds = np.asarray(fold_ids, dtype=np.int32)
    if folds.shape != (len(records),) or heldout_fold not in set(folds.tolist()):
        raise ValueError("T1 fold IDs do not align with records")
    training = [records[int(index)] for index in np.flatnonzero(folds != heldout_fold)]
    heldout = [records[int(index)] for index in np.flatnonzero(folds == heldout_fold)]
    training_groups = sorted({str(record["group_id"]) for record in training})
    heldout_groups = sorted({str(record["group_id"]) for record in heldout})
    if set(training_groups) & set(heldout_groups):
        raise RuntimeError("T1 producer fold trained on a held-out group")
    producer, history = train_count_controlled_producer(
        training,
        model_config,
        training_config,
        device=device,
        seed_offset=1000 + heldout_fold,
        initial_state=initial_state,
    )
    scores = score_count_controlled_producer(
        heldout,
        producer,
        batch_size=training_config.producer_batch_size,
        device=device,
    )
    for row in scores:
        row["heldout_fold"] = int(heldout_fold)
    return {
        "heldout_fold": int(heldout_fold),
        "derived_seed": int(training_config.seed) + 1000 + int(heldout_fold),
        "training_groups": training_groups,
        "heldout_groups": heldout_groups,
        "group_overlap": 0,
        "producer_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in producer.state_dict().items()
        },
        "training_history": history,
        "heldout_scores": scores,
        "validation_segmentation_quality_used": False,
    }


def count_independence_loss(
    bag_logits: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Squared Pearson correlation of bag probability and log candidate count."""

    if bag_logits.ndim != 1 or candidate_counts.shape != bag_logits.shape:
        raise ValueError("T1 bag logits and candidate counts must be aligned vectors")
    if len(bag_logits) < 1 or not torch.isfinite(bag_logits).all():
        raise ValueError("T1 count loss requires finite bag logits")
    if not torch.isfinite(candidate_counts).all() or torch.any(candidate_counts < 1):
        raise ValueError("T1 candidate counts must be finite and positive")
    if len(bag_logits) == 1:
        return bag_logits.sum() * 0.0
    probabilities = torch.sigmoid(bag_logits)
    nuisance = torch.log1p(candidate_counts.to(dtype=probabilities.dtype))
    probabilities = probabilities - probabilities.mean()
    nuisance = nuisance - nuisance.mean()
    covariance = torch.mean(probabilities * nuisance)
    denominator = torch.sqrt(
        torch.mean(probabilities.square()) * torch.mean(nuisance.square()) + epsilon
    )
    return (covariance / denominator).square()


def _average_tie_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("T1 ranks require a finite nontrivial vector")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _absolute_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = _average_tie_ranks(first)
    right = _average_tie_ranks(second)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("T1 Spearman ranks must be nonconstant")
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_array = np.asarray(labels, dtype=np.int32)
    ranks = _average_tie_ranks(scores)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("T1 AUROC requires both image labels")
    positive = label_array == 1
    positive_count = int(positive.sum())
    negative_count = len(label_array) - positive_count
    return float(
        (
            ranks[positive].sum()
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )


def audit_count_controlled_oof_producer(
    records: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    fold_artifacts: Sequence[Mapping[str, Any]],
    config: CountControlledSelfPacedConfig,
) -> dict[str, Any]:
    """Verify exact OOF exclusion/coverage and the producer-only gate."""

    folds = np.asarray(fold_ids, dtype=np.int32)
    if folds.shape != (len(records),):
        raise ValueError("T1 fold IDs do not align with records")
    expected_folds = sorted(np.unique(folds).tolist())
    indexed = {int(item["heldout_fold"]): item for item in fold_artifacts}
    if len(indexed) != len(fold_artifacts) or sorted(indexed) != expected_folds:
        raise ValueError("T1 OOF artifacts must cover each fold once")
    exclusion = audit_crossfit_training_exclusion(
        [str(record["group_id"]) for record in records],
        folds,
        {fold: indexed[fold]["training_groups"] for fold in expected_folds},
    )
    predictions = [
        row
        for fold in expected_folds
        for row in indexed[fold]["heldout_scores"]
    ]
    by_id = {str(row["image_id"]): row for row in predictions}
    expected_ids = {str(record["image_id"]) for record in records}
    if len(predictions) != len(records) or set(by_id) != expected_ids:
        raise RuntimeError("T1 OOF scores do not cover every image exactly once")
    ordered: list[dict[str, Any]] = []
    agreements: list[bool] = []
    for record, fold in zip(records, folds, strict=True):
        row = by_id[str(record["image_id"])]
        original = np.asarray(row["original_logits"], dtype=np.float64)
        flipped = np.asarray(row["flipped_logits"], dtype=np.float64)
        count = len(record["candidate_indices"])
        if (
            str(row["group_id"]) != str(record["group_id"])
            or int(row["image_label"]) != int(record["label"])
            or int(row["heldout_fold"]) != int(fold)
            or int(row["candidate_count"]) != count
            or original.shape != (count,)
            or flipped.shape != original.shape
            or not np.isfinite(original).all()
            or not np.isfinite(flipped).all()
        ):
            raise RuntimeError("T1 OOF score identity/content mismatch")
        probability = float(row["bag_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("T1 OOF bag probability is invalid")
        agreements.append(int(np.argmax(original)) == int(np.argmax(flipped)))
        ordered.append(dict(row))
    count_spearman = _absolute_spearman(
        [int(row["candidate_count"]) for row in ordered],
        [float(row["bag_probability"]) for row in ordered],
    )
    auroc = _binary_auroc(
        [int(row["image_label"]) for row in ordered],
        [float(row["bag_probability"]) for row in ordered],
    )
    agreement = float(np.mean(agreements))
    checks = {
        "count_spearman": count_spearman <= config.maximum_count_spearman,
        "image_auroc": auroc >= config.minimum_oof_auroc,
        "view_agreement": agreement >= config.minimum_view_agreement,
        "group_exclusion": exclusion.get("group_overlap") == 0,
    }
    return {
        "complete": True,
        "records": len(ordered),
        "folds": len(expected_folds),
        "group_overlap": int(exclusion.get("group_overlap", -1)),
        "absolute_candidate_count_probability_spearman": count_spearman,
        "image_auroc": auroc,
        "original_flip_top1_agreement": agreement,
        "checks": checks,
        "producer_gate_pass": all(checks.values()),
        "ordered_scores": ordered,
    }


def _conservative_margin(original: np.ndarray, flipped: np.ndarray) -> tuple[int, float]:
    if original.ndim != 1 or original.shape != flipped.shape or len(original) < 1:
        raise ValueError("T1 view logits must be aligned nonempty vectors")
    top = int(np.argmax(original))
    if top != int(np.argmax(flipped)):
        raise ValueError("T1 positive target requires original/flip top-1 agreement")
    if len(original) == 1:
        return top, float(np.finfo(np.float32).max)
    other = np.arange(len(original)) != top
    margin = min(
        float(original[top] - np.max(original[other])),
        float(flipped[top] - np.max(flipped[other])),
    )
    return top, margin


def build_self_paced_targets(
    records: Sequence[Mapping[str, Any]],
    producer_audit: Mapping[str, Any],
    config: CountControlledSelfPacedConfig,
) -> dict[str, Any]:
    """Create nested positive stages and hierarchically weighted negatives."""

    if producer_audit.get("producer_gate_pass") is not True:
        raise RuntimeError("T1 consumer remains locked until the producer gate passes")
    scores = producer_audit.get("ordered_scores")
    if not isinstance(scores, list):
        raise ValueError("T1 producer audit is missing ordered OOF scores")
    by_id = {str(row["image_id"]): row for row in scores}
    if len(by_id) != len(scores):
        raise ValueError("T1 producer scores contain duplicate image IDs")
    eligible: list[dict[str, Any]] = []
    negative_targets: list[dict[str, Any]] = []
    negative_records = [record for record in records if int(record["label"]) == 0]
    negative_image_weight = 1.0 / max(1, len(negative_records))
    for record in records:
        image_id = str(record["image_id"])
        if image_id not in by_id:
            raise ValueError("T1 producer scores do not cover every record")
        count = len(record["candidate_indices"])
        if int(record["label"]) == 0:
            families = [str(value) for value in record["family_ids"]]
            if len(families) != count:
                raise ValueError("T1 family IDs do not align with candidates")
            family_members: dict[str, list[int]] = {}
            for candidate, family in enumerate(families):
                family_members.setdefault(family, []).append(candidate)
            family_weight = negative_image_weight / len(family_members)
            for family in sorted(family_members):
                members = family_members[family]
                candidate_weight = family_weight / len(members)
                negative_targets.extend(
                    {
                        "image_id": image_id,
                        "candidate_index": candidate,
                        "target": 0,
                        "weight": candidate_weight,
                        "family_id": family,
                        "producer_fold": int(by_id[image_id]["heldout_fold"]),
                    }
                    for candidate in members
                )
            continue
        row = by_id[image_id]
        original = np.asarray(row["original_logits"], dtype=np.float64)
        flipped = np.asarray(row["flipped_logits"], dtype=np.float64)
        if original.shape != (count,) or flipped.shape != original.shape:
            raise ValueError("T1 positive score/candidate alignment mismatch")
        if int(np.argmax(original)) != int(np.argmax(flipped)):
            continue
        candidate, margin = _conservative_margin(original, flipped)
        if margin > 0.0:
            eligible.append(
                {
                    "image_id": image_id,
                    "candidate_index": candidate,
                    "margin": margin,
                    "producer_fold": int(row["heldout_fold"]),
                }
            )
    eligible.sort(key=lambda row: (-float(row["margin"]), str(row["image_id"])))
    stages: list[dict[str, Any]] = []
    for stage, fraction in enumerate(config.pace_fractions, start=1):
        selected_count = min(len(eligible), int(math.ceil(fraction * len(eligible))))
        selected = [dict(row) for row in eligible[:selected_count]]
        weight = 1.0 / max(1, selected_count)
        for row in selected:
            row.update({"target": 1, "weight": weight})
        stages.append(
            {
                "stage": stage,
                "fraction": fraction,
                "positive_targets": selected,
            }
        )
    return {
        "eligible_positive_bags": len(eligible),
        "negative_targets": negative_targets,
        "stages": stages,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


__all__ = [
    "CountControlledResidual",
    "CountControlledSelfPacedConfig",
    "audit_count_controlled_oof_producer",
    "build_self_paced_targets",
    "count_independence_loss",
    "default_producer_model_config",
    "deterministic_label_group_balanced_batches",
    "fit_count_controlled_oof_fold",
    "initial_producer_state",
    "score_count_controlled_producer",
    "train_count_controlled_producer",
]
