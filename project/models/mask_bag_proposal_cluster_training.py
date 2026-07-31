from __future__ import annotations

"""Cross-fitted, image-label-only training primitives for the S4 selector.

The OOF teacher supplies proposal-cluster seeds.  The accepted Geometry-v3
scorer is only a frozen student baseline and is never used as an OOF teacher.
This module is dataset-agnostic and accepts no segmentation target or subgroup.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_crossfit import audit_crossfit_training_exclusion
from models.mask_bag_proposal_clusters import (
    build_teacher_proposal_clusters,
    continuation_temperature,
    proposal_cluster_smooth_pool,
)
from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    self_guided_instance_loss,
)


@dataclass(frozen=True)
class ProposalClusterTrainingConfig:
    fold_count: int = 5
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    teacher_instance_loss_weight: float = 0.25
    consistency_weight: float = 0.10
    instance_warmup_epochs: int = 2
    maximum_clusters: int = 4
    minimum_iou: float = 0.50
    minimum_containment: float = 0.75
    start_temperature: float = 1.0
    end_temperature: float = 0.20
    residual_hidden_dim: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.fold_count < 2 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("S4 fold/epoch/batch controls are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("S4 optimizer controls are invalid")
        if self.teacher_instance_loss_weight < 0 or self.consistency_weight < 0:
            raise ValueError("S4 loss weights must be nonnegative")
        if not 0 <= self.instance_warmup_epochs < self.epochs:
            raise ValueError("S4 instance warm-up must precede the final epoch")
        if self.maximum_clusters < 1 or self.residual_hidden_dim < 2:
            raise ValueError("S4 cluster/model size is invalid")
        if not 0.0 <= self.minimum_iou <= 1.0:
            raise ValueError("S4 IoU threshold is invalid")
        if not 0.0 <= self.minimum_containment <= 1.0:
            raise ValueError("S4 containment threshold is invalid")
        if self.start_temperature < self.end_temperature or self.end_temperature <= 0:
            raise ValueError("S4 continuation temperatures are invalid")


class ProposalClusterResidual(nn.Module):
    """Zero-initialized residual applied only to OOF-cluster members."""

    def __init__(self, descriptor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if descriptor_dim < 1 or hidden_dim < 2:
            raise ValueError("S4 residual dimensions are invalid")
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
        cluster_members: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or base_logits.shape != descriptors.shape[:2]:
            raise ValueError("S4 descriptors/base logits must align")
        if cluster_members.shape != base_logits.shape:
            raise ValueError("S4 cluster-member mask must align with logits")
        members = cluster_members.bool()
        residual = self.network(descriptors).squeeze(-1)
        residual = residual.masked_fill(~members, 0.0)
        return base_logits + residual, residual


def default_teacher_model_config() -> MaskBagMILConfig:
    """Use the accepted descriptor/scorer geometry without its fitted weights."""

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


def _padded_batch(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    device: torch.device,
    *,
    require_clusters: bool,
) -> tuple[torch.Tensor, ...]:
    chosen = [records[int(index)] for index in indices]
    if not chosen:
        raise ValueError("S4 batch cannot be empty")
    maximum = max(len(record["candidate_indices"]) for record in chosen)
    descriptor_dim = int(np.asarray(chosen[0]["descriptors"]).shape[1])
    original = np.zeros((len(chosen), maximum, descriptor_dim), dtype=np.float32)
    flipped = np.zeros_like(original)
    valid = np.zeros((len(chosen), maximum), dtype=bool)
    labels = np.zeros(len(chosen), dtype=np.float32)
    cluster_count = max(
        (np.asarray(record.get("clusters", [])).shape[0] for record in chosen),
        default=0,
    )
    clusters = np.zeros((len(chosen), cluster_count, maximum), dtype=bool)
    cluster_valid = np.zeros((len(chosen), cluster_count), dtype=bool)
    for row, record in enumerate(chosen):
        count = len(record["candidate_indices"])
        arrays = (
            np.asarray(record["descriptors"], dtype=np.float32),
            np.asarray(record["flipped_descriptors"], dtype=np.float32),
        )
        if count < 1 or arrays[0].shape != (count, descriptor_dim):
            raise ValueError("S4 descriptor/candidate alignment mismatch")
        if arrays[1].shape != arrays[0].shape:
            raise ValueError("S4 original/flip descriptors differ in shape")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("S4 descriptors must be finite")
        original[row, :count] = arrays[0]
        flipped[row, :count] = arrays[1]
        valid[row, :count] = True
        labels[row] = float(record["label"])
        if require_clusters:
            membership = np.asarray(record["clusters"], dtype=bool)
            membership_valid = np.asarray(record["cluster_valid"], dtype=bool)
            if membership.ndim != 2 or membership.shape[1] != count:
                raise ValueError("S4 stored cluster membership is invalid")
            if membership_valid.shape != (membership.shape[0],):
                raise ValueError("S4 stored cluster validity is invalid")
            clusters[row, : len(membership), :count] = membership
            cluster_valid[row, : len(membership)] = membership_valid
    tensors = tuple(
        torch.from_numpy(value).to(device)
        for value in (original, flipped, valid, labels)
    )
    if not require_clusters:
        return tensors
    return tensors + (
        torch.from_numpy(clusters).to(device),
        torch.from_numpy(cluster_valid).to(device),
    )


def initial_teacher_state(
    model_config: MaskBagMILConfig, *, seed: int
) -> dict[str, torch.Tensor]:
    _seed_torch(seed)
    model = RadDinoMaskBagMIL(model_config)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_proposal_teacher(
    records: Sequence[Mapping[str, Any]],
    model_config: MaskBagMILConfig,
    training_config: ProposalClusterTrainingConfig,
    *,
    device: torch.device,
    seed_offset: int,
) -> tuple[RadDinoMaskBagMIL, list[dict[str, float]]]:
    """Fit one image-label-only teacher with fixed final-epoch selection."""

    if not records:
        raise ValueError("S4 teacher training records cannot be empty")
    derived_seed = int(training_config.seed) + int(seed_offset)
    _seed_torch(derived_seed)
    model = RadDinoMaskBagMIL(model_config).to(device)
    model.load_state_dict(initial_teacher_state(model_config, seed=derived_seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = np.random.default_rng(derived_seed + epoch).permutation(len(records))
        sums = {"total": 0.0, "image": 0.0, "instance": 0.0, "consistency": 0.0}
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            original, flipped, valid, labels = _padded_batch(
                records, indices, device, require_clusters=False
            )
            original_logits, original_bag = model.score_descriptors(original, valid)
            flipped_logits, flipped_bag = model.score_descriptors(flipped, valid)
            image_loss = 0.5 * (
                image_bag_loss(original_bag, labels)
                + image_bag_loss(flipped_bag, labels)
            )
            if epoch > training_config.instance_warmup_epochs:
                instance_loss = 0.5 * (
                    self_guided_instance_loss(original_logits, valid, labels)
                    + self_guided_instance_loss(flipped_logits, valid, labels)
                )
            else:
                instance_loss = original_logits.sum() * 0.0
            consistency = aligned_candidate_consistency_loss(
                original_logits, flipped_logits, valid
            )
            total = (
                image_loss
                + training_config.teacher_instance_loss_weight * instance_loss
                + training_config.consistency_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total),
                ("image", image_loss),
                ("instance", instance_loss),
                ("consistency", consistency),
            ):
                sums[key] += float(value.detach().item())
            batches += 1
        history.append(
            {"epoch": float(epoch), **{key: value / batches for key, value in sums.items()}}
        )
    return model, history


def score_proposal_teacher(
    records: Sequence[Mapping[str, Any]],
    model: RadDinoMaskBagMIL,
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score both aligned views; conservative seed evidence is their minimum."""

    if batch_size < 1:
        raise ValueError("S4 teacher scoring batch size must be positive")
    model.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        original, flipped, valid, _labels = _padded_batch(
            records, indices, device, require_clusters=False
        )
        with torch.inference_mode():
            original_logits, _ = model.score_descriptors(original, valid)
            flipped_logits, _ = model.score_descriptors(flipped, valid)
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
                    "conservative_seed_logits": np.minimum(original_row, flipped_row),
                    "selected_view_agreement": bool(
                        int(np.argmax(original_row)) == int(np.argmax(flipped_row))
                    ),
                }
            )
    return output


def attach_teacher_clusters(
    records: Sequence[Mapping[str, Any]],
    teacher_scores: Sequence[Mapping[str, Any]],
    config: ProposalClusterTrainingConfig,
) -> list[dict[str, Any]]:
    """Freeze disjoint S4 clusters from conservative OOF teacher evidence."""

    by_id = {str(row["image_id"]): row for row in teacher_scores}
    if len(by_id) != len(teacher_scores):
        raise ValueError("S4 teacher scores contain duplicate image IDs")
    enriched: list[dict[str, Any]] = []
    for record in records:
        image_id = str(record["image_id"])
        if image_id not in by_id:
            raise ValueError("S4 teacher scores do not cover every record")
        score = by_id[image_id]
        count = len(record["candidate_indices"])
        seed_logits = np.asarray(score["conservative_seed_logits"], dtype=np.float32)
        iou = np.asarray(record["pairwise_iou"], dtype=np.float32)
        containment = np.asarray(record["pairwise_containment"], dtype=np.float32)
        if seed_logits.shape != (count,) or iou.shape != (count, count):
            raise ValueError("S4 teacher/IoU candidate alignment mismatch")
        if containment.shape != iou.shape:
            raise ValueError("S4 containment candidate alignment mismatch")
        adjacency = (iou >= config.minimum_iou) | (
            containment >= config.minimum_containment
        )
        adjacency = adjacency | adjacency.T
        np.fill_diagonal(adjacency, True)
        clusters, cluster_valid, seeds = build_teacher_proposal_clusters(
            torch.from_numpy(seed_logits[None]),
            torch.ones((1, count), dtype=torch.bool),
            torch.from_numpy(adjacency.astype(np.float32)[None]),
            maximum_clusters=config.maximum_clusters,
            minimum_overlap=1.0,
        )
        item = dict(record)
        item["clusters"] = clusters[0].cpu().numpy()
        item["cluster_valid"] = cluster_valid[0].cpu().numpy()
        item["seed_indices"] = seeds[0].cpu().numpy()
        item["teacher_original_logits"] = np.asarray(
            score["original_logits"], dtype=np.float32
        )
        item["teacher_flipped_logits"] = np.asarray(
            score["flipped_logits"], dtype=np.float32
        )
        item["teacher_conservative_seed_logits"] = seed_logits
        item["teacher_selected_view_agreement"] = bool(
            score["selected_view_agreement"]
        )
        enriched.append(item)
    return enriched


def audit_oof_teacher_coverage(
    records: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    fold_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove each train record was scored once by a group-excluded teacher."""

    folds = np.asarray(fold_ids, dtype=np.int32)
    expected_folds = sorted(np.unique(folds).tolist())
    indexed = {int(item["heldout_fold"]): item for item in fold_artifacts}
    if len(indexed) != len(fold_artifacts) or sorted(indexed) != expected_folds:
        raise ValueError("S4 OOF artifacts do not cover each fold exactly once")
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
    if len(by_id) != len(records) or set(by_id) != {
        str(record["image_id"]) for record in records
    }:
        raise RuntimeError("S4 OOF scores do not cover each train image once")
    for record, fold in zip(records, folds, strict=True):
        row = by_id[str(record["image_id"])]
        if (
            str(row["group_id"]) != str(record["group_id"])
            or int(row["heldout_fold"]) != int(fold)
        ):
            raise RuntimeError("S4 OOF score identity/fold mismatch")
    return {
        "complete": True,
        "records": len(records),
        "folds": len(expected_folds),
        "group_overlap": 0,
        "crossfit_exclusion": exclusion,
        "ordered_scores": [by_id[str(record["image_id"])] for record in records],
    }


def train_cluster_residual(
    records: Sequence[Mapping[str, Any]],
    frozen_base_scorer: nn.Module,
    model_config: MaskBagMILConfig,
    training_config: ProposalClusterTrainingConfig,
    *,
    device: torch.device,
) -> tuple[ProposalClusterResidual, list[dict[str, float]]]:
    """Fit only cluster-member residuals with normalized continuation bags."""

    if not records:
        raise ValueError("S4 residual training records cannot be empty")
    frozen_base_scorer.requires_grad_(False).eval()
    _seed_torch(training_config.seed + 9000)
    residual = ProposalClusterResidual(
        model_config.descriptor_dim, training_config.residual_hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        residual.train()
        temperature = continuation_temperature(
            epoch,
            training_config.epochs,
            start_temperature=training_config.start_temperature,
            end_temperature=training_config.end_temperature,
        )
        order = np.random.default_rng(training_config.seed + 9000 + epoch).permutation(
            len(records)
        )
        sums = {"total": 0.0, "image": 0.0, "consistency": 0.0}
        batches = 0
        for start in range(0, len(order), training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            original, flipped, valid, labels, clusters, cluster_valid = _padded_batch(
                records, indices, device, require_clusters=True
            )
            members = clusters.any(dim=1) & valid
            with torch.inference_mode():
                original_base, _ = frozen_base_scorer.score_descriptors(original, valid)
                flipped_base, _ = frozen_base_scorer.score_descriptors(flipped, valid)
            original_logits, _ = residual(original, original_base, members)
            flipped_logits, _ = residual(flipped, flipped_base, members)
            _original_cluster, original_bag = proposal_cluster_smooth_pool(
                original_logits,
                clusters,
                cluster_valid,
                within_temperature=temperature,
                between_temperature=temperature,
            )
            _flipped_cluster, flipped_bag = proposal_cluster_smooth_pool(
                flipped_logits,
                clusters,
                cluster_valid,
                within_temperature=temperature,
                between_temperature=temperature,
            )
            image_loss = 0.5 * (
                image_bag_loss(original_bag, labels)
                + image_bag_loss(flipped_bag, labels)
            )
            consistency = aligned_candidate_consistency_loss(
                original_logits, flipped_logits, members
            )
            total = image_loss + training_config.consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total),
                ("image", image_loss),
                ("consistency", consistency),
            ):
                sums[key] += float(value.detach().item())
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                "temperature": float(temperature),
                **{key: value / batches for key, value in sums.items()},
            }
        )
    if any(parameter.requires_grad for parameter in frozen_base_scorer.parameters()):
        raise RuntimeError("S4 frozen base scorer became trainable")
    return residual, history


__all__ = [
    "ProposalClusterResidual",
    "ProposalClusterTrainingConfig",
    "attach_teacher_clusters",
    "audit_oof_teacher_coverage",
    "default_teacher_model_config",
    "initial_teacher_state",
    "score_proposal_teacher",
    "train_cluster_residual",
    "train_proposal_teacher",
]
