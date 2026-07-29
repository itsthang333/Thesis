from __future__ import annotations

"""Group-excluded OOF orchestration for the R1 normal-prototype selector."""

from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.mask_bag_crossfit import audit_crossfit_training_exclusion
from models.mask_bag_normal_residual_training import (
    NormalResidualTrainingConfig,
    attach_normal_prototype_features,
    fit_normal_prototype_bank,
    score_normal_residual_records,
    train_normal_residual_adapter,
)
from models.mask_bag_residual_objective import ResidualObjectiveConfig


def _binary_bce_from_logit(logit: float, label: int) -> float:
    value = float(logit)
    target = float(label)
    return float(max(value, 0.0) - value * target + np.log1p(np.exp(-abs(value))))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("Spearman inputs must be aligned nontrivial vectors")
    ranks_first = _average_ranks(first.astype(np.float64))
    ranks_second = _average_ranks(second.astype(np.float64))
    if np.ptp(ranks_first) == 0.0 or np.ptp(ranks_second) == 0.0:
        return 0.0
    return float(np.corrcoef(ranks_first, ranks_second)[0, 1])


def fit_normal_oof_fold(
    records: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    *,
    heldout_fold: int,
    prototype_count: int,
    frozen_base_scorer: nn.Module,
    descriptor_dim: int,
    objective_config: ResidualObjectiveConfig,
    training_config: NormalResidualTrainingConfig,
    device: torch.device,
    initial_adapter_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Fit one fold with complete held-out-group exclusion."""

    folds = np.asarray(fold_ids, dtype=np.int32)
    if folds.shape != (len(records),) or heldout_fold not in set(folds.tolist()):
        raise ValueError("fold IDs do not align with records")
    heldout_indices = np.flatnonzero(folds == heldout_fold)
    training_indices = np.flatnonzero(folds != heldout_fold)
    training = [records[int(index)] for index in training_indices]
    heldout = [records[int(index)] for index in heldout_indices]
    training_groups = sorted({str(record["group_id"]) for record in training})
    heldout_groups = sorted({str(record["group_id"]) for record in heldout})
    overlap = set(training_groups) & set(heldout_groups)
    if overlap:
        raise RuntimeError("OOF fold training contains a held-out group")

    derived_seed = (
        int(training_config.seed)
        + 1000 * int(prototype_count)
        + int(heldout_fold)
    )
    prototypes, prototype_audit = fit_normal_prototype_bank(
        training,
        prototype_count=prototype_count,
        seed=derived_seed,
    )
    enriched_training = attach_normal_prototype_features(
        training,
        prototypes,
        temperature=training_config.prototype_temperature,
    )
    enriched_heldout = attach_normal_prototype_features(
        heldout,
        prototypes,
        temperature=training_config.prototype_temperature,
    )
    fold_training_config = replace(training_config, seed=derived_seed)
    adapter, history = train_normal_residual_adapter(
        enriched_training,
        frozen_base_scorer,
        descriptor_dim=descriptor_dim,
        objective_config=objective_config,
        training_config=fold_training_config,
        device=device,
        initial_adapter_state=initial_adapter_state,
    )
    scored = score_normal_residual_records(
        enriched_heldout,
        frozen_base_scorer,
        adapter,
        bag_temperature=objective_config.bag_temperature,
        batch_size=training_config.batch_size,
        device=device,
    )
    heldout_by_id = {str(record["image_id"]): record for record in heldout}
    predictions: list[dict[str, Any]] = []
    losses: list[float] = []
    for prediction in scored:
        image_id = str(prediction["image_id"])
        record = heldout_by_id[image_id]
        loss = _binary_bce_from_logit(
            float(prediction["bag_logit"]), int(record["label"])
        )
        losses.append(loss)
        predictions.append(
            {
                "image_id": image_id,
                "group_id": str(record["group_id"]),
                "image_label": int(record["label"]),
                "heldout_fold": int(heldout_fold),
                "candidate_count": int(prediction["candidate_count"]),
                "bag_logit": float(prediction["bag_logit"]),
                "bag_probability": float(prediction["bag_probability"]),
                "image_bce": loss,
            }
        )
    state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    return {
        "prototype_count": int(prototype_count),
        "heldout_fold": int(heldout_fold),
        "derived_seed": derived_seed,
        "training_groups": training_groups,
        "heldout_groups": heldout_groups,
        "group_overlap": 0,
        "prototype_bank": prototypes,
        "prototype_audit": prototype_audit,
        "adapter_state_dict": state,
        "adapter_training_config": asdict(fold_training_config),
        "objective_config": asdict(objective_config),
        "training_history": history,
        "heldout_predictions": predictions,
        "heldout_mean_image_bce": float(np.mean(losses)),
        "validation_segmentation_quality_used": False,
    }


def assemble_normal_oof_candidate(
    records: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    fold_artifacts: Sequence[Mapping[str, Any]],
    *,
    prototype_count: int,
) -> dict[str, Any]:
    """Audit and aggregate one K across all group-excluded folds."""

    folds = np.asarray(fold_ids, dtype=np.int32)
    expected_folds = sorted(np.unique(folds).tolist())
    indexed = {
        int(artifact["heldout_fold"]): artifact for artifact in fold_artifacts
    }
    if len(indexed) != len(fold_artifacts) or sorted(indexed) != expected_folds:
        raise ValueError("OOF artifacts do not cover each fold exactly once")
    if any(int(artifact["prototype_count"]) != prototype_count for artifact in indexed.values()):
        raise ValueError("OOF artifacts mix prototype counts")
    training_groups_by_fold = {
        fold: indexed[fold]["training_groups"] for fold in expected_folds
    }
    exclusion = audit_crossfit_training_exclusion(
        [str(record["group_id"]) for record in records],
        folds,
        training_groups_by_fold,
    )
    predictions = [
        prediction
        for fold in expected_folds
        for prediction in indexed[fold]["heldout_predictions"]
    ]
    by_id = {str(row["image_id"]): row for row in predictions}
    expected_ids = {str(record["image_id"]) for record in records}
    if len(predictions) != len(by_id) or set(by_id) != expected_ids:
        raise RuntimeError("OOF predictions do not cover every training image once")
    ordered = [by_id[str(record["image_id"])] for record in records]
    for record, prediction, fold in zip(records, ordered, folds, strict=True):
        if (
            prediction["group_id"] != str(record["group_id"])
            or int(prediction["image_label"]) != int(record["label"])
            or int(prediction["heldout_fold"]) != int(fold)
        ):
            raise RuntimeError("OOF prediction identity/fold mismatch")
    count_probability_spearman = _spearman(
        np.asarray([row["candidate_count"] for row in ordered], dtype=np.float64),
        np.asarray([row["bag_probability"] for row in ordered], dtype=np.float64),
    )
    return {
        "prototype_count": int(prototype_count),
        "fold_image_bce": [
            float(indexed[fold]["heldout_mean_image_bce"])
            for fold in expected_folds
        ],
        "mean_oof_image_bce": float(
            np.mean([row["image_bce"] for row in ordered])
        ),
        "count_probability_spearman": count_probability_spearman,
        "crossfit_exclusion": exclusion,
        "oof_predictions": ordered,
        "validation_segmentation_quality_used": False,
    }


__all__ = [
    "assemble_normal_oof_candidate",
    "fit_normal_oof_fold",
]
