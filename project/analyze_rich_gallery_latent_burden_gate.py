from __future__ import annotations

"""Retrospective, group-held-out feasibility test for a latent burden gate.

The rich-gallery baseline has three useful fixed extent experts, but true lesion
size is unavailable to an image-label-only system.  This diagnostic asks a
narrower question: do already-frozen, annotation-free bag statistics contain
enough scale information to route those experts?

Spatial GT is used only after the candidate table has been frozen, to fit and
evaluate an out-of-fold *retrospective upper bound*.  Consequently, none of the
resulting gates is deployable and none may be evaluated on test.  Group IDs are
kept wholly within a fold to avoid paired-view leakage.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from models.rich_gallery_g2_objective import (
    average_percentile_rank,
    rank_fusion_scores,
    stable_select,
)


GROUPS = ("small", "medium", "large")
GROUP_TO_INDEX = {group: index for index, group in enumerate(GROUPS)}
EXPERT_BETAS = np.asarray((-1.0, 0.0, 0.5), dtype=np.float64)
SOURCES = ("classifier448", "layercam320", "external_saliency")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
RIDGE_LAMBDA = 10.0
FEATURE_LIMIT = 8
UTILITY_TEMPERATURE = 0.05
ABSTENTION_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
FOLD_COUNT = 5
SEED = 20260802

CAUSAL_COLUMNS = (
    "matched_logit_delta",
    "random_logit_delta",
    "matched_class_contrast_delta",
    "random_class_contrast_delta",
    "pool0_relative_l2_contrast",
    "pool0_delta_energy_inside",
    "transition1_relative_l2_contrast",
    "transition1_delta_energy_inside",
    "transition2_relative_l2_contrast",
    "transition2_delta_energy_inside",
    "transition3_relative_l2_contrast",
    "transition3_delta_energy_inside",
    "norm5_relative_l2_contrast",
    "norm5_delta_energy_inside",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--expected-per-candidate-sha256", required=True)
    parser.add_argument("--cross-view-score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()[:16], 16)


def add_stats(target: dict[str, float], prefix: str, values: Iterable[float]) -> None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        for suffix in ("mean", "std", "min", "max", *(f"q{int(q * 100):02d}" for q in QUANTILES)):
            target[f"{prefix}_{suffix}"] = 0.0
        return
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values for {prefix}")
    target[f"{prefix}_mean"] = float(array.mean())
    target[f"{prefix}_std"] = float(array.std())
    target[f"{prefix}_min"] = float(array.min())
    target[f"{prefix}_max"] = float(array.max())
    for q in QUANTILES:
        target[f"{prefix}_q{int(q * 100):02d}"] = float(np.quantile(array, q))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(values.max())
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / float(exp.sum())


def normalized_entropy(weights: np.ndarray) -> float:
    if weights.size <= 1:
        return 0.0
    entropy = -float(np.sum(weights * np.log(np.clip(weights, 1.0e-12, None))))
    return entropy / float(np.log(weights.size))


def build_group_folds(
    group_ids: list[str], targets: np.ndarray, fold_count: int = FOLD_COUNT
) -> np.ndarray:
    """Greedily balance class counts while keeping each group in one fold."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)
    vectors = {
        group_id: np.bincount(targets[indices], minlength=len(GROUPS)).astype(np.float64)
        for group_id, indices in grouped.items()
    }
    total = np.bincount(targets, minlength=len(GROUPS)).astype(np.float64)
    target_per_fold = total / fold_count
    fold_counts = np.zeros((fold_count, len(GROUPS)), dtype=np.float64)
    fold_sizes = np.zeros(fold_count, dtype=np.float64)
    assignment: dict[str, int] = {}

    ordered_groups = sorted(
        grouped,
        key=lambda group_id: (
            -float(np.max(vectors[group_id] / np.maximum(total, 1.0))),
            -len(grouped[group_id]),
            stable_hash(group_id),
        ),
    )
    target_size = len(group_ids) / fold_count
    for group_id in ordered_groups:
        vector = vectors[group_id]
        scores: list[tuple[float, float, int]] = []
        for candidate_fold in range(fold_count):
            proposed_counts = fold_counts.copy()
            proposed_sizes = fold_sizes.copy()
            proposed_counts[candidate_fold] += vector
            proposed_sizes[candidate_fold] += len(grouped[group_id])
            class_cost = float(
                np.sum(
                    ((proposed_counts - target_per_fold[None, :])
                    / np.maximum(target_per_fold[None, :], 1.0))
                    ** 2
                )
            )
            size_cost = float(
                np.sum(((proposed_sizes - target_size) / max(target_size, 1.0)) ** 2)
            )
            scores.append((class_cost + 0.05 * size_cost, fold_sizes[candidate_fold], candidate_fold))
        fold = min(scores)[2]
        assignment[group_id] = fold
        fold_counts[fold] += vector
        fold_sizes[fold] += len(grouped[group_id])

    folds = np.asarray([assignment[group_id] for group_id in group_ids], dtype=np.int64)
    for fold in range(fold_count):
        held_out_groups = {group_ids[index] for index in np.flatnonzero(folds == fold)}
        train_groups = {group_ids[index] for index in np.flatnonzero(folds != fold)}
        if held_out_groups & train_groups:
            raise AssertionError("group leakage across folds")
        held_out_targets = set(targets[folds == fold].tolist())
        if held_out_targets != set(range(len(GROUPS))):
            raise ValueError(f"fold {fold} is missing classes: {held_out_targets}")
    return folds


def select_features(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    limit: int | None,
) -> np.ndarray:
    if limit is None or train_x.shape[1] <= limit:
        return np.arange(train_x.shape[1], dtype=np.int64)
    if train_targets.ndim == 1:
        train_targets = train_targets[:, None]
    relevance = np.zeros(train_x.shape[1], dtype=np.float64)
    for feature_index in range(train_x.shape[1]):
        feature = train_x[:, feature_index]
        if float(feature.std()) < 1.0e-12:
            continue
        for target_index in range(train_targets.shape[1]):
            target = train_targets[:, target_index]
            if float(target.std()) < 1.0e-12:
                continue
            correlation = float(np.corrcoef(feature, target)[0, 1])
            if np.isfinite(correlation):
                relevance[feature_index] = max(relevance[feature_index], abs(correlation))
    order = np.argsort(-relevance, kind="stable")
    return order[:limit]


def fit_predict_ridge(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    validation_x: np.ndarray,
    sample_weights: np.ndarray,
    feature_limit: int | None,
) -> np.ndarray:
    selected_features = select_features(train_x, train_targets, feature_limit)
    train_x = train_x[:, selected_features]
    validation_x = validation_x[:, selected_features]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    train_z = (train_x - mean) / scale
    validation_z = (validation_x - mean) / scale
    train_z = np.column_stack([np.ones(train_z.shape[0]), train_z])
    validation_z = np.column_stack([np.ones(validation_z.shape[0]), validation_z])

    weighted_x = train_z * sample_weights[:, None]
    penalty = np.eye(train_z.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_z.T @ weighted_x + penalty,
        weighted_x.T @ train_targets,
    )
    return validation_z @ coefficients


def fit_predict_balanced_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    feature_limit: int | None,
) -> np.ndarray:
    class_counts = np.bincount(train_y, minlength=len(GROUPS)).astype(np.float64)
    sample_weights = train_y.size / (len(GROUPS) * class_counts[train_y])
    target = np.eye(len(GROUPS), dtype=np.float64)[train_y]
    logits = fit_predict_ridge(
        train_x,
        target,
        validation_x,
        sample_weights,
        feature_limit,
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(logits, -60.0, 60.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def fit_predict_utility_ridge(
    train_x: np.ndarray,
    train_utilities: np.ndarray,
    validation_x: np.ndarray,
    feature_limit: int | None,
) -> np.ndarray:
    return fit_predict_ridge(
        train_x,
        train_utilities,
        validation_x,
        np.ones(train_x.shape[0], dtype=np.float64),
        feature_limit,
    )


def utility_to_probabilities(utilities: np.ndarray) -> np.ndarray:
    logits = utilities / UTILITY_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(logits, -60.0, 60.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def metrics(values: np.ndarray, true_groups: list[str]) -> dict[str, float]:
    return {
        group: float(
            np.mean(
                [
                    value
                    for value, true_group in zip(values.tolist(), true_groups, strict=True)
                    if group == "overall" or true_group == group
                ]
            )
        )
        for group in ("overall", *GROUPS)
    }


def paired_group_bootstrap(
    values: np.ndarray,
    baseline: np.ndarray,
    group_ids: list[str],
    draws: int = 10000,
) -> dict[str, float]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped_indices[group_id].append(index)
    groups = sorted(grouped_indices)
    rng = np.random.default_rng(SEED)
    deltas = np.zeros(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = [index for group_index in sampled for index in grouped_indices[groups[group_index]]]
        deltas[draw] = float(np.mean(values[indices] - baseline[indices]))
    return {
        "draws": draws,
        "mean_delta": float(np.mean(values - baseline)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
    }


def main() -> None:
    args = parse_args()
    actual_hash = sha256_file(args.per_candidate)
    if actual_hash != args.expected_per_candidate_sha256:
        raise ValueError(f"per-candidate hash mismatch: {actual_hash}")

    bags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.per_candidate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(CAUSAL_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing causal columns: {sorted(missing)}")
        for raw in reader:
            row = {
                "image_id": str(raw["image_id"]),
                "group_id": str(raw["group_id"]),
                "size_group": str(raw["size_group"]),
                "candidate_local_index": int(raw["candidate_local_index"]),
                "candidate_index": int(raw["candidate_index"]),
                "source": str(raw["source"]),
                "area": float(raw["candidate_area_ratio"]),
                "dice": float(raw["candidate_dice"]),
                "g1": float(raw["g1_logit"]),
                "upstream": float(raw["upstream_score"]),
                "baseline_selected": int(raw["is_baseline_selected"]),
            }
            for column in CAUSAL_COLUMNS:
                row[column] = float(raw[column])
            bags[row["image_id"]].append(row)
    if len(bags) != 184:
        raise ValueError(f"expected 184 tumor bags, got {len(bags)}")

    feature_blocks: dict[str, list[dict[str, float]]] = defaultdict(list)
    image_records: list[dict[str, Any]] = []
    candidate_state: dict[str, dict[str, Any]] = {}
    cross_view_hashes: dict[str, str] = {}

    for image_id, rows in sorted(bags.items()):
        group_ids = {str(row["group_id"]) for row in rows}
        true_groups = {str(row["size_group"]) for row in rows}
        if len(group_ids) != 1 or len(true_groups) != 1:
            raise ValueError(f"inconsistent bag metadata: {image_id}")
        true_group = next(iter(true_groups))
        if true_group not in GROUP_TO_INDEX:
            raise ValueError(f"invalid group {true_group}: {image_id}")

        g1 = np.asarray([row["g1"] for row in rows], dtype=np.float64)
        upstream = np.asarray([row["upstream"] for row in rows], dtype=np.float64)
        area = np.asarray([row["area"] for row in rows], dtype=np.float64)
        dice = np.asarray([row["dice"] for row in rows], dtype=np.float64)
        baseline_scores = rank_fusion_scores(g1, upstream)
        area_rank = average_percentile_rank(np.log(np.clip(area, 1.0e-8, None)))
        baseline_index = stable_select(baseline_scores, g1)
        frozen_indices = [index for index, row in enumerate(rows) if row["baseline_selected"] == 1]
        if frozen_indices != [baseline_index]:
            raise ValueError(f"baseline did not reproduce: {image_id}")
        source = np.asarray([SOURCES.index(str(row["source"])) for row in rows], dtype=np.int64)

        evidence: dict[str, float] = {"log_candidate_count": float(np.log1p(len(rows)))}
        for name, values in (("g1", g1), ("upstream", upstream), ("fusion", baseline_scores)):
            add_stats(evidence, name, values)
        evidence_weights = softmax(baseline_scores / 0.10)
        evidence["fusion_entropy"] = normalized_entropy(evidence_weights)
        evidence["fusion_effective_fraction"] = float(
            1.0 / np.sum(evidence_weights**2) / len(evidence_weights)
        )
        order = np.argsort(-baseline_scores, kind="stable")
        evidence["fusion_top1_top2_gap"] = float(
            baseline_scores[order[0]] - baseline_scores[order[1]] if len(order) > 1 else 0.0
        )
        evidence["fusion_top1_top5_gap"] = float(
            baseline_scores[order[0]] - baseline_scores[order[min(4, len(order) - 1)]]
        )
        evidence["g1_upstream_rank_correlation"] = float(
            np.corrcoef(average_percentile_rank(g1), average_percentile_rank(upstream))[0, 1]
        )

        area_features: dict[str, float] = {}
        log_area = np.log(np.clip(area, 1.0e-8, None))
        add_stats(area_features, "log_area", log_area)
        area_features["baseline_selected_log_area"] = float(log_area[baseline_index])
        area_features["top5_median_log_area"] = float(np.median(log_area[order[: min(5, len(order))]]))
        area_features["fusion_weighted_log_area"] = float(evidence_weights @ log_area)
        area_features["fusion_log_area_rank_correlation"] = float(
            np.corrcoef(average_percentile_rank(baseline_scores), area_rank)[0, 1]
        )

        source_best_log_areas: list[float] = []
        source_best_scores: list[float] = []
        for source_index, source_name in enumerate(SOURCES):
            members = np.flatnonzero(source == source_index)
            evidence[f"source_{source_name}_fraction"] = float(len(members) / len(rows))
            if len(members) == 0:
                for name in ("g1", "upstream", "fusion"):
                    add_stats(evidence, f"source_{source_name}_{name}", [])
                add_stats(area_features, f"source_{source_name}_log_area", [])
                continue
            for name, values in (("g1", g1), ("upstream", upstream), ("fusion", baseline_scores)):
                add_stats(evidence, f"source_{source_name}_{name}", values[members])
            add_stats(area_features, f"source_{source_name}_log_area", log_area[members])
            source_best = members[
                max(
                    range(len(members)),
                    key=lambda local: (baseline_scores[members[local]], g1[members[local]], -members[local]),
                )
            ]
            source_best_scores.append(float(baseline_scores[source_best]))
            source_best_log_areas.append(float(log_area[source_best]))
        evidence["source_best_score_spread"] = float(np.ptp(source_best_scores))
        evidence["source_best_score_std"] = float(np.std(source_best_scores))
        area_features["source_best_median_log_area"] = float(np.median(source_best_log_areas))
        area_features["source_best_log_area_spread"] = float(np.ptp(source_best_log_areas))

        causal: dict[str, float] = {}
        derived_causal: dict[str, np.ndarray] = {
            column: np.asarray([row[column] for row in rows], dtype=np.float64)
            for column in CAUSAL_COLUMNS
        }
        derived_causal["matched_minus_random_logit"] = (
            derived_causal["matched_logit_delta"] - derived_causal["random_logit_delta"]
        )
        derived_causal["matched_minus_random_class_contrast"] = (
            derived_causal["matched_class_contrast_delta"]
            - derived_causal["random_class_contrast_delta"]
        )
        for name, values in derived_causal.items():
            add_stats(causal, name, values)
            causal[f"{name}_fusion_weighted_mean"] = float(evidence_weights @ values)

        cross_view_path = args.cross_view_score_dir / f"{Path(image_id).stem}.npz"
        if not cross_view_path.is_file():
            raise FileNotFoundError(cross_view_path)
        cross_view_hashes[image_id] = sha256_file(cross_view_path)
        cross_view: dict[str, float] = {}
        with np.load(cross_view_path, allow_pickle=False) as payload:
            indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
            expected_indices = np.asarray(
                [int(row["candidate_index"]) for row in rows], dtype=np.int64
            )
            if not np.array_equal(indices, expected_indices):
                raise ValueError(f"cross-view candidate mismatch: {image_id}")
            full_residual = np.asarray(payload["full_residual"], dtype=np.float64)
            control_residual = np.asarray(payload["control_residual"], dtype=np.float64)
        residual_sets = {
            "cross_view_full": full_residual,
            "cross_view_control": control_residual,
            "cross_view_full_minus_control": full_residual - control_residual,
        }
        for name, values in residual_sets.items():
            add_stats(cross_view, name, values)
            cross_view[f"{name}_fusion_weighted_mean"] = float(evidence_weights @ values)
            for source_index, source_name in enumerate(SOURCES):
                add_stats(cross_view, f"{name}_source_{source_name}", values[source == source_index])

        blocks = {
            "area_only": area_features,
            "frozen_scores": evidence,
            "causal_diagnostics": {**causal, **cross_view},
            "frozen_scores_plus_area": {**evidence, **area_features},
            "frozen_scores_plus_causal": {**evidence, **causal, **cross_view},
            "all_annotation_free": {**evidence, **area_features, **causal, **cross_view},
        }
        for block_name, block in blocks.items():
            if not block or not all(np.isfinite(list(block.values()))):
                raise ValueError(f"invalid features in {block_name}: {image_id}")
            feature_blocks[block_name].append(block)

        image_records.append(
            {
                "image_id": image_id,
                "group_id": next(iter(group_ids)),
                "true_size_group": true_group,
                "baseline_dice": float(dice[baseline_index]),
            }
        )
        candidate_state[image_id] = {
            "rows": rows,
            "g1": g1,
            "baseline_scores": baseline_scores,
            "area_rank": area_rank,
            "dice": dice,
        }

    counts = Counter(record["true_size_group"] for record in image_records)
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise ValueError(f"unexpected cohort: {counts}")
    targets = np.asarray([GROUP_TO_INDEX[record["true_size_group"]] for record in image_records])
    group_ids = [str(record["group_id"]) for record in image_records]
    folds = build_group_folds(group_ids, targets)
    true_groups = [str(record["true_size_group"]) for record in image_records]
    baseline_dice = np.asarray([float(record["baseline_dice"]) for record in image_records])

    # Fixed retrospective reference bounds from exactly the same candidate bags.
    expert_dice_matrix = np.zeros((len(image_records), len(EXPERT_BETAS)), dtype=np.float64)
    for index, record in enumerate(image_records):
        state = candidate_state[str(record["image_id"])]
        for expert_index, beta in enumerate(EXPERT_BETAS):
            scores = state["baseline_scores"] + beta * (state["area_rank"] - 0.5)
            selected = stable_select(scores, state["g1"])
            expert_dice_matrix[index, expert_index] = float(state["dice"][selected])
    true_group_dice = expert_dice_matrix[np.arange(len(image_records)), targets]
    expert_oracle_dice = expert_dice_matrix.max(axis=1)
    best_expert_targets = np.zeros(len(image_records), dtype=np.int64)
    for index, values in enumerate(expert_dice_matrix):
        tied = np.flatnonzero(np.isclose(values, values.max(), atol=1.0e-12, rtol=0.0))
        # Prefer the immutable baseline expert on exact ties, then the milder expert.
        best_expert_targets[index] = 1 if 1 in tied else int(tied[0])

    output_rows: list[dict[str, Any]] = []
    block_results: dict[str, Any] = {}
    univariate_correlations: dict[str, list[dict[str, float | str]]] = {}

    def evaluate_gate(
        block_name: str,
        gate_definition: str,
        probabilities: np.ndarray,
        gate_targets: np.ndarray,
    ) -> dict[str, Any]:
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-8):
            raise AssertionError("invalid OOF probabilities")
        hard_predictions = probabilities.argmax(axis=1)
        soft_betas = probabilities @ EXPERT_BETAS
        hard_dice = np.zeros(len(image_records), dtype=np.float64)
        soft_dice = np.zeros(len(image_records), dtype=np.float64)
        hard_selected_area = np.zeros(len(image_records), dtype=np.float64)
        soft_selected_area = np.zeros(len(image_records), dtype=np.float64)
        for index, record in enumerate(image_records):
            state = candidate_state[str(record["image_id"])]
            hard_beta = EXPERT_BETAS[hard_predictions[index]]
            hard_scores = state["baseline_scores"] + hard_beta * (state["area_rank"] - 0.5)
            soft_scores = state["baseline_scores"] + soft_betas[index] * (state["area_rank"] - 0.5)
            hard_selected = stable_select(hard_scores, state["g1"])
            soft_selected = stable_select(soft_scores, state["g1"])
            hard_dice[index] = state["dice"][hard_selected]
            soft_dice[index] = state["dice"][soft_selected]
            hard_selected_area[index] = float(state["rows"][hard_selected]["area"])
            soft_selected_area[index] = float(state["rows"][soft_selected]["area"])
            output_rows.append(
                {
                    "feature_block": block_name,
                    "gate_definition": gate_definition,
                    "image_id": record["image_id"],
                    "group_id": record["group_id"],
                    "fold": int(folds[index]),
                    "true_size_group": record["true_size_group"],
                    "gate_target_expert": GROUPS[int(gate_targets[index])],
                    "prob_small": float(probabilities[index, 0]),
                    "prob_medium": float(probabilities[index, 1]),
                    "prob_large": float(probabilities[index, 2]),
                    "hard_predicted_expert": GROUPS[int(hard_predictions[index])],
                    "soft_beta": float(soft_betas[index]),
                    "baseline_dice": float(baseline_dice[index]),
                    "hard_routed_dice": float(hard_dice[index]),
                    "soft_routed_dice": float(soft_dice[index]),
                    "hard_selected_area": float(hard_selected_area[index]),
                    "soft_selected_area": float(soft_selected_area[index]),
                }
            )
        confusion = Counter(
            (GROUPS[int(truth)], GROUPS[int(prediction)])
            for truth, prediction in zip(gate_targets, hard_predictions, strict=True)
        )
        recalls = [
            float(np.mean(hard_predictions[gate_targets == class_index] == class_index))
            for class_index in range(len(GROUPS))
        ]
        expected_ordinal = probabilities @ np.arange(len(GROUPS), dtype=np.float64)
        abstention_results: dict[str, Any] = {}
        for threshold in ABSTENTION_THRESHOLDS:
            confidence_over_baseline = (
                probabilities[np.arange(len(image_records)), hard_predictions]
                - probabilities[:, 1]
            )
            changed = (hard_predictions != 1) & (confidence_over_baseline >= threshold)
            routed = np.where(changed, hard_dice, baseline_dice)
            abstention_results[f"{threshold:g}"] = {
                "changed_images": int(changed.sum()),
                "dice": metrics(routed, true_groups),
                "delta_overall": float(routed.mean() - baseline_dice.mean()),
            }
        return {
            "target_accuracy": float(np.mean(hard_predictions == gate_targets)),
            "target_balanced_accuracy": float(np.mean(recalls)),
            "target_ordinal_spearman": float(spearmanr(expected_ordinal, gate_targets).statistic),
            "confusion": {
                f"{truth}->{prediction}": int(value)
                for (truth, prediction), value in sorted(confusion.items())
            },
            "hard_routed_dice": metrics(hard_dice, true_groups),
            "soft_routed_dice": metrics(soft_dice, true_groups),
            "hard_delta_overall": float(hard_dice.mean() - baseline_dice.mean()),
            "soft_delta_overall": float(soft_dice.mean() - baseline_dice.mean()),
            "confidence_abstention": abstention_results,
        }

    for block_name, feature_dicts in feature_blocks.items():
        feature_names = sorted(feature_dicts[0])
        if any(sorted(feature_dict) != feature_names for feature_dict in feature_dicts):
            raise ValueError(f"inconsistent features: {block_name}")
        matrix = np.asarray(
            [[feature_dict[name] for name in feature_names] for feature_dict in feature_dicts],
            dtype=np.float64,
        )
        gate_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for gate_definition, gate_targets, feature_limit in (
            ("true_size_all_features", targets, None),
            ("true_size_nested_top8", targets, FEATURE_LIMIT),
            ("best_expert_nested_top8", best_expert_targets, FEATURE_LIMIT),
        ):
            probabilities = np.zeros((len(image_records), len(GROUPS)), dtype=np.float64)
            for fold in range(FOLD_COUNT):
                train = folds != fold
                held_out = folds == fold
                probabilities[held_out] = fit_predict_balanced_ridge(
                    matrix[train], gate_targets[train], matrix[held_out], feature_limit
                )
            gate_probabilities[gate_definition] = (probabilities, gate_targets)

        predicted_utilities = np.zeros_like(expert_dice_matrix)
        for fold in range(FOLD_COUNT):
            train = folds != fold
            held_out = folds == fold
            predicted_utilities[held_out] = fit_predict_utility_ridge(
                matrix[train], expert_dice_matrix[train], matrix[held_out], FEATURE_LIMIT
            )
        gate_probabilities["expert_utility_nested_top8"] = (
            utility_to_probabilities(predicted_utilities),
            best_expert_targets,
        )
        block_results[block_name] = {
            "feature_count": len(feature_names),
            "gate_results": {
                gate_definition: evaluate_gate(
                    block_name, gate_definition, probabilities, gate_targets
                )
                for gate_definition, (probabilities, gate_targets) in gate_probabilities.items()
            },
        }
        correlations = []
        for feature_index, feature_name in enumerate(feature_names):
            statistic = float(spearmanr(matrix[:, feature_index], targets).statistic)
            if np.isfinite(statistic):
                correlations.append({"feature": feature_name, "spearman": statistic})
        univariate_correlations[block_name] = sorted(
            correlations, key=lambda item: (-abs(float(item["spearman"])), str(item["feature"]))
        )[:20]

    # Post-discovery nested confirmation of the only exploratory policy that
    # exceeded baseline: frozen-score features, true-size target, top-8 linear
    # gate, and confidence abstention.  Each outer fold selects its threshold
    # solely from four-fold OOF predictions inside the outer training split.
    confirm_feature_dicts = feature_blocks["frozen_scores"]
    confirm_feature_names = sorted(confirm_feature_dicts[0])
    confirm_matrix = np.asarray(
        [
            [feature_dict[name] for name in confirm_feature_names]
            for feature_dict in confirm_feature_dicts
        ],
        dtype=np.float64,
    )
    nested_routed_dice = baseline_dice.copy()
    nested_fold_details: dict[str, Any] = {}
    for outer_fold in range(FOLD_COUNT):
        outer_train_indices = np.flatnonzero(folds != outer_fold)
        outer_held_indices = np.flatnonzero(folds == outer_fold)
        inner_targets = targets[outer_train_indices]
        inner_group_ids = [group_ids[index] for index in outer_train_indices]
        inner_folds = build_group_folds(inner_group_ids, inner_targets, fold_count=4)
        inner_probabilities = np.zeros((len(outer_train_indices), len(GROUPS)), dtype=np.float64)
        for inner_fold in range(4):
            inner_train = inner_folds != inner_fold
            inner_held = inner_folds == inner_fold
            inner_probabilities[inner_held] = fit_predict_balanced_ridge(
                confirm_matrix[outer_train_indices][inner_train],
                inner_targets[inner_train],
                confirm_matrix[outer_train_indices][inner_held],
                FEATURE_LIMIT,
            )
        inner_predictions = inner_probabilities.argmax(axis=1)
        inner_confidence = (
            inner_probabilities[np.arange(len(outer_train_indices)), inner_predictions]
            - inner_probabilities[:, 1]
        )
        threshold_scores: list[tuple[float, float, int]] = []
        for threshold in ABSTENTION_THRESHOLDS:
            changed = (inner_predictions != 1) & (inner_confidence >= threshold)
            inner_action_dice = expert_dice_matrix[
                outer_train_indices, inner_predictions
            ]
            routed = np.where(changed, inner_action_dice, baseline_dice[outer_train_indices])
            threshold_scores.append((float(routed.mean()), float(threshold), int(changed.sum())))
        # Conservative higher-threshold tie break.
        selected_score, selected_threshold, inner_changed = max(
            threshold_scores, key=lambda item: (item[0], item[1])
        )

        outer_probabilities = fit_predict_balanced_ridge(
            confirm_matrix[outer_train_indices],
            targets[outer_train_indices],
            confirm_matrix[outer_held_indices],
            FEATURE_LIMIT,
        )
        outer_predictions = outer_probabilities.argmax(axis=1)
        outer_confidence = (
            outer_probabilities[np.arange(len(outer_held_indices)), outer_predictions]
            - outer_probabilities[:, 1]
        )
        outer_changed = (outer_predictions != 1) & (outer_confidence >= selected_threshold)
        outer_action_dice = expert_dice_matrix[outer_held_indices, outer_predictions]
        nested_routed_dice[outer_held_indices] = np.where(
            outer_changed,
            outer_action_dice,
            baseline_dice[outer_held_indices],
        )
        nested_fold_details[str(outer_fold)] = {
            "selected_threshold": selected_threshold,
            "inner_selected_dice": selected_score,
            "inner_changed_images": inner_changed,
            "outer_images": len(outer_held_indices),
            "outer_changed_images": int(outer_changed.sum()),
        }
    nested_confirmation = {
        "feature_block": "frozen_scores",
        "gate_definition": "true_size_nested_top8_with_inner_selected_abstention",
        "dice": metrics(nested_routed_dice, true_groups),
        "delta_overall": float(nested_routed_dice.mean() - baseline_dice.mean()),
        "changed_images": int(np.sum(np.abs(nested_routed_dice - baseline_dice) > 1.0e-12)),
        "outer_fold_details": nested_fold_details,
        "paired_group_bootstrap": paired_group_bootstrap(
            nested_routed_dice, baseline_dice, group_ids
        ),
    }

    fold_report = {
        str(fold): {
            "images": int(np.sum(folds == fold)),
            "groups": len({group_ids[index] for index in np.flatnonzero(folds == fold)}),
            "class_counts": {
                group: int(np.sum((folds == fold) & (targets == group_index)))
                for group_index, group in enumerate(GROUPS)
            },
        }
        for fold in range(FOLD_COUNT)
    }
    baseline_metrics = metrics(baseline_dice, true_groups)
    summary = {
        "stage": "rich_gallery_latent_burden_gate_retrospective_v4",
        "input_sha256": actual_hash,
        "cross_view_score_set_sha256": hashlib.sha256(
            "".join(f"{key}:{cross_view_hashes[key]}\n" for key in sorted(cross_view_hashes)).encode("utf-8")
        ).hexdigest(),
        "cohort": {"tumor": 184, "small": 94, "medium": 72, "large": 18},
        "protocol": {
            "folds": FOLD_COUNT,
            "fold_unit": "group_id",
            "models": [
                "class-balanced linear one-hot ridge",
                "linear expert-utility ridge",
            ],
            "ridge_lambda": RIDGE_LAMBDA,
            "nested_feature_limit": FEATURE_LIMIT,
            "utility_softmax_temperature": UTILITY_TEMPERATURE,
            "confidence_abstention_thresholds": list(ABSTENTION_THRESHOLDS),
            "expert_betas": EXPERT_BETAS.tolist(),
            "exploratory_confidence_threshold_sweep": True,
            "nested_confirmation_selects_threshold_inside_outer_training_only": True,
        },
        "fold_report": fold_report,
        "baseline_dice": baseline_metrics,
        "true_group_routing_dice": metrics(true_group_dice, true_groups),
        "three_expert_per_image_oracle_dice": metrics(expert_oracle_dice, true_groups),
        "best_expert_target_distribution": {
            "overall": {
                GROUPS[expert_index]: int(np.sum(best_expert_targets == expert_index))
                for expert_index in range(len(GROUPS))
            },
            "by_true_size_group": {
                true_group: {
                    GROUPS[expert_index]: int(
                        np.sum(
                            (targets == GROUP_TO_INDEX[true_group])
                            & (best_expert_targets == expert_index)
                        )
                    )
                    for expert_index in range(len(GROUPS))
                }
                for true_group in GROUPS
            },
        },
        "fixed_expert_delta_statistics": {
            GROUPS[expert_index]: {
                "beta": float(EXPERT_BETAS[expert_index]),
                "mean_delta": float(
                    np.mean(expert_dice_matrix[:, expert_index] - baseline_dice)
                ),
                "improved_images": int(
                    np.sum(expert_dice_matrix[:, expert_index] > baseline_dice + 1.0e-12)
                ),
                "harmed_images": int(
                    np.sum(expert_dice_matrix[:, expert_index] < baseline_dice - 1.0e-12)
                ),
                "unchanged_images": int(
                    np.sum(
                        np.isclose(
                            expert_dice_matrix[:, expert_index],
                            baseline_dice,
                            atol=1.0e-12,
                            rtol=0.0,
                        )
                    )
                ),
                "mean_positive_gain": float(
                    np.mean(
                        (expert_dice_matrix[:, expert_index] - baseline_dice)[
                            expert_dice_matrix[:, expert_index] > baseline_dice + 1.0e-12
                        ]
                    )
                    if np.any(expert_dice_matrix[:, expert_index] > baseline_dice + 1.0e-12)
                    else 0.0
                ),
                "mean_negative_harm": float(
                    np.mean(
                        (expert_dice_matrix[:, expert_index] - baseline_dice)[
                            expert_dice_matrix[:, expert_index] < baseline_dice - 1.0e-12
                        ]
                    )
                    if np.any(expert_dice_matrix[:, expert_index] < baseline_dice - 1.0e-12)
                    else 0.0
                ),
            }
            for expert_index in range(len(GROUPS))
        },
        "feature_block_results": block_results,
        "nested_confirmation": nested_confirmation,
        "top_univariate_correlations": univariate_correlations,
        "decision": {},
        "academic_status": {
            "candidate_scores_frozen_before_gt": True,
            "gt_size_group_used_to_fit_retrospective_oof_gate": True,
            "gt_candidate_dice_used_for_retrospective_best_expert_and_utility_targets": True,
            "gate_is_deployable": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
    }
    candidates: list[tuple[float, str, str, str]] = []
    for block_name, block_result in block_results.items():
        for gate_definition, gate_result in block_result["gate_results"].items():
            for routing in ("hard", "soft"):
                candidates.append(
                    (
                        float(gate_result[f"{routing}_routed_dice"]["overall"]),
                        block_name,
                        gate_definition,
                        routing,
                    )
                )
            for threshold, abstention in gate_result["confidence_abstention"].items():
                candidates.append(
                    (
                        float(abstention["dice"]["overall"]),
                        block_name,
                        gate_definition,
                        f"confidence_abstention_{threshold}",
                    )
                )
    best_dice, best_block, best_gate_definition, best_routing = max(candidates)
    summary["decision"] = {
        "exploratory_best_oof_feature_block": best_block,
        "exploratory_best_oof_gate_definition": best_gate_definition,
        "exploratory_best_oof_routing": best_routing,
        "exploratory_best_oof_dice": best_dice,
        "exploratory_best_oof_delta": best_dice - float(baseline_metrics["overall"]),
        "nested_confirmation_dice": float(nested_confirmation["dice"]["overall"]),
        "nested_confirmation_delta": float(nested_confirmation["delta_overall"]),
        "existing_annotation_free_bag_features_show_material_scale_signal": bool(
            nested_confirmation["delta_overall"] >= 0.005
            and nested_confirmation["paired_group_bootstrap"]["probability_delta_positive"] >= 0.90
        ),
        "deploy_existing_gate": False,
        "reason": (
            "all gates remain retrospective because validation spatial size labels train the gate; "
            "nested confirmation only tests whether the frozen feature family contains route signal"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "oof_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    features_path = args.output_dir / "per_image_annotation_free_features.csv"
    all_feature_names = sorted(
        {name for blocks in feature_blocks.values() for block in blocks for name in block}
    )
    with features_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "group_id", "true_size_group", *all_feature_names],
        )
        writer.writeheader()
        for index, record in enumerate(image_records):
            flattened: dict[str, Any] = {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "true_size_group": record["true_size_group"],
            }
            for block in feature_blocks.values():
                flattened.update(block[index])
            writer.writerow(flattened)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "input_sha256": actual_hash,
        "summary_sha256": sha256_file(summary_path),
        "oof_predictions_sha256": sha256_file(predictions_path),
        "per_image_features_sha256": sha256_file(features_path),
        "tumor_images": 184,
        "oof_prediction_rows": len(output_rows),
        "group_leakage": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
