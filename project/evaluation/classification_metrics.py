from __future__ import annotations

"""Dependency-light classifier metrics with explicit edge-case conventions."""

import numpy as np
from collections import defaultdict


def confusion_from_predictions(target: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    target = np.asarray(target).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    if target.shape != prediction.shape:
        raise ValueError(f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}")
    return {
        "tp": int(np.logical_and(prediction, target).sum()),
        "fp": int(np.logical_and(prediction, ~target).sum()),
        "fn": int(np.logical_and(~prediction, target).sum()),
        "tn": int(np.logical_and(~prediction, ~target).sum()),
    }


def binary_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "sensitivity": recall,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _validated_binary_inputs(target: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target).astype(bool).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if target.shape != score.shape:
        raise ValueError(f"target/score shape mismatch: {target.shape} vs {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("scores must all be finite")
    return target, score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target, score = _validated_binary_inputs(target, score)
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(score)
    return float((ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """Non-interpolated AP = sum_n (R_n - R_{n-1}) P_n.

    Thresholds advance only after a complete equal-score group, making the
    result invariant to input order when predictions contain ties.
    """
    target, score = _validated_binary_inputs(target, score)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ordered_target = target[order]
    ordered_score = score[order]
    cumulative_tp = np.cumsum(ordered_target, dtype=np.int64)
    cumulative_fp = np.cumsum(~ordered_target, dtype=np.int64)
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), len(score) - 1]
    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = cumulative_fp[group_ends].astype(np.float64)
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def multiclass_summary(
    confusion: np.ndarray,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    supports = confusion.sum(axis=1)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    per_class: list[dict[str, object]] = []
    for index in range(confusion.shape[0]):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class.append({
            "class_index": index,
            "class_name": class_names[index] if class_names and index < len(class_names) else str(index),
            "support": int(supports[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / max(1, total)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports)) if supports.sum() else float("nan"),
        "per_class": per_class,
    }


def classifier_group_bootstrap_confidence_intervals(
    rows: list[dict[str, object]],
    *,
    num_classes: int,
    group_key: str = "group_id",
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Percentile CIs from complete heuristic-group resampling.

    Metrics are recomputed after sampling groups with replacement; images in a
    sampled group remain together. Undefined AUROC/AP replicates are omitted
    and their valid replicate counts are reported explicitly.
    """
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get(group_key, "") or f"image:{index}")
        grouped[group].append(row)
    group_ids = sorted(grouped)
    if not group_ids:
        raise ValueError("cannot bootstrap an empty classifier result set")

    metric_names = (
        "macro_f1",
        "tumor_gate_auroc",
        "tumor_gate_auprc",
        "tumor_gate_sensitivity",
        "tumor_gate_specificity",
    )

    def summarize(sample: list[dict[str, object]]) -> dict[str, float]:
        true_class = np.asarray([int(row["true_class"]) for row in sample], dtype=np.int64)
        pred_class = np.asarray([int(row["predicted_class"]) for row in sample], dtype=np.int64)
        true_tumor = np.asarray([int(row["true_tumor"]) for row in sample], dtype=np.int64)
        pred_tumor = np.asarray([int(row["predicted_tumor"]) for row in sample], dtype=np.int64)
        tumor_score = np.asarray([float(row["tumor_probability"]) for row in sample], dtype=np.float64)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for target, prediction in zip(true_class, pred_class):
            confusion[target, prediction] += 1
        gate_counts = confusion_from_predictions(true_tumor, pred_tumor)
        gate = binary_metrics(gate_counts)
        return {
            "macro_f1": float(multiclass_summary(confusion)["macro_f1"]),
            "tumor_gate_auroc": binary_auroc(true_tumor, tumor_score),
            "tumor_gate_auprc": binary_average_precision(true_tumor, tumor_score),
            "tumor_gate_sensitivity": (
                float(gate["sensitivity"])
                if gate_counts["tp"] + gate_counts["fn"] else float("nan")
            ),
            "tumor_gate_specificity": (
                float(gate["specificity"])
                if gate_counts["tn"] + gate_counts["fp"] else float("nan")
            ),
        }

    point = summarize(rows)
    samples = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled_rows = [row for group in chosen for row in grouped[str(group)]]
        values = summarize(sampled_rows)
        for metric, value in values.items():
            if np.isfinite(value):
                samples[metric].append(float(value))
    intervals = {
        metric: {
            "point_estimate": float(point[metric]) if np.isfinite(point[metric]) else None,
            "ci95_low": float(np.percentile(values, 2.5)) if values else None,
            "ci95_high": float(np.percentile(values, 97.5)) if values else None,
            "valid_iterations": len(values),
        }
        for metric, values in samples.items()
    }
    return {
        "method": "nonparametric percentile bootstrap of complete heuristic groups",
        "group_key": group_key,
        "groups": len(group_ids),
        "iterations": iterations,
        "seed": seed,
        "intervals": intervals,
    }
