from __future__ import annotations

"""Case-level binary segmentation metrics and grouped uncertainty estimates."""

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - environment validation reports this
    ndimage = None


def _safe_ratio(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def _surface_distances(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if not pred.any() and not target.any():
        return 0.0, 0.0
    if not pred.any() or not target.any():
        return float("nan"), float("nan")
    if ndimage is None:
        return float("nan"), float("nan")
    structure = ndimage.generate_binary_structure(2, 2)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    distance_to_target = ndimage.distance_transform_edt(~target_surface)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
    if distances.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def _lesion_detection(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    if ndimage is None:
        return 0, 0, 0, 0
    structure = ndimage.generate_binary_structure(2, 2)
    target_labels, target_count = ndimage.label(target, structure=structure)
    pred_labels, pred_count = ndimage.label(pred, structure=structure)
    detected = sum(bool((pred & (target_labels == label)).any()) for label in range(1, target_count + 1))
    matched_predictions = sum(
        bool((target & (pred_labels == label)).any()) for label in range(1, pred_count + 1)
    )
    return int(target_count), int(detected), int(pred_count), int(matched_predictions)


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int | bool]:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"pred/target must be matching 2-D masks, got {pred.shape} and {target.shape}")
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    pred_sum = tp + fp
    target_sum = tp + fn
    both_empty = pred_sum == 0 and target_sum == 0
    dice = 1.0 if both_empty else _safe_ratio(2 * tp, pred_sum + target_sum)
    iou = 1.0 if both_empty else _safe_ratio(tp, tp + fp + fn)
    precision = _safe_ratio(tp, tp + fp, empty_value=1.0 if target_sum == 0 else 0.0)
    recall = _safe_ratio(tp, tp + fn, empty_value=1.0)
    specificity = _safe_ratio(tn, tn + fp, empty_value=1.0)
    hd95_px, assd_px = _surface_distances(pred, target)
    gt_lesions, detected_lesions, pred_lesions, matched_pred_lesions = _lesion_detection(pred, target)
    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pixel_specificity": specificity,
        "hd95_px": hd95_px,
        "assd_px": assd_px,
        "gt_area_pixels": target_sum,
        "pred_area_pixels": pred_sum,
        "gt_area_ratio": target_sum / float(target.size),
        "pred_area_ratio": pred_sum / float(pred.size),
        "predicted_positive": pred_sum > 0,
        "gt_positive": target_sum > 0,
        "gt_lesions": gt_lesions,
        "detected_lesions": detected_lesions,
        "predicted_lesions": pred_lesions,
        "matched_predicted_lesions": matched_pred_lesions,
    }


def _finite_mean(rows: Iterable[dict[str, object]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def summarize_segmentation_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    tumor = [row for row in rows if bool(row.get("gt_positive"))]
    normal = [row for row in rows if not bool(row.get("gt_positive"))]
    tp = sum(int(row.get("tp_pixels", 0)) for row in rows)
    fp = sum(int(row.get("fp_pixels", 0)) for row in rows)
    fn = sum(int(row.get("fn_pixels", 0)) for row in rows)
    tn = sum(int(row.get("tn_pixels", 0)) for row in rows)
    gt_lesions = sum(int(row.get("gt_lesions", 0)) for row in tumor)
    detected = sum(int(row.get("detected_lesions", 0)) for row in tumor)
    predicted_lesions = sum(int(row.get("predicted_lesions", 0)) for row in rows)
    matched_predictions = sum(int(row.get("matched_predicted_lesions", 0)) for row in rows)
    normal_empty = sum(not bool(row.get("predicted_positive")) for row in normal)
    return {
        "images": len(rows),
        "tumor_images": len(tumor),
        "normal_images": len(normal),
        "main_population": "tumor images only",
        "mean_tumor_dice": _finite_mean(tumor, "dice"),
        "mean_tumor_iou": _finite_mean(tumor, "iou"),
        "mean_tumor_precision": _finite_mean(tumor, "precision"),
        "mean_tumor_recall": _finite_mean(tumor, "recall"),
        "mean_tumor_hd95_px": _finite_mean(tumor, "hd95_px"),
        "mean_tumor_assd_px": _finite_mean(tumor, "assd_px"),
        "tumor_boundary_metric_failures": sum(
            not isinstance(row.get("hd95_px"), (int, float))
            or not math.isfinite(float(row.get("hd95_px", float("nan"))))
            for row in tumor
        ),
        "tumor_non_empty_prediction_rate": _safe_ratio(
            sum(bool(row.get("predicted_positive")) for row in tumor), len(tumor)
        ),
        "tumor_overlap_detection_rate": _safe_ratio(
            sum(int(row.get("tp_pixels", 0)) > 0 for row in tumor), len(tumor)
        ),
        "lesion_detection_recall": _safe_ratio(detected, gt_lesions),
        "lesion_detection_precision": _safe_ratio(matched_predictions, predicted_lesions),
        "gt_lesions": gt_lesions,
        "detected_lesions": detected,
        "predicted_lesions": predicted_lesions,
        "normal_empty_prediction_rate": _safe_ratio(normal_empty, len(normal), empty_value=float("nan")),
        "normal_false_positive_case_rate": _safe_ratio(
            len(normal) - normal_empty, len(normal), empty_value=float("nan")
        ),
        "pixel_specificity": _safe_ratio(tn, tn + fp, empty_value=float("nan")),
        "pixel_precision": _safe_ratio(tp, tp + fp, empty_value=float("nan")),
        "pixel_recall": _safe_ratio(tp, tp + fn, empty_value=float("nan")),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
    }


def _lesion_size_bucket(row: dict[str, object]) -> str:
    ratio = float(row.get("gt_area_ratio", 0.0) or 0.0)
    if ratio <= 0:
        return "normal"
    if ratio < 0.01:
        return "small_lt_1pct"
    if ratio < 0.05:
        return "medium_1_to_5pct"
    return "large_ge_5pct"


def subgroup_summaries(
    rows: list[dict[str, object]],
    fields: tuple[str, ...] = ("center", "anatomy", "view", "tumor_type_name"),
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    fields_and_values = [("lesion_size", _lesion_size_bucket)] + [
        (field, lambda row, key=field: str(row.get(key, "unknown") or "unknown")) for field in fields
    ]
    for field, value_fn in fields_and_values:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(value_fn(row))].append(row)
        for value, group_rows in sorted(grouped.items()):
            summary = summarize_segmentation_rows(group_rows)
            output.append({"subgroup_field": field, "subgroup_value": value, **summary})
    return output


def bootstrap_group_confidence_intervals(
    rows: list[dict[str, object]],
    *,
    group_key: str = "group_id",
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get(group_key, "") or f"image:{index}")
        grouped[group].append(row)
    group_ids = sorted(grouped)
    if not group_ids:
        raise ValueError("Cannot bootstrap an empty result set")
    metrics = (
        "mean_tumor_dice",
        "mean_tumor_iou",
        "mean_tumor_precision",
        "mean_tumor_recall",
        "mean_tumor_hd95_px",
        "mean_tumor_assd_px",
        "lesion_detection_recall",
        "normal_empty_prediction_rate",
        "normal_false_positive_case_rate",
    )
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled_rows = [row for group in chosen for row in grouped[str(group)]]
        summary = summarize_segmentation_rows(sampled_rows)
        for metric in metrics:
            value = summary.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                samples[metric].append(float(value))
    intervals: dict[str, dict[str, float | int | None]] = {}
    point = summarize_segmentation_rows(rows)
    for metric in metrics:
        values = samples[metric]
        intervals[metric] = {
            "point_estimate": float(point[metric]) if math.isfinite(float(point[metric])) else None,
            "ci95_low": float(np.percentile(values, 2.5)) if values else None,
            "ci95_high": float(np.percentile(values, 97.5)) if values else None,
            "valid_iterations": len(values),
        }
    return {
        "method": "nonparametric bootstrap resampling complete groups with replacement",
        "group_key": group_key,
        "groups": len(group_ids),
        "iterations": iterations,
        "seed": seed,
        "intervals": intervals,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
