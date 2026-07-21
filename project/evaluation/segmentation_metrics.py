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


LESION_IOU_THRESHOLDS = (0.10, 0.25, 0.50)


def _iou_key(threshold: float) -> str:
    return f"iou{int(round(threshold * 100)):02d}"


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


def _lesion_detection(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    if ndimage is None:
        empty = {
            "gt_lesions": 0,
            "detected_lesions_any_overlap": 0,
            "predicted_lesions": 0,
            "matched_predicted_lesions_any_overlap": 0,
        }
        empty.update({f"lesion_tp_one_to_one_{_iou_key(t)}": 0 for t in LESION_IOU_THRESHOLDS})
        return empty
    structure = ndimage.generate_binary_structure(2, 2)
    target_labels, target_count = ndimage.label(target, structure=structure)
    pred_labels, pred_count = ndimage.label(pred, structure=structure)
    detected = sum(bool((pred & (target_labels == label)).any()) for label in range(1, target_count + 1))
    matched_predictions = sum(
        bool((target & (pred_labels == label)).any()) for label in range(1, pred_count + 1)
    )

    pairwise_iou = np.zeros((target_count, pred_count), dtype=np.float64)
    for gt_label in range(1, target_count + 1):
        gt_component = target_labels == gt_label
        for pred_label in range(1, pred_count + 1):
            pred_component = pred_labels == pred_label
            intersection = int(np.logical_and(gt_component, pred_component).sum())
            union = int(np.logical_or(gt_component, pred_component).sum())
            pairwise_iou[gt_label - 1, pred_label - 1] = intersection / union if union else 0.0

    def maximum_matches(threshold: float) -> int:
        adjacency = [
            [pred_index for pred_index in range(pred_count) if pairwise_iou[gt_index, pred_index] >= threshold]
            for gt_index in range(target_count)
        ]
        matched_gt_for_pred = [-1] * pred_count

        def augment(gt_index: int, seen: set[int]) -> bool:
            for pred_index in adjacency[gt_index]:
                if pred_index in seen:
                    continue
                seen.add(pred_index)
                if matched_gt_for_pred[pred_index] < 0 or augment(matched_gt_for_pred[pred_index], seen):
                    matched_gt_for_pred[pred_index] = gt_index
                    return True
            return False

        return sum(augment(gt_index, set()) for gt_index in range(target_count))

    one_to_one = {
        f"lesion_tp_one_to_one_{_iou_key(threshold)}": int(maximum_matches(threshold))
        for threshold in LESION_IOU_THRESHOLDS
    }
    return {
        "gt_lesions": int(target_count),
        "detected_lesions_any_overlap": int(detected),
        "predicted_lesions": int(pred_count),
        "matched_predicted_lesions_any_overlap": int(matched_predictions),
        **one_to_one,
    }


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
    lesion_metrics = _lesion_detection(pred, target)
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
        **lesion_metrics,
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
    detected = sum(int(row.get("detected_lesions_any_overlap", 0)) for row in tumor)
    predicted_lesions = sum(int(row.get("predicted_lesions", 0)) for row in rows)
    matched_predictions = sum(int(row.get("matched_predicted_lesions_any_overlap", 0)) for row in rows)
    one_to_one_tp = {
        _iou_key(threshold): sum(
            int(row.get(f"lesion_tp_one_to_one_{_iou_key(threshold)}", 0)) for row in rows
        )
        for threshold in LESION_IOU_THRESHOLDS
    }
    boundary_eligible = sum(
        isinstance(row.get("hd95_px"), (int, float))
        and math.isfinite(float(row.get("hd95_px", float("nan"))))
        for row in tumor
    )
    complete_misses = sum(not bool(row.get("predicted_positive")) for row in tumor)
    multifocal_images = sum(int(row.get("gt_lesions", 0)) > 1 for row in tumor)
    component_histogram: dict[str, int] = {}
    for row in tumor:
        key = str(int(row.get("gt_lesions", 0)))
        component_histogram[key] = component_histogram.get(key, 0) + 1
    normal_empty = sum(not bool(row.get("predicted_positive")) for row in normal)
    summary = {
        "images": len(rows),
        "tumor_images": len(tumor),
        "normal_images": len(normal),
        "main_population": "tumor images only",
        "mean_tumor_dice": _finite_mean(tumor, "dice"),
        "mean_tumor_iou": _finite_mean(tumor, "iou"),
        "mean_tumor_precision": _finite_mean(tumor, "precision"),
        "mean_tumor_recall": _finite_mean(tumor, "recall"),
        "mean_tumor_hd95_px_conditional_defined": _finite_mean(tumor, "hd95_px"),
        "mean_tumor_assd_px_conditional_defined": _finite_mean(tumor, "assd_px"),
        "boundary_metric_definition": (
            "conditional mean over tumor images with both GT and prediction non-empty; "
            "distances are pixels on the resized evaluation grid (not mm); "
            "complete misses are excluded and counted separately"
        ),
        "boundary_metric_eligible_tumor_images": boundary_eligible,
        "boundary_metric_excluded_tumor_images": len(tumor) - boundary_eligible,
        "boundary_metric_complete_misses": complete_misses,
        "tumor_non_empty_prediction_rate": _safe_ratio(
            sum(bool(row.get("predicted_positive")) for row in tumor), len(tumor)
        ),
        "tumor_overlap_detection_rate": _safe_ratio(
            sum(int(row.get("tp_pixels", 0)) > 0 for row in tumor), len(tumor)
        ),
        "lesion_any_overlap_recall": _safe_ratio(detected, gt_lesions),
        "lesion_any_overlap_precision": _safe_ratio(matched_predictions, predicted_lesions),
        "gt_lesions": gt_lesions,
        "detected_lesions_any_overlap": detected,
        "predicted_lesions": predicted_lesions,
        "multifocal_tumor_images": multifocal_images,
        "multifocal_tumor_image_rate": _safe_ratio(multifocal_images, len(tumor)),
        "gt_component_count_histogram": component_histogram,
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
    for threshold in LESION_IOU_THRESHOLDS:
        key = _iou_key(threshold)
        tp_at_threshold = one_to_one_tp[key]
        summary.update({
            f"lesion_one_to_one_{key}_recall": _safe_ratio(tp_at_threshold, gt_lesions),
            f"lesion_one_to_one_{key}_precision": _safe_ratio(tp_at_threshold, predicted_lesions),
            f"lesion_one_to_one_{key}_f1": _safe_ratio(
                2 * tp_at_threshold, gt_lesions + predicted_lesions
            ),
            f"lesion_tp_one_to_one_{key}": tp_at_threshold,
        })
    summary["lesion_matching_definition"] = (
        "maximum-cardinality one-to-one connected-component matching at IoU thresholds "
        + ", ".join(f"{threshold:.2f}" for threshold in LESION_IOU_THRESHOLDS)
    )
    return summary


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
        "mean_tumor_hd95_px_conditional_defined",
        "mean_tumor_assd_px_conditional_defined",
        "lesion_one_to_one_iou10_recall",
        "lesion_one_to_one_iou10_precision",
        "lesion_one_to_one_iou25_recall",
        "lesion_one_to_one_iou25_precision",
        "lesion_one_to_one_iou50_recall",
        "lesion_one_to_one_iou50_precision",
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
        "group_provenance": "heuristic grouping; not verified patient/case identifiers",
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
