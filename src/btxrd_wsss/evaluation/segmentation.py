from __future__ import annotations

import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def _surface_distances(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_surface, target_surface = _surface(prediction), _surface(target)
    if not pred_surface.any() or not target_surface.any():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    distance_to_target = ndimage.distance_transform_edt(~target_surface)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface)
    return distance_to_target[pred_surface], distance_to_prediction[target_surface]


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | bool]:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must share the native grid")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    union = tp + fp + fn
    pred_area, target_area = int(prediction.sum()), int(target.sum())
    both_empty = pred_area == target_area == 0
    dice = 1.0 if both_empty else 2 * tp / max(1, pred_area + target_area)
    iou = 1.0 if both_empty else tp / max(1, union)
    d1, d2 = _surface_distances(prediction, target)
    surface_defined = bool(d1.size and d2.size)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "predicted_area_ratio": float(pred_area / prediction.size),
        "target_area_ratio": float(target_area / target.size),
        "complete_miss": bool(target_area > 0 and tp == 0),
        "surface_defined": surface_defined,
        "hd95": float(max(np.percentile(d1, 95), np.percentile(d2, 95)))
        if surface_defined
        else float("nan"),
        "assd": float((d1.mean() + d2.mean()) / 2) if surface_defined else float("nan"),
    }
