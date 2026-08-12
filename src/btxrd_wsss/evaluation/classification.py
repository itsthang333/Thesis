from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Labels and probabilities must be aligned and finite")
    predictions = probabilities >= 0.5
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }
    result["auroc"] = (
        float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan")
    )
    return result
