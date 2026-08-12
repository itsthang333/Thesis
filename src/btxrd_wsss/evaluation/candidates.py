from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from btxrd_wsss.evaluation.segmentation import segmentation_metrics
from btxrd_wsss.types import CandidateMask


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    return np.asarray(rankdata(values, method="average"), np.float32)


def candidate_quality_diagnostics(
    candidates: list[CandidateMask], target: np.ndarray
) -> dict[str, float | int]:
    """Audit SAM quality heads against annotation; never used by selection or training."""
    if not candidates:
        return {"candidate_count": 0}
    actual_iou = np.asarray(
        [segmentation_metrics(candidate.mask, target)["iou"] for candidate in candidates],
        np.float32,
    )
    predicted_iou = np.clip(
        np.asarray([candidate.predicted_iou for candidate in candidates], np.float32), 0, 1
    )
    stability = np.clip(
        np.asarray([candidate.stability for candidate in candidates], np.float32), 0, 1
    )
    combined = (predicted_iou + stability) / 2
    return {
        "candidate_count": len(candidates),
        "predicted_iou_mae": float(np.abs(predicted_iou - actual_iou).mean()),
        "predicted_iou_correlation": _correlation(predicted_iou, actual_iou),
        "predicted_iou_rank_correlation": _correlation(
            _ranks(predicted_iou), _ranks(actual_iou)
        ),
        "stability_rank_correlation": _correlation(_ranks(stability), _ranks(actual_iou)),
        "combined_rank_correlation": _correlation(_ranks(combined), _ranks(actual_iou)),
        "top_predicted_iou_actual_iou": float(actual_iou[int(np.argmax(predicted_iou))]),
        "top_stability_actual_iou": float(actual_iou[int(np.argmax(stability))]),
        "top_combined_actual_iou": float(actual_iou[int(np.argmax(combined))]),
        "oracle_iou": float(actual_iou.max()),
    }


def candidate_gallery_metrics(
    candidates: list[CandidateMask], target: np.ndarray
) -> dict[str, object]:
    if not candidates:
        return {"candidate_count": 0, "oracle_dice": 0.0, "oracle_iou": 0.0, "source_counts": {}}
    metrics = [segmentation_metrics(candidate.mask, target) for candidate in candidates]
    oracle = max(range(len(candidates)), key=lambda index: float(metrics[index]["dice"]))
    source_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    for candidate in candidates:
        source_counts[candidate.proposal_source] = (
            source_counts.get(candidate.proposal_source, 0) + 1
        )
        backend_counts[candidate.sam_backend] = backend_counts.get(candidate.sam_backend, 0) + 1
    return {
        "candidate_count": len(candidates),
        "oracle_candidate_id": candidates[oracle].candidate_id,
        "oracle_dice": metrics[oracle]["dice"],
        "oracle_iou": metrics[oracle]["iou"],
        "oracle_complete_miss": metrics[oracle]["complete_miss"],
        "source_counts": source_counts,
        "backend_counts": backend_counts,
    }
