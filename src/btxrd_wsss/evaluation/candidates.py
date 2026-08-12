from __future__ import annotations

import numpy as np

from btxrd_wsss.evaluation.segmentation import segmentation_metrics
from btxrd_wsss.types import CandidateMask


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
