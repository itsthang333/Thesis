import numpy as np

from btxrd_wsss.evaluation.candidates import candidate_quality_diagnostics
from btxrd_wsss.types import CandidateMask


def _candidate(identifier: str, size: int, predicted_iou: float) -> CandidateMask:
    mask = np.zeros((10, 10), bool)
    mask[:size, :size] = True
    return CandidateMask(
        candidate_id=identifier,
        mask=mask,
        proposal_id=identifier,
        proposal_source="hrnet_tile",
        sam_backend="sam_med2d_vit_b_roi",
        prompt_type="box+positive+negative",
        predicted_iou=predicted_iou,
        stability=predicted_iou,
        roi_scale=1.5,
        metadata={},
    )


def test_quality_diagnostics_compare_sam_scores_with_actual_iou() -> None:
    target = np.zeros((10, 10), bool)
    target[:3, :3] = True
    diagnostics = candidate_quality_diagnostics(
        [_candidate("poor", 1, 0.1), _candidate("best", 3, 0.9)], target
    )
    assert diagnostics["oracle_iou"] == 1.0
    assert diagnostics["top_combined_actual_iou"] == 1.0
    assert np.isclose(diagnostics["predicted_iou_rank_correlation"], 1.0)
