import numpy as np

from btxrd_wsss.artifacts import load_gallery, load_source_maps, save_gallery, save_source_maps
from btxrd_wsss.types import CandidateMask


def test_source_maps_and_candidate_gallery_round_trip(tmp_path) -> None:
    maps = {
        "hrnet_full": np.zeros((7, 9), np.float32),
        "hrnet_tile": np.ones((7, 9), np.float32),
        "biomedclip": np.full((7, 9), 0.5, np.float32),
    }
    confidences = {"hrnet_full": 0.1, "hrnet_tile": 0.8, "biomedclip": 0.4}
    save_source_maps(tmp_path, "image", maps, confidences)
    restored_maps, restored_confidences = load_source_maps(tmp_path, "image")
    np.testing.assert_array_equal(restored_maps["hrnet_tile"], maps["hrnet_tile"])
    assert abs(restored_confidences["biomedclip"] - 0.4) < 1e-6

    mask = np.zeros((7, 9), bool)
    mask[2:4, 3:6] = True
    candidate = CandidateMask(
        candidate_id="c",
        mask=mask,
        proposal_id="p",
        proposal_source="hrnet_tile",
        sam_backend="mock",
        prompt_type="box",
        predicted_iou=0.8,
        stability=0.9,
        roi_scale=1.5,
        metadata={
            "source_component": mask.copy(),
            "source_confidence": 0.7,
            "peak_x": 3,
            "peak_y": 2,
            "branch_thresholds": (0.97, 0.99),
        },
    )
    save_gallery(tmp_path, "image", [candidate])
    restored = load_gallery(tmp_path, "image")
    assert len(restored) == 1
    np.testing.assert_array_equal(restored[0].mask, mask)
    np.testing.assert_array_equal(restored[0].metadata["source_component"], mask)
    assert restored[0].metadata["branch_thresholds"] == [0.97, 0.99]
