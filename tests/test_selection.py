import numpy as np

from btxrd_wsss.config import SAMConfig, SelectionConfig
from btxrd_wsss.pipeline.sam_gallery import select_diverse_gallery
from btxrd_wsss.pipeline.selection import percentile_ranks
from btxrd_wsss.types import CandidateMask


def test_percentile_rank_singleton_is_half() -> None:
    np.testing.assert_allclose(percentile_ranks([7]), [0.5])


def _candidate(
    identifier: str, source: str, box: tuple[int, int, int, int], shape=(100, 100)
) -> CandidateMask:
    mask = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    component = mask.copy()
    return CandidateMask(
        candidate_id=identifier,
        mask=mask,
        proposal_id=identifier,
        proposal_source=source,
        sam_backend="mock",
        prompt_type="box",
        predicted_iou=0.9,
        stability=0.9,
        roi_scale=1.5,
        metadata={
            "source_component": component,
            "peak_x": x0,
            "peak_y": y0,
            "source_confidence": 0.8,
            "source_score": 0.8,
        },
    )


def test_diversity_gallery_preserves_tiny_candidates() -> None:
    sam = SAMConfig(
        model_type="vit_b",
        checkpoint="x",
        image_size=1024,
        initial_roi_scale=1.5,
        expansion_roi_scale=3.0,
        multimask=True,
        duplicate_iou=0.98,
        expansion_roi_quotas={"hrnet_full": 1, "hrnet_tile": 1, "biomedclip": 1},
        gallery_minimum_quotas={"hrnet_full": 1, "hrnet_tile": 1, "biomedclip": 1},
        maximum_raw_candidates=10,
        maximum_selected_candidates=4,
        minimum_tiny_candidates=1,
        minimum_small_candidates=1,
        tiny_area_ratio=0.001,
        small_area_ratio=0.01,
        diversity_weight=0.2,
    )
    selection = SelectionConfig(
        hrnet_weights={
            "contrast": 0.3,
            "purity": 0.25,
            "coverage": 0.2,
            "sam_quality": 0.15,
            "peak": 0.1,
        },
        biomedclip_weights={"purity": 0.35, "contrast": 0.25, "coverage": 0.2, "sam_quality": 0.2},
        source_confidence_floor=0.5,
        g1_rank_weight=0.5,
        upstream_rank_weight=0.5,
        minimum_mask_area=1,
        maximum_mask_area_ratio=0.5,
        minimum_stability=0.5,
        minimum_component_coverage=0.25,
        add_multifocal_unions=True,
        maximum_union_masks=2,
        maximum_components_per_union=3,
    )
    candidates = [
        _candidate("tiny", "hrnet_tile", (2, 2, 4, 4)),
        _candidate("full", "hrnet_full", (10, 10, 30, 30)),
        _candidate("bio", "biomedclip", (40, 40, 60, 60)),
        _candidate("tile", "hrnet_tile", (65, 65, 90, 90)),
    ]
    maps = {
        source: np.ones((100, 100), np.float32)
        for source in ("hrnet_full", "hrnet_tile", "biomedclip")
    }
    selected = select_diverse_gallery(candidates, maps, sam_config=sam, selection_config=selection)
    assert "tiny" in {item.candidate_id for item in selected}
