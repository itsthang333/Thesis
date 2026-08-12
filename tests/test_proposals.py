import numpy as np

from btxrd_wsss.config import ProposalConfig
from btxrd_wsss.pipeline.proposals import ProposalGenerator, component_tree


def proposal_config() -> ProposalConfig:
    return ProposalConfig(
        hrnet_full_percentiles=[0.5, 0.7, 0.9],
        hrnet_tile_percentiles=[0.5, 0.7, 0.9],
        biomedclip_percentiles=[0.5, 0.7, 0.9],
        minimum_native_area=4,
        max_components_per_threshold=5,
        box_padding=[0.15, 0.4],
        positive_point_counts=[1, 3],
        negative_points=4,
        source_quotas={"hrnet_full": 8, "hrnet_tile": 16, "biomedclip": 8},
    )


def test_nested_threshold_components_collapse_to_one_branch() -> None:
    evidence = np.zeros((32, 32), dtype=np.float32)
    evidence[10:18, 10:18] = 0.6
    evidence[12:16, 12:16] = 0.8
    evidence[13:15, 13:15] = 1
    branches = component_tree(evidence, [0.5, 0.7, 0.9], minimum_area=4, per_threshold=5)
    assert len(branches) == 1
    assert branches[0][1] == 3


def test_small_component_is_kept_once_not_repeated_per_prompt_variant() -> None:
    evidence = np.zeros((32, 32), dtype=np.float32)
    evidence[10:12, 11:13] = 1
    proposals = ProposalGenerator(proposal_config()).from_map(
        evidence, image_id="x", source="hrnet_full", source_view="native", thresholds=[0.5, 0.9]
    )
    assert len(proposals) == 1
    assert proposals[0].component_mask.sum() == 4
    assert len(proposals[0].positive_points) <= 3
