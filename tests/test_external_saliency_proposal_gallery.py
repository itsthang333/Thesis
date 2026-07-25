from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from generate_pseudo_masks import build_external_saliency_proposal_gallery
from pseudo.tumor_morphology import TumorComponent


def component(component_id: int) -> TumorComponent:
    return TumorComponent(
        component_id=component_id,
        mask=np.ones((4, 4), dtype=np.uint8),
        score=1.0,
        bbox=(0, 0, 3, 3),
        positive_points=((1, 1),),
        negative_points=(),
    )


class ExternalSaliencyProposalGalleryTests(unittest.TestCase):
    def test_percentile_galleries_receive_contiguous_component_ids(self) -> None:
        with patch(
            "generate_pseudo_masks.morphology.build_class_conditioned_components",
            side_effect=[
                (None, None, [component(0), component(1)]),
                (None, None, [component(0)]),
                (None, None, [component(0), component(1)]),
            ],
        ):
            result = build_external_saliency_proposal_gallery(
                image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                saliency_map=np.ones((4, 4), dtype=np.float32),
                prompt_percentiles=(85.0, 90.0, 95.0),
                min_component_area=100,
                max_components=3,
                all_cam_components=True,
                points_per_component=5,
                bbox_padding_ratio=0.02,
                negative_points_per_component=4,
            )

        self.assertEqual([item.component_id for item in result], [0, 1, 2, 3, 4])

    def test_gallery_cap_matches_replacement_protocol(self) -> None:
        with patch(
            "generate_pseudo_masks.morphology.build_class_conditioned_components",
            return_value=(None, None, [component(0), component(1), component(2)]),
        ):
            result = build_external_saliency_proposal_gallery(
                image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                saliency_map=np.ones((4, 4), dtype=np.float32),
                prompt_percentiles=(85.0, 90.0, 95.0),
                min_component_area=100,
                max_components=2,
                all_cam_components=True,
                points_per_component=5,
                bbox_padding_ratio=0.02,
                negative_points_per_component=4,
            )

        self.assertEqual(len(result), 6)


if __name__ == "__main__":
    unittest.main()
