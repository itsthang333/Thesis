from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT))

from pseudo.mask_selection import score_masks, select_and_fuse_masks


class SourceConsensusSelectorTests(unittest.TestCase):
    def test_requires_teacher_evidence_and_component_boundary(self) -> None:
        masks = np.ones((1, 4, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "source_consensus requires"):
            score_masks(
                masks,
                np.ones((4, 4), dtype=np.float32),
                method="source_consensus",
                component_ids=np.asarray([0], dtype=np.int32),
            )

    def test_teacher_candidate_uses_teacher_source_map(self) -> None:
        cam = np.zeros((6, 6), dtype=np.float32)
        cam[1:3, 1:3] = 1.0
        teacher = np.zeros((6, 6), dtype=np.float32)
        teacher[3:5, 3:5] = 1.0

        cam_candidate = np.zeros((6, 6), dtype=np.uint8)
        cam_candidate[0:4, 0:4] = 1
        teacher_candidate = np.zeros((6, 6), dtype=np.uint8)
        teacher_candidate[3:5, 3:5] = 1
        masks = np.stack([cam_candidate, teacher_candidate])
        components = np.stack(
            [
                (cam > 0).astype(np.uint8),
                (teacher > 0).astype(np.uint8),
            ]
        )

        aligned_scores = score_masks(
            masks,
            cam,
            method="source_consensus",
            sam_scores=np.asarray([0.5, 0.5], dtype=np.float32),
            component_ids=np.asarray([0, 1], dtype=np.int32),
            component_masks=components,
            proposal_teacher_probability=teacher,
            proposal_teacher_component_start=1,
        )
        empty_teacher_scores = score_masks(
            masks,
            cam,
            method="source_consensus",
            sam_scores=np.asarray([0.5, 0.5], dtype=np.float32),
            component_ids=np.asarray([0, 1], dtype=np.int32),
            component_masks=components,
            proposal_teacher_probability=np.zeros_like(teacher),
            proposal_teacher_component_start=1,
        )

        self.assertAlmostEqual(
            float(aligned_scores[0]),
            float(empty_teacher_scores[0]),
        )
        self.assertGreater(
            float(aligned_scores[1]),
            float(empty_teacher_scores[1]),
        )

    def test_source_consensus_retains_cam_support_clipping(self) -> None:
        cam = np.zeros((6, 6), dtype=np.float32)
        cam[0:2, 0:2] = 1.0
        teacher = np.zeros((6, 6), dtype=np.float32)
        teacher[4:6, 4:6] = 1.0
        candidate = (teacher > 0).astype(np.uint8)[None]

        selected = select_and_fuse_masks(
            candidate,
            cam,
            mask_score_threshold=0.0,
            selection_method="source_consensus",
            fusion_topk=1,
            bone_support=(cam > 0).astype(np.uint8),
            sam_scores=np.asarray([1.0], dtype=np.float32),
            component_ids=np.asarray([1], dtype=np.int32),
            component_masks=(teacher > 0).astype(np.uint8)[None],
            proposal_teacher_probability=teacher,
            proposal_teacher_component_start=1,
            support_clip_kernel=1,
        )

        self.assertEqual(int(selected.sum()), 0)


if __name__ == "__main__":
    unittest.main()
