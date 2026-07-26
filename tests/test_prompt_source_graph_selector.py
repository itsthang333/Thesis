from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT))

from pseudo.mask_selection import (  # noqa: E402
    prompt_source_graph_selection,
    select_and_fuse_masks,
)


def rectangle(
    shape: tuple[int, int],
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


class PromptSourceGraphSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        shape = (16, 16)
        masks: list[np.ndarray] = []
        components: list[int] = []
        modes: list[str] = []
        sources: list[str] = []

        definitions = [
            # Two independent sources agree on the same upper-left lesion.
            (0, "layercam", [(1, 1, 6, 6), (1, 1, 6, 6), (1, 1, 6, 6)]),
            (1, "external_saliency", [(1, 1, 6, 6), (1, 1, 6, 6), (1, 1, 6, 6)]),
            # Stable single-source fallback.
            (2, "layercam", [(9, 1, 13, 5), (9, 1, 13, 5), (9, 1, 13, 5)]),
            # Unstable single-source component.
            (3, "external_saliency", [(9, 9, 13, 13), (7, 7, 10, 10), (12, 12, 15, 15)]),
        ]
        prompt_modes = ("box_point", "point", "box")
        for component, source, boxes in definitions:
            for mode, box in zip(prompt_modes, boxes):
                masks.append(rectangle(shape, *box))
                components.append(component)
                modes.append(mode)
                sources.append(source)

        self.masks = np.stack(masks)
        self.components = np.asarray(components, dtype=np.int32)
        self.modes = np.asarray(modes, dtype="U32")
        self.sources = np.asarray(sources, dtype="U32")
        self.sam_scores = np.full(len(masks), 0.9, dtype=np.float32)

    def test_cross_source_cluster_ranks_first_with_single_source_fallback(self) -> None:
        _, selected, details = prompt_source_graph_selection(
            self.masks,
            self.sam_scores,
            self.components,
            self.modes,
            self.sources,
            component_topk=2,
        )

        selected_components = set(int(self.components[index]) for index in selected)
        self.assertEqual(len(selected), 2)
        self.assertTrue(selected_components & {0, 1})
        self.assertIn(2, selected_components)
        self.assertNotIn(3, selected_components)
        self.assertEqual(details["proposal_clusters"], 3)
        self.assertEqual(details["cross_source_clusters"], 1)

    def test_selection_unions_distinct_clusters_and_keeps_support_clip(self) -> None:
        support = np.zeros((16, 16), dtype=np.uint8)
        support[0:14, 0:7] = 1
        selected, details = select_and_fuse_masks(
            self.masks,
            np.ones((16, 16), dtype=np.float32),
            selection_method="prompt_source_graph",
            sam_scores=self.sam_scores,
            component_ids=self.components,
            prompt_modes=self.modes,
            proposal_source_ids=self.sources,
            component_topk=2,
            bone_support=support,
            support_clip_kernel=1,
            return_details=True,
        )

        self.assertGreater(int(selected[1:6, 1:6].sum()), 0)
        self.assertGreater(int(selected[9:13, 1:5].sum()), 0)
        self.assertEqual(int(selected[:, 8:].sum()), 0)
        self.assertEqual(details["selected_candidates"], 2)
        self.assertEqual(details["cross_source_clusters"], 1)

    def test_fails_closed_on_unaligned_source_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned masks"):
            prompt_source_graph_selection(
                self.masks,
                self.sam_scores,
                self.components,
                self.modes,
                self.sources[:-1],
                component_topk=2,
            )


if __name__ == "__main__":
    unittest.main()
