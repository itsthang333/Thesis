from __future__ import annotations

import unittest

import numpy as np

from pseudo.mask_selection import (
    AFFINITY_RANK_PERCENTILES,
    affinity_rank_single_selection,
    select_and_fuse_masks,
)


class AffinityRankSingleSelectorTests(unittest.TestCase):
    def test_selects_compact_candidate_over_broad_anatomy_mask(self) -> None:
        affinity = np.zeros((20, 20), dtype=np.float32)
        affinity[2:8, 3:9] = np.linspace(
            0.5, 1.0, num=36, dtype=np.float32
        ).reshape(6, 6)
        compact = np.zeros_like(affinity, dtype=np.uint8)
        compact[2:8, 3:9] = 1
        broad = np.zeros_like(compact)
        broad[:15, :15] = 1
        unrelated = np.zeros_like(compact)
        unrelated[14:19, 14:19] = 1
        masks = np.stack([compact, broad, unrelated])

        _, selected, details = affinity_rank_single_selection(
            masks,
            affinity,
            sam_scores=np.asarray([0.1, 0.99, 0.5], dtype=np.float32),
            component_ids=np.asarray([0, 1, 2], dtype=np.int32),
        )

        self.assertEqual(selected, [0])
        self.assertIn(
            details["affinity_selected_percentile"],
            AFFINITY_RANK_PERCENTILES,
        )

    def test_support_family_can_select_large_variable_area_candidate(self) -> None:
        affinity = np.zeros((32, 32), dtype=np.float32)
        affinity[6:16, 6:26] = np.linspace(
            0.4, 1.0, num=200, dtype=np.float32
        ).reshape(10, 20)
        large = np.zeros_like(affinity, dtype=np.uint8)
        large[6:16, 6:26] = 1
        core = np.zeros_like(large)
        core[6:8, 6:16] = 1
        masks = np.stack([core, large])

        _, selected, details = affinity_rank_single_selection(
            masks,
            affinity,
            sam_scores=np.asarray([0.99, 0.1], dtype=np.float32),
            component_ids=np.asarray([0, 1], dtype=np.int32),
        )

        self.assertEqual(selected, [1])
        self.assertEqual(details["affinity_selected_percentile"], 80)

    def test_exact_tie_uses_stable_input_order(self) -> None:
        affinity = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
        candidate = np.zeros((10, 10), dtype=np.uint8)
        candidate[7:10, 7:10] = 1
        masks = np.stack([candidate, candidate])

        _, selected, _ = affinity_rank_single_selection(
            masks,
            affinity,
            sam_scores=np.asarray([0.5, 0.5], dtype=np.float32),
            component_ids=np.asarray([0, 0], dtype=np.int32),
        )

        self.assertEqual(selected, [0])

    def test_pipeline_contract_returns_one_raw_candidate(self) -> None:
        affinity = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
        first = np.zeros((10, 10), dtype=np.uint8)
        first[8:10, 8:10] = 1
        second = np.zeros_like(first)
        second[:5, :5] = 1
        masks = np.stack([first, second])

        selected, details = select_and_fuse_masks(
            masks,
            affinity,
            mask_score_threshold=0.0,
            selection_method="affinity_rank_single",
            fusion_topk=1,
            sam_scores=np.asarray([0.5, 0.5], dtype=np.float32),
            component_ids=np.asarray([0, 1], dtype=np.int32),
            best_per_component=False,
            support_clip_kernel=-1,
            return_details=True,
        )

        self.assertTrue(np.array_equal(selected, first))
        self.assertEqual(details["selected_candidates"], 1)
        self.assertEqual(details["affinity_supports"], 6)

    def test_pipeline_contract_rejects_union_or_support_clip(self) -> None:
        affinity = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
        masks = np.ones((1, 10, 10), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "global top-1"):
            select_and_fuse_masks(
                masks,
                affinity,
                mask_score_threshold=0.0,
                selection_method="affinity_rank_single",
                fusion_topk=2,
                best_per_component=False,
                support_clip_kernel=-1,
            )

    def test_constant_map_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-constant"):
            affinity_rank_single_selection(
                np.ones((1, 8, 8), dtype=np.uint8),
                np.ones((8, 8), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
