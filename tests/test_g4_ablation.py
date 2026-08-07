from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from g4_ablation import (
    candidate_filter,
    deterministic_random_candidate,
    fusion_score,
    source_correct_upstream_components,
    source_correct_upstream_components_by_source,
    upstream_components,
    upstream_score,
    within_group_percentile_rank,
)
from evaluate_g4_offline_ablations import _candidate_dice, _oracle_from_dice


class G4AblationTests(unittest.TestCase):
    def test_upstream_components_reproduce_density_mass_and_local_rank(self) -> None:
        masks = np.asarray(
            [
                [[1, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            dtype=np.uint8,
        )
        prompt = np.asarray([[1.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        components = upstream_components(masks, prompt, [0.1, 0.9, 0.3], [5, 5, 7])
        np.testing.assert_allclose(components.cam_density, [1.0, 0.5, 0.0])
        np.testing.assert_allclose(components.cam_mass_coverage, [0.5, 0.75, 0.25])
        np.testing.assert_allclose(components.sam_component_rank, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            upstream_score(components, "U5"),
            0.60 * components.cam_density
            + 0.25 * components.cam_mass_coverage
            + 0.15 * components.sam_component_rank,
        )

    def test_average_ties_are_preserved_within_each_component(self) -> None:
        actual = within_group_percentile_rank([1.0, 1.0, 3.0, 7.0], [0, 0, 0, 1])
        np.testing.assert_allclose(actual, [0.25, 0.25, 1.0, 0.0])

    def test_source_correct_components_equal_shared_map_case(self) -> None:
        masks = np.asarray(
            [
                [[1, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            dtype=np.uint8,
        )
        prompt = np.asarray([[1.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        sam = np.asarray([0.1, 0.9, 0.3], dtype=np.float32)
        groups = np.asarray(["layercam:5", "layercam:5", "external:7"])
        expected = upstream_components(masks, prompt, sam, groups)
        actual = source_correct_upstream_components(
            masks, np.repeat(prompt[None], len(masks), axis=0), sam, groups
        )
        for field in (
            "sam_score",
            "cam_density",
            "cam_mass_coverage",
            "sam_component_rank",
            "sam_global_rank",
        ):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field))

    def test_source_correct_components_use_own_map_and_source_group(self) -> None:
        masks = np.ones((3, 2, 2), dtype=np.uint8)
        prompt_maps = np.asarray(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        actual = source_correct_upstream_components(
            masks,
            prompt_maps,
            np.asarray([0.1, 0.9, 0.3]),
            np.asarray(["layercam:0", "layercam:0", "external:0"]),
        )
        np.testing.assert_allclose(actual.cam_density, [0.5, 0.25, 0.0])
        np.testing.assert_allclose(actual.cam_mass_coverage, [1.0, 1.0, 0.0])
        # A singleton group has the implementation's neutral rank zero. If the
        # integer suffix had incorrectly merged it with LayerCAM, it would have
        # received an intermediate rank instead.
        np.testing.assert_allclose(actual.sam_component_rank, [0.0, 1.0, 0.0])

    def test_by_source_matches_per_candidate_map_implementation(self) -> None:
        masks = np.asarray(
            [
                [[1, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            dtype=np.uint8,
        )
        source_ids = np.asarray(["a", "a", "b"])
        source_maps = {
            "a": np.asarray([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32),
            "b": np.asarray([[0.0, 0.0], [0.8, 0.8]], dtype=np.float32),
        }
        prompt_maps = np.stack([source_maps[source] for source in source_ids])
        sam = np.asarray([0.1, 0.9, 0.3])
        component = np.asarray([5, 5, 5])
        expected = source_correct_upstream_components(
            masks,
            prompt_maps,
            sam,
            np.asarray([f"{source}:{item}" for source, item in zip(source_ids, component)]),
        )
        actual = source_correct_upstream_components_by_source(
            masks, source_ids, source_maps, sam, component
        )
        for field in (
            "sam_score",
            "cam_density",
            "cam_mass_coverage",
            "sam_component_rank",
            "sam_global_rank",
        ):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field))

    def test_fusion_arms_are_finite_and_r7_is_frozen_rule(self) -> None:
        g1 = np.asarray([1.0, 4.0, 2.0, 3.0])
        upstream = np.asarray([4.0, 1.0, 2.0, 3.0])
        for arm in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
            self.assertTrue(np.isfinite(fusion_score(g1, upstream, arm)).all())
        np.testing.assert_allclose(
            fusion_score(g1, upstream, "R7"),
            0.5 * within_group_percentile_rank(g1, np.zeros(4))
            + 0.5 * within_group_percentile_rank(upstream, np.zeros(4)),
        )

    def test_balanced_cap_keeps_top_scores_per_source(self) -> None:
        sources = np.asarray(["a", "a", "a", "b", "b", "b"])
        scores = np.asarray([0.1, 0.9, 0.8, 0.3, 0.2, 0.7])
        actual = candidate_filter(
            sources,
            scores,
            allowed_sources=("a", "b"),
            per_source_cap=2,
        )
        np.testing.assert_array_equal(actual, [1, 2, 3, 5])

    def test_random_selector_is_stable_but_seed_sensitive(self) -> None:
        candidates = np.arange(100)
        first = deterministic_random_candidate("IMG000001.jpeg", candidates, 17)
        self.assertEqual(first, deterministic_random_candidate("IMG000001.jpeg", candidates, 17))
        self.assertNotEqual(first, deterministic_random_candidate("IMG000001.jpeg", candidates, 18))

    def test_candidate_dice_and_oracle_are_deterministic(self) -> None:
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[1, :3, :3] = 1
        masks[2, 3, 3] = 1
        target = masks[0].astype(bool)
        candidate_dice = _candidate_dice(masks, target)
        score, index = _oracle_from_dice(candidate_dice, np.asarray([0, 1, 2]))
        self.assertEqual(index, 0)
        self.assertAlmostEqual(score, 1.0)
        restricted_score, restricted_index = _oracle_from_dice(
            candidate_dice, np.asarray([1, 2])
        )
        self.assertEqual(restricted_index, 1)
        self.assertAlmostEqual(restricted_score, 8.0 / 13.0)


if __name__ == "__main__":
    unittest.main()
