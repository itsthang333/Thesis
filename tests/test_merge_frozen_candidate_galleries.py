from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT))

from merge_frozen_candidate_galleries import (  # noqa: E402
    merge_payloads,
    resize_binary_masks_nearest,
)


def payload(
    masks: np.ndarray,
    source: str,
    prompt_value: float,
    *,
    exact_provenance: bool = False,
) -> dict[str, np.ndarray]:
    count = len(masks)
    result = {
        "sam_masks": masks.astype(np.uint8),
        "sam_scores": np.linspace(0.5, 0.8, count, dtype=np.float32),
        "selection_scores": np.linspace(0.2, 0.4, count, dtype=np.float32),
        "classifier_causal_scores": np.zeros(count, dtype=np.float32),
        "component_ids": np.arange(count, dtype=np.int32),
        "prompt_modes": np.asarray(["box"] * count),
        "proposal_source_ids": np.asarray([source] * count),
        "prompt_map": np.full(masks.shape[1:], prompt_value, dtype=np.float32),
    }
    if exact_provenance:
        result.update(
            {
                "cam_levels": np.full(count, 90.0, dtype=np.float32),
                "prompt_ids": np.asarray(
                    [f"{source}|p90|c{index}|box" for index in range(count)]
                ),
                "multimask_indices": np.zeros(count, dtype=np.int16),
            }
        )
    return result


class MergeFrozenCandidateGalleriesTests(unittest.TestCase):
    def test_exact_provenance_survives_dedup_and_namespaces_addition(self) -> None:
        a = np.zeros((1, 4, 4), dtype=np.uint8)
        a[0, :2, :2] = 1
        b = np.zeros((1, 4, 4), dtype=np.uint8)
        b[0, 2:, 2:] = 1
        merged, _ = merge_payloads(
            payload(a, "layercam", 0.25, exact_provenance=True),
            payload(b, "layercam", 0.75, exact_provenance=True),
            addition_namespace="classifier448",
        )
        self.assertEqual(merged["cam_levels"].tolist(), [90.0, 90.0])
        self.assertEqual(merged["multimask_indices"].tolist(), [0, 0])
        self.assertEqual(
            merged["prompt_ids"].astype(str).tolist(),
            ["layercam|p90|c0|box", "classifier448:layercam|p90|c0|box"],
        )

    def test_unconditional_union_deduplicates_and_preserves_anchor_prompt(self) -> None:
        a = np.zeros((2, 4, 4), dtype=np.uint8)
        a[0, :2, :2] = 1
        a[1, 2:, 2:] = 1
        b = np.zeros((2, 4, 4), dtype=np.uint8)
        b[0] = a[1]
        b[1, 1:3, 1:3] = 1
        merged, stats = merge_payloads(
            payload(a, "layercam", 0.25),
            payload(b, "layercam", 0.75),
            addition_namespace="classifier448",
        )
        self.assertEqual(merged["sam_masks"].shape, (3, 4, 4))
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(
            merged["proposal_source_ids"].tolist(),
            ["layercam", "layercam", "classifier448:layercam"],
        )
        np.testing.assert_array_equal(
            merged["prompt_map"], np.full((4, 4), 0.25, dtype=np.float32)
        )

    def test_legacy_addition_provenance_is_explicitly_backfilled(self) -> None:
        a = np.zeros((1, 4, 4), dtype=np.uint8)
        a[0, :2, :2] = 1
        b = np.zeros((1, 4, 4), dtype=np.uint8)
        b[0, 2:, 2:] = 1
        merged, stats = merge_payloads(
            payload(a, "dsll", 0.25, exact_provenance=True),
            payload(b, "layercam", 0.75, exact_provenance=False),
            addition_namespace="classifier448",
            allow_missing_addition_provenance=True,
        )
        self.assertEqual(stats["addition_provenance_backfilled"], 1)
        self.assertEqual(merged["multimask_indices"].tolist(), [0, -1])
        self.assertTrue(np.isnan(merged["cam_levels"][1]))
        self.assertIn(
            "classifier448:legacy_provenance_unavailable",
            str(merged["prompt_ids"][1]),
        )

    def test_legacy_addition_provenance_still_fails_closed_by_default(self) -> None:
        masks = np.ones((1, 4, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "availability differs"):
            merge_payloads(
                payload(masks, "dsll", 0.25, exact_provenance=True),
                payload(masks, "layercam", 0.75, exact_provenance=False),
                addition_namespace="classifier448",
            )

    def test_rejects_invalid_namespace(self) -> None:
        masks = np.ones((1, 4, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "source prefix"):
            merge_payloads(
                payload(masks, "a", 0.1),
                payload(masks, "b", 0.2),
                addition_namespace="bad:namespace",
            )

    def test_rejects_misaligned_candidate_vectors(self) -> None:
        masks = np.ones((1, 4, 4), dtype=np.uint8)
        bad = payload(masks, "b", 0.2)
        bad["sam_scores"] = np.asarray([], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "does not align"):
            merge_payloads(
                payload(masks, "a", 0.1),
                bad,
                addition_namespace="classifier448",
            )

    def test_resizes_448_style_addition_to_anchor_before_dedup(self) -> None:
        anchor = np.zeros((1, 4, 4), dtype=np.uint8)
        anchor[0, :2, :2] = 1
        addition = np.zeros((2, 8, 8), dtype=np.uint8)
        addition[0, :4, :4] = 1
        addition[1, 4:, 4:] = 1
        merged, stats = merge_payloads(
            payload(anchor, "layercam", 0.25),
            payload(addition, "layercam", 0.75),
            addition_namespace="classifier448",
        )
        self.assertEqual(merged["sam_masks"].shape, (2, 4, 4))
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["addition_resized"], 1)
        np.testing.assert_array_equal(merged["sam_masks"][0], anchor[0])
        np.testing.assert_array_equal(
            merged["sam_masks"][1],
            resize_binary_masks_nearest(addition[1:], (4, 4))[0],
        )

    def test_preserves_candidate_that_nearest_projection_would_empty(self) -> None:
        anchor = np.zeros((1, 4, 4), dtype=np.uint8)
        anchor[0, :2, :2] = 1
        addition = np.zeros((2, 8, 8), dtype=np.uint8)
        addition[0, 1, 1] = 1  # No 4x4 output sample lands on this source cell.
        addition[1, 6:, 6:] = 1
        merged, stats = merge_payloads(
            payload(anchor, "layercam", 0.25),
            payload(addition, "layercam", 0.75),
            addition_namespace="classifier448",
        )
        self.assertEqual(stats["addition_input"], 2)
        self.assertEqual(stats["addition_empty_after_nearest_recovered"], 1)
        self.assertEqual(stats["addition_kept"], 2)
        self.assertTrue(merged["sam_masks"].reshape(3, -1).any(axis=1).all())
        self.assertEqual(
            merged["proposal_source_ids"].tolist(),
            [
                "layercam",
                "classifier448:layercam",
                "classifier448:layercam",
            ],
        )

    def test_removes_empty_candidate_already_present_in_frozen_supply(self) -> None:
        valid = np.ones((1, 4, 4), dtype=np.uint8)
        empty = np.zeros((1, 8, 8), dtype=np.uint8)
        merged, stats = merge_payloads(
            payload(valid, "layercam", 0.25),
            payload(empty, "layercam", 0.75),
            addition_namespace="classifier448",
        )
        self.assertEqual(stats["addition_original_empty_removed"], 1)
        self.assertEqual(stats["addition_kept"], 0)
        self.assertEqual(merged["sam_masks"].shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
