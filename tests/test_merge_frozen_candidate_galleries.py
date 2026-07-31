from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT))

from merge_frozen_candidate_galleries import merge_payloads  # noqa: E402


def payload(masks: np.ndarray, source: str, prompt_value: float) -> dict[str, np.ndarray]:
    count = len(masks)
    return {
        "sam_masks": masks.astype(np.uint8),
        "sam_scores": np.linspace(0.5, 0.8, count, dtype=np.float32),
        "selection_scores": np.linspace(0.2, 0.4, count, dtype=np.float32),
        "classifier_causal_scores": np.zeros(count, dtype=np.float32),
        "component_ids": np.arange(count, dtype=np.int32),
        "prompt_modes": np.asarray(["box"] * count),
        "proposal_source_ids": np.asarray([source] * count),
        "prompt_map": np.full(masks.shape[1:], prompt_value, dtype=np.float32),
    }


class MergeFrozenCandidateGalleriesTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
