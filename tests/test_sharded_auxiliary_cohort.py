from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generate_pseudo_masks import load_image_list, validate_auxiliary_cohort


class ShardedAuxiliaryCohortTest(unittest.TestCase):
    def test_utf8_bom_is_not_part_of_first_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.txt"
            path.write_text("\ufeffIMG000001.jpeg\nIMG000002.jpeg\n", encoding="utf-8")
            self.assertEqual(load_image_list(path), {"IMG000001.jpeg", "IMG000002.jpeg"})

    def test_exact_shard_can_use_locked_full_manifest(self) -> None:
        validate_auxiliary_cohort(
            ["A.jpeg", "C.jpeg"],
            {"A.jpeg": object(), "B.jpeg": object(), "C.jpeg": object()},
            allow_manifest_superset=True,
            label="External saliency",
        )

    def test_missing_shard_map_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*C.jpeg"):
            validate_auxiliary_cohort(
                ["A.jpeg", "C.jpeg"],
                {"A.jpeg": object(), "B.jpeg": object()},
                allow_manifest_superset=True,
                label="External saliency",
            )

    def test_full_run_still_requires_exact_cohort(self) -> None:
        with self.assertRaisesRegex(ValueError, "extra_count=1"):
            validate_auxiliary_cohort(
                ["A.jpeg"],
                {"A.jpeg": object(), "B.jpeg": object()},
                allow_manifest_superset=False,
                label="External saliency",
            )


if __name__ == "__main__":
    unittest.main()
