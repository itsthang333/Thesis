from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.generate_biomedclip_saliency import load_rows, save_map, sha256_file


class GenerateBiomedClipSaliencyTests(unittest.TestCase):
    def test_load_rows_selects_only_requested_eligible_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            fields = ["image_id", "group_id", "split", "eligible", "tumor", "image_sha256"]
            rows = [
                {
                    "image_id": "a.jpeg",
                    "group_id": "a",
                    "split": "val",
                    "eligible": "1",
                    "tumor": "1",
                    "image_sha256": "a" * 64,
                },
                {
                    "image_id": "b.jpeg",
                    "group_id": "b",
                    "split": "train",
                    "eligible": "1",
                    "tumor": "0",
                    "image_sha256": "b" * 64,
                },
                {
                    "image_id": "c.jpeg",
                    "group_id": "c",
                    "split": "val",
                    "eligible": "0",
                    "tumor": "1",
                    "image_sha256": "c" * 64,
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            selected = load_rows(
                path,
                expected_sha256=sha256_file(path),
                split="val",
            )
            self.assertEqual([row["image_id"] for row in selected], ["a.jpeg"])

    def test_split_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text(
                "image_id,group_id,split,eligible,tumor,image_sha256\n"
                "a.jpeg,a,val,1,1," + "a" * 64 + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_rows(path, expected_sha256="0" * 64, split="val")

    def test_save_map_is_float16_and_pickle_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npy"
            values = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
            save_map(path, values)
            loaded = np.load(path, allow_pickle=False)
            self.assertEqual(loaded.dtype, np.float16)
            self.assertEqual(loaded.shape, (4, 4))

    def test_save_map_rejects_out_of_range_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npy"
            with self.assertRaisesRegex(ValueError, "normalized"):
                save_map(path, np.asarray([[0.0, 1.1]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
