from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.tools.audit_biomedclip_saliency import validate_manifest_rows


def image_row(name: str, tumor: int) -> dict[str, str]:
    return {
        "image_id": name,
        "group_id": f"group-{name}",
        "split": "val",
        "tumor": str(tumor),
        "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "width": "10",
        "height": "8",
    }


def manifest_row(root: Path, expected: dict[str, str], tumor: int) -> dict[str, str]:
    values = (
        np.linspace(0.0, 1.0, 16, dtype=np.float16).reshape(4, 4)
        if tumor
        else np.zeros((4, 4), dtype=np.float16)
    )
    relative = Path("maps") / f"{Path(expected['image_id']).stem}.npy"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    scores = [float(index) / 10 for index in range(9)] if tumor else []
    selected = (
        [
            {"box_xyxy": [0, 0, 2, 2], "contrast_score": score}
            for score in sorted(scores, reverse=True)[:3]
        ]
        if tumor
        else []
    )
    return {
        "image_id": expected["image_id"],
        "group_id": expected["group_id"],
        "split": "val",
        "tumor_image_label": str(tumor),
        "source_image_sha256": expected["image_sha256"],
        "source_width": expected["width"],
        "source_height": expected["height"],
        "map_path": str(relative).replace("\\", "/"),
        "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "map_height": "4",
        "map_width": "4",
        "map_min": str(float(values.min())),
        "map_max": str(float(values.max())),
        "map_mean": str(float(values.astype(np.float32).mean())),
        "map_dynamic_range": str(float(values.max() - values.min())),
        "full_contrast_score": "0.3" if tumor else "",
        "selected_tiles": json.dumps(selected, separators=(",", ":")),
        "all_tile_scores": json.dumps(scores, separators=(",", ":")),
        "generation": (
            "frozen_biomedclip_full_plus_top3_tiles"
            if tumor
            else "known_normal_image_label_empty"
        ),
    }


class AuditBiomedClipSaliencyTests(unittest.TestCase):
    def test_valid_small_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [image_row("normal.jpeg", 0), image_row("tumor.jpeg", 1)]
            manifest = [
                manifest_row(root, expected[0], 0),
                manifest_row(root, expected[1], 1),
            ]
            result = validate_manifest_rows(
                root, manifest, expected, output_size=4
            )
            self.assertEqual(result["images"], 2)
            self.assertEqual(result["tumor"], 1)

    def test_identity_order_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [image_row("normal.jpeg", 0), image_row("tumor.jpeg", 1)]
            manifest = [
                manifest_row(root, expected[1], 1),
                manifest_row(root, expected[0], 0),
            ]
            with self.assertRaisesRegex(ValueError, "identities/order"):
                validate_manifest_rows(root, manifest, expected, output_size=4)

    def test_nonempty_normal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [image_row("normal.jpeg", 0), image_row("tumor.jpeg", 1)]
            normal = manifest_row(root, expected[0], 0)
            normal_path = root / normal["map_path"]
            np.save(normal_path, np.ones((4, 4), dtype=np.float16), allow_pickle=False)
            normal["map_sha256"] = hashlib.sha256(normal_path.read_bytes()).hexdigest()
            normal["map_min"] = normal["map_max"] = normal["map_mean"] = "1.0"
            manifest = [normal, manifest_row(root, expected[1], 1)]
            with self.assertRaisesRegex(ValueError, "maximum mismatch|empty-map"):
                validate_manifest_rows(root, manifest, expected, output_size=4)

    def test_unmanifested_map_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [image_row("normal.jpeg", 0), image_row("tumor.jpeg", 1)]
            manifest = [
                manifest_row(root, expected[0], 0),
                manifest_row(root, expected[1], 1),
            ]
            np.save(root / "maps" / "extra.npy", np.zeros((4, 4), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                validate_manifest_rows(root, manifest, expected, output_size=4)


if __name__ == "__main__":
    unittest.main()
