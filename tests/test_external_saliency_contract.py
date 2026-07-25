from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.generate_pseudo_masks import (
    load_external_saliency_contract,
    load_external_saliency_map,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExternalSaliencyContractTests(unittest.TestCase):
    def build_artifact(self, root: Path):
        maps = root / "maps"
        maps.mkdir(parents=True)
        values = np.linspace(0.0, 1.0, 16, dtype=np.float16).reshape(4, 4)
        map_path = maps / "image.npy"
        np.save(map_path, values, allow_pickle=False)
        manifest_path = root / "saliency_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "tumor_image_label",
                    "map_path",
                    "map_sha256",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_id": "image.jpeg",
                    "tumor_image_label": "1",
                    "map_path": "maps/image.npy",
                    "map_sha256": sha256(map_path),
                }
            )
        manifest_hash = sha256(manifest_path)
        metadata_path = root / "run_metadata.json"
        metadata = {
            "stage": "prediction-first BiomedCLIP saliency generation",
            "supervision": "images and binary image-level labels only",
            "source_commit": "a" * 40,
            "split": "val",
            "split_manifest_sha256": "b" * 64,
            "manifest_sha256": manifest_hash,
            "model": {"weight_sha256": "c" * 64},
            "source_files": {"generator.py": "d" * 64},
            "prompts": {"sha256": "e" * 64},
            "view_contract": {"output_size": 4},
            "validation_gt_read": False,
            "test_evaluated": False,
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return manifest_path, metadata_path, manifest_hash

    def test_exact_contract_and_map_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, metadata, manifest_hash = self.build_artifact(root)
            rows, contract = load_external_saliency_contract(
                manifest_path=manifest,
                metadata_path=metadata,
                expected_manifest_sha256=manifest_hash,
                expected_metadata_sha256=sha256(metadata),
                expected_source_commit="a" * 40,
                expected_model_weight_sha256="c" * 64,
                split="val",
                split_manifest_sha256="b" * 64,
                image_size=4,
            )
            self.assertIsNotNone(contract)
            values = load_external_saliency_map(
                rows["image.jpeg"],
                root=root,
                expected_image_id="image.jpeg",
                expected_image_label=1,
                image_size=4,
            )
            self.assertEqual(values.shape, (4, 4))

    def test_validation_gt_access_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, metadata, manifest_hash = self.build_artifact(root)
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["validation_gt_read"] = True
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation GT"):
                load_external_saliency_contract(
                    manifest_path=manifest,
                    metadata_path=metadata,
                    expected_manifest_sha256=manifest_hash,
                    expected_metadata_sha256=sha256(metadata),
                    expected_source_commit="a" * 40,
                    expected_model_weight_sha256="c" * 64,
                    split="val",
                    split_manifest_sha256="b" * 64,
                    image_size=4,
                )

    def test_map_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, metadata, manifest_hash = self.build_artifact(root)
            rows, _ = load_external_saliency_contract(
                manifest_path=manifest,
                metadata_path=metadata,
                expected_manifest_sha256=manifest_hash,
                expected_metadata_sha256=sha256(metadata),
                expected_source_commit="a" * 40,
                expected_model_weight_sha256="c" * 64,
                split="val",
                split_manifest_sha256="b" * 64,
                image_size=4,
            )
            np.save(root / "maps" / "image.npy", np.zeros((4, 4), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_external_saliency_map(
                    rows["image.jpeg"],
                    root=root,
                    expected_image_id="image.jpeg",
                    expected_image_label=1,
                    image_size=4,
                )


if __name__ == "__main__":
    unittest.main()
