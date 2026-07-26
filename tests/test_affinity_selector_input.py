from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pseudo.affinity_selector_input import (
    load_affinity_selector_contract,
    load_affinity_selector_map,
    sha256_file,
)


SOURCE_COMMIT = "1" * 40
PROTOCOL_SHA = "2" * 64
SPLIT_SHA = "3" * 64
CHECKPOINT_SHA = "4" * 64


class AffinitySelectorInputTests(unittest.TestCase):
    def _build_package(
        self,
        root: Path,
        *,
        contains_validation_gt_derived_metrics: bool = False,
    ) -> tuple[Path, Path, Path, str, str, str]:
        maps = root / "maps"
        maps.mkdir(parents=True)
        map_path = maps / "sample.npy"
        np.save(
            map_path,
            np.linspace(0.0, 1.0, num=16, dtype=np.float16).reshape(4, 4),
            allow_pickle=False,
        )
        map_hash = sha256_file(map_path)
        manifest = root / "prediction_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "group_id",
                    "tumor",
                    "map_path",
                    "map_sha256",
                    "raw_mean",
                    "raw_p99",
                    "raw_max",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_id": "sample.jpeg",
                    "group_id": "group-1",
                    "tumor": 1,
                    "map_path": "maps/sample.npy",
                    "map_sha256": map_hash,
                    "raw_mean": 0.5,
                    "raw_p99": 0.99,
                    "raw_max": 1.0,
                }
            )
        manifest_hash = sha256_file(manifest)
        assert manifest_hash is not None

        freeze = root / "prediction_freeze.json"
        freeze.write_text(
            json.dumps(
                {
                    "source_commit": SOURCE_COMMIT,
                    "protocol_sha256": PROTOCOL_SHA,
                    "split_sha256": SPLIT_SHA,
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "prediction_manifest_sha256": manifest_hash,
                    "validation_predictions": 1,
                    "validation_gt_read": False,
                    "consumer_trained": False,
                    "test_evaluated": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        freeze_hash = sha256_file(freeze)
        assert freeze_hash is not None

        package = root / "selector_input_manifest.json"
        package.write_text(
            json.dumps(
                {
                    "stage": "prediction-first affinity selector input",
                    "supervision": "images and binary image-level labels only",
                    "contains_validation_gt_derived_metrics": (
                        contains_validation_gt_derived_metrics
                    ),
                    "source_commit": SOURCE_COMMIT,
                    "protocol_sha256": PROTOCOL_SHA,
                    "split": "val",
                    "split_manifest_sha256": SPLIT_SHA,
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "prediction_manifest_sha256": manifest_hash,
                    "prediction_freeze_sha256": freeze_hash,
                    "image_size": 4,
                    "validation_gt_read_at_prediction_freeze": False,
                    "consumer_trained": False,
                    "test_evaluated": False,
                    "cohort": {
                        "validation": 1,
                        "validation_tumor": 1,
                        "validation_normal": 0,
                    },
                    "maps": {"count": 1, "bytes": map_path.stat().st_size},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        package_hash = sha256_file(package)
        assert package_hash is not None
        return manifest, package, freeze, manifest_hash, package_hash, freeze_hash

    def test_contract_and_physical_map_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, package, freeze, manifest_hash, package_hash, freeze_hash = (
                self._build_package(root)
            )

            rows, contract = load_affinity_selector_contract(
                manifest_path=manifest,
                package_metadata_path=package,
                prediction_freeze_path=freeze,
                expected_manifest_sha256=manifest_hash,
                expected_package_metadata_sha256=package_hash,
                expected_prediction_freeze_sha256=freeze_hash,
                expected_source_commit=SOURCE_COMMIT,
                expected_protocol_sha256=PROTOCOL_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                split="val",
                split_manifest_sha256=SPLIT_SHA,
                image_size=4,
            )
            values = load_affinity_selector_map(
                rows["sample.jpeg"],
                root=root,
                expected_image_id="sample.jpeg",
                expected_group_id="group-1",
                expected_image_label=1,
                image_size=4,
            )

            self.assertEqual(values.shape, (4, 4))
            self.assertEqual(contract["contains_validation_gt_derived_metrics"], False)

    def test_package_with_gt_metrics_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, package, freeze, manifest_hash, package_hash, freeze_hash = (
                self._build_package(
                    root,
                    contains_validation_gt_derived_metrics=True,
                )
            )

            with self.assertRaisesRegex(ValueError, "GT-derived"):
                load_affinity_selector_contract(
                    manifest_path=manifest,
                    package_metadata_path=package,
                    prediction_freeze_path=freeze,
                    expected_manifest_sha256=manifest_hash,
                    expected_package_metadata_sha256=package_hash,
                    expected_prediction_freeze_sha256=freeze_hash,
                    expected_source_commit=SOURCE_COMMIT,
                    expected_protocol_sha256=PROTOCOL_SHA,
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    split="val",
                    split_manifest_sha256=SPLIT_SHA,
                    image_size=4,
                )

    def test_tampered_map_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, package, freeze, manifest_hash, package_hash, freeze_hash = (
                self._build_package(root)
            )
            rows, _ = load_affinity_selector_contract(
                manifest_path=manifest,
                package_metadata_path=package,
                prediction_freeze_path=freeze,
                expected_manifest_sha256=manifest_hash,
                expected_package_metadata_sha256=package_hash,
                expected_prediction_freeze_sha256=freeze_hash,
                expected_source_commit=SOURCE_COMMIT,
                expected_protocol_sha256=PROTOCOL_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                split="val",
                split_manifest_sha256=SPLIT_SHA,
                image_size=4,
            )
            with (root / "maps" / "sample.npy").open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_affinity_selector_map(
                    rows["sample.jpeg"],
                    root=root,
                    expected_image_id="sample.jpeg",
                    expected_group_id="group-1",
                    expected_image_label=1,
                    image_size=4,
                )


if __name__ == "__main__":
    unittest.main()
