from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
SOURCE = PROJECT / "models" / "mask_bag_selector_cache_io.py"
SPEC = importlib.util.spec_from_file_location("mask_bag_selector_cache_io", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PackedCandidateMasks = MODULE.PackedCandidateMasks


def _arrays() -> dict[str, object]:
    descriptors = np.arange(24, dtype=np.float32).reshape(3, 8)
    masks = np.zeros((3, 4, 5), dtype=np.uint8)
    masks[0, :2, :2] = 1
    masks[1, 1:3, 1:4] = 1
    masks[2, 2:, 3:] = 1
    flat = masks.reshape(3, -1).astype(np.float32)
    area = flat.sum(axis=1)
    intersection = flat @ flat.T
    union = area[:, None] + area[None, :] - intersection
    iou = intersection / np.maximum(union, 1)
    containment = intersection / np.maximum(
        np.minimum(area[:, None], area[None, :]), 1
    )
    distance = np.asarray(
        [[0.0, 0.2, 0.8], [0.2, 0.0, 0.6], [0.8, 0.6, 0.0]],
        dtype=np.float32,
    )
    packed = PackedCandidateMasks(
        packed=np.packbits(masks.reshape(3, -1), axis=1),
        candidate_count=3,
        height=4,
        width=5,
    )
    return {
        "descriptors": descriptors,
        "flipped_descriptors": descriptors + 1,
        "candidate_indices": np.asarray([0, 2, 5]),
        "family_ids": np.asarray([0, 0, 1]),
        "component_ids": np.asarray([4, 4, 9]),
        "prompt_modes": np.asarray(["point", "point", "box"]),
        "proposal_source_ids": np.asarray(["cam", "cam", "teacher"]),
        "fallback_flags": np.asarray([0, 0, 0]),
        "shape_features": np.ones((3, 4), dtype=np.float32),
        "pairwise_iou": iou,
        "pairwise_containment": containment,
        "pairwise_distance": distance,
        "packed_masks": packed,
    }


class SelectorCacheIOTests(unittest.TestCase):
    def _write_manifest_fixture(
        self, root: Path
    ) -> tuple[dict[str, object], dict[str, dict[str, dict[str, object]]]]:
        rows = []
        expected: dict[str, dict[str, dict[str, object]]] = {"train": {}, "val": {}}
        for split, masks_included in (("train", False), ("val", True)):
            values = _arrays()
            if not masks_included:
                values["packed_masks"] = None
            relative = Path(split) / "image.npz"
            saved = MODULE.save_selector_cache_record(root / relative, **values)
            image_id = f"{split}.jpeg"
            candidate_hash = ("a" if split == "train" else "b") * 64
            rows.append(
                {
                    "image_id": image_id,
                    "group_id": f"{split}_group",
                    "tumor": 1,
                    "split": split,
                    "candidate_payload_sha256": candidate_hash,
                    **saved,
                    "cache_path": str(relative),
                }
            )
            expected[split][image_id] = {
                "group_id": f"{split}_group",
                "tumor": 1,
                "candidate_payload_sha256": candidate_hash,
            }
        return MODULE.write_selector_cache_manifest(root, rows), expected

    def test_validation_round_trip_preserves_descriptors_geometry_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            saved = MODULE.save_selector_cache_record(path, **_arrays())
            loaded = MODULE.load_selector_cache_record(
                path,
                expected_sha256=saved["cache_sha256"],
                require_packed_masks=True,
            )

            self.assertTrue(
                np.array_equal(
                    loaded["descriptors"],
                    np.asarray(_arrays()["descriptors"], dtype=np.float16),
                )
            )
            self.assertTrue(
                np.array_equal(loaded["candidate_indices"], np.asarray([0, 2, 5]))
            )
            self.assertEqual(loaded["prompt_modes"].tolist(), ["point", "point", "box"])
            self.assertEqual(
                loaded["proposal_source_ids"].tolist(), ["cam", "cam", "teacher"]
            )
            self.assertIsInstance(loaded["packed_masks"], PackedCandidateMasks)

    def test_training_record_discards_masks_and_loader_enforces_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            values = _arrays()
            values["packed_masks"] = None
            saved = MODULE.save_selector_cache_record(path, **values)
            loaded = MODULE.load_selector_cache_record(
                path,
                expected_sha256=saved["cache_sha256"],
                require_packed_masks=False,
            )
            self.assertNotIn("packed_masks", loaded)
            with self.assertRaisesRegex(ValueError, "omits required"):
                MODULE.load_selector_cache_record(
                    path,
                    expected_sha256=saved["cache_sha256"],
                    require_packed_masks=True,
                )

    def test_record_hash_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            saved = MODULE.save_selector_cache_record(path, **_arrays())
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.load_selector_cache_record(
                    path,
                    expected_sha256=saved["cache_sha256"],
                    require_packed_masks=True,
                )

    def test_hash_valid_but_malformed_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.npz"
            values = _arrays()
            MODULE.save_selector_cache_record(path, **values)
            with np.load(path, allow_pickle=False) as payload:
                rewritten = {key: payload[key] for key in payload.files}
            rewritten["candidate_indices"] = np.asarray([0, 5, 2], dtype=np.int32)
            np.savez_compressed(path, **rewritten)
            with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
                MODULE.load_selector_cache_record(
                    path,
                    expected_sha256=MODULE.sha256_file(path),
                    require_packed_masks=True,
                )

    def test_manifest_allows_masks_only_for_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _expected = self._write_manifest_fixture(root)

            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["train_records"], 1)
            self.assertEqual(summary["validation_records"], 1)

    def test_manifest_validator_opens_every_physical_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, expected = self._write_manifest_fixture(root)
            validated = MODULE.validate_selector_cache_manifest(
                root,
                expected_manifest_sha256=summary["manifest_sha256"],
                expected_images=expected,
            )
            self.assertEqual(len(validated["train"]), 1)
            self.assertEqual(len(validated["val"]), 1)

            expected["val"]["val.jpeg"]["candidate_payload_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                MODULE.validate_selector_cache_manifest(
                    root,
                    expected_manifest_sha256=summary["manifest_sha256"],
                    expected_images=expected,
                )

    def test_nonascending_candidate_indices_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = _arrays()
            values["candidate_indices"] = np.asarray([0, 5, 2])
            with self.assertRaisesRegex(ValueError, "index/family/shape"):
                MODULE.save_selector_cache_record(
                    Path(directory) / "record.npz", **values
                )

    def test_source_is_gt_and_subgroup_free(self) -> None:
        source = SOURCE.read_text(encoding="utf-8").lower()
        for forbidden in (
            "datasets.factory",
            "segmentation_dataset",
            "mask_tensor",
            "ground_truth",
            "size_group",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIsNone(re.search(r"\bdice\b", source))
