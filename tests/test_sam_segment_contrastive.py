from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local audit runtime")
class SamSegmentContrastiveLossTests(unittest.TestCase):
    def test_denseblock2_tap_is_stride8_and_preserves_logits(self) -> None:
        from models.classifier import DenseNet121AnatomyClassifier

        torch.manual_seed(7)
        model = DenseNet121AnatomyClassifier(
            num_classes=1,
            pretrained=False,
            dropout=0.0,
        ).eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            plain_logits = model(images)
            tapped_logits, features = model(
                images,
                return_features=True,
                feature_stage="denseblock2",
            )
        self.assertEqual(tuple(features.shape), (1, 512, 8, 8))
        self.assertTrue(torch.equal(plain_logits, tapped_logits))
        self.assertFalse(
            any("sam_segment" in key or "projection" in key for key in model.state_dict())
        )

    def test_denseblock2_tap_rejects_unknown_stage(self) -> None:
        from models.classifier import DenseNet121AnatomyClassifier

        model = DenseNet121AnatomyClassifier(
            num_classes=1,
            pretrained=False,
            dropout=0.0,
        ).eval()
        with self.assertRaisesRegex(ValueError, "feature_stage"):
            model(
                torch.randn(1, 3, 64, 64),
                return_features=True,
                feature_stage="unknown",
            )

    def test_coherent_region_features_have_lower_loss(self) -> None:
        from models.sam_segment_contrastive import sam_segment_contrastive_loss

        regions = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.long)
        coherent = torch.tensor(
            [[[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]],
            requires_grad=True,
        )
        mixed = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]],
            requires_grad=True,
        )
        coherent_loss = sam_segment_contrastive_loss(coherent, regions)
        mixed_loss = sam_segment_contrastive_loss(mixed, regions)
        self.assertLess(coherent_loss.item(), mixed_loss.item())
        coherent_loss.backward()
        self.assertTrue(torch.isfinite(coherent.grad).all())

    def test_ignore_zero_does_not_create_a_prototype(self) -> None:
        from models.sam_segment_contrastive import sam_segment_contrastive_loss

        features = torch.tensor(
            [[[[100.0, 1.0, 1.0]], [[100.0, 0.0, 0.0]]]], requires_grad=True
        )
        regions = torch.tensor([[[0, 1, 1]]], dtype=torch.long)
        loss = sam_segment_contrastive_loss(features, regions)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertEqual(float(features.grad[0, :, 0, 0].abs().sum()), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local audit runtime")
class SamSegmentMapStoreTests(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_store_verifies_population_hashes_and_loads_uint16(self) -> None:
        from models.sam_segment_contrastive import SamSegmentMapStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = root / "region_maps"
            maps.mkdir()
            map_path = maps / "IMG1.png"
            Image.fromarray(
                np.asarray([[0, 1], [2, 2]], dtype=np.uint16), mode="I;16"
            ).save(map_path)
            manifest = root / "region_map_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_id",
                        "source_image_sha256",
                        "region_map_path",
                        "region_map_sha256",
                        "regions",
                        "map_width",
                        "map_height",
                        "map_dtype",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": "IMG1.jpeg",
                        "source_image_sha256": "source-sha",
                        "region_map_path": "region_maps/IMG1.png",
                        "region_map_sha256": self._sha(map_path),
                        "regions": 2,
                        "map_width": 2,
                        "map_height": 2,
                        "map_dtype": "uint16",
                    }
                )
            store = SamSegmentMapStore(
                root,
                [{"image_id": "IMG1.jpeg", "image_sha256": "source-sha"}],
                expected_manifest_sha256=self._sha(manifest),
            )
            batch = store.load_batch(["IMG1.jpeg"], device=torch.device("cpu"))
            self.assertEqual(tuple(batch.shape), (1, 2, 2))
            self.assertEqual(batch.dtype, torch.int64)

    def test_store_rejects_manifest_hash_mismatch(self) -> None:
        from models.sam_segment_contrastive import SamSegmentMapStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "region_map_manifest.csv").write_text(
                "image_id,source_image_sha256,region_map_path,region_map_sha256,"
                "regions,map_width,map_height,map_dtype\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                SamSegmentMapStore(
                    root,
                    [],
                    expected_manifest_sha256="0" * 64,
                )
