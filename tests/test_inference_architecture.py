from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised in Kaggle/CI with torch
    torch = None


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

if torch is not None:
    from inference import resolve_segmentation_architecture
    from models.unet import architecture_metadata


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight audit environment")
class InferenceArchitectureTests(unittest.TestCase):
    def test_legacy_checkpoint_defaults_to_plain_unet(self) -> None:
        self.assertEqual(resolve_segmentation_architecture({}), "unet")

    def test_current_resnet18_unet_metadata_is_supported(self) -> None:
        checkpoint = {
            "architecture": architecture_metadata("resnet18_unet"),
        }
        self.assertEqual(
            resolve_segmentation_architecture(checkpoint),
            "resnet18_unet",
        )

    def test_noncanonical_architecture_metadata_fails_closed(self) -> None:
        checkpoint = {
            "architecture": {
                **architecture_metadata("resnet18_unet"),
                "encoder": "resnet50",
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported checkpoint architecture",
        ):
            resolve_segmentation_architecture(checkpoint)


if __name__ == "__main__":
    unittest.main()
