from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "project"
        / "tools"
        / "paired_reference_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "paired_reference_contract_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = load_module()
ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "artifacts"
    / "reference"
    / "gt_resnet18_unet_448_v1"
    / "reference_lock.json"
)
SPLIT = (
    ROOT
    / "artifacts"
    / "best_pipeline"
    / "fs_resnet18_pw10_full_448_e20"
    / "data"
    / "split_manifest.csv"
)


def valid_args() -> Namespace:
    return Namespace(
        train_pred_mask_root=Path("pseudo/masks"),
        val_pred_mask_root=None,
        train_split="train",
        val_split="val",
        split_manifest=SPLIT,
        image_size=448,
        model_architecture="resnet18_unet",
        no_pretrained_encoder=False,
        batch_size=8,
        lr=0.0001,
        weight_decay=0.0001,
        epochs=35,
        seed=42,
        early_stop_patience=10,
        checkpoint_dice_tolerance=0.0001,
        pos_weight_mode="manual",
        pos_weight_value=10.0,
        use_clahe=False,
    )


class PairedReferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.runtime_split = (
            Path(cls._temporary_directory.name) / "frozen_split_manifest.csv"
        )
        canonical = SPLIT.read_bytes().replace(b"\r\n", b"\n")
        cls.runtime_split.write_bytes(canonical.replace(b"\n", b"\r\n"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def valid_args(self) -> Namespace:
        args = valid_args()
        args.split_manifest = self.runtime_split
        return args

    def test_exact_wsl_consumer_contract_passes(self) -> None:
        result = CONTRACT.validate_paired_wsl_contract(LOCK, self.valid_args())
        self.assertEqual(result["reference_id"], "gt_resnet18_unet_448_v1")
        self.assertEqual(result["allowed_training_difference"], "train mask source only")
        self.assertFalse(result["test_evaluated"])

    def test_consumer_change_fails_closed(self) -> None:
        args = self.valid_args()
        args.lr = 0.0002
        with self.assertRaisesRegex(ValueError, "fixes lr"):
            CONTRACT.validate_paired_wsl_contract(LOCK, args)

    def test_missing_pseudo_masks_or_gt_validation_override_fails(self) -> None:
        args = self.valid_args()
        args.train_pred_mask_root = None
        with self.assertRaisesRegex(ValueError, "train-pred-mask-root"):
            CONTRACT.validate_paired_wsl_contract(LOCK, args)
        args = self.valid_args()
        args.val_pred_mask_root = Path("pseudo/val")
        with self.assertRaisesRegex(ValueError, "validation"):
            CONTRACT.validate_paired_wsl_contract(LOCK, args)


if __name__ == "__main__":
    unittest.main()
