from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def validate_paired_wsl_contract(lock_path: Path, args: Namespace) -> dict[str, Any]:
    """Fail closed when a WSL consumer diverges from its GT reference arm."""
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "hash_locked":
        raise ValueError("Paired GT reference must be hash_locked")
    isolation = lock.get("weak_supervision_isolation", {})
    if isolation.get("test_evaluated") is not False:
        raise ValueError("Paired GT reference does not keep test locked")
    if isolation.get("train_gt_may_influence_wsl_generation_or_selection") is not False:
        raise ValueError("Paired GT reference permits GT leakage into WSL")
    if getattr(args, "train_pred_mask_root", None) is None:
        raise ValueError("Paired WSL training requires --train-pred-mask-root")
    if getattr(args, "val_pred_mask_root", None) is not None:
        raise ValueError("Paired WSL validation must use GT only after prediction")
    if getattr(args, "train_split", None) != "train" or getattr(args, "val_split", None) != "val":
        raise ValueError("Paired WSL training requires frozen train/val partitions")
    if getattr(args, "split_manifest", None) is None:
        raise ValueError("Paired WSL training requires --split-manifest")

    split_path = Path(args.split_manifest).resolve()
    expected_split = str(lock["data"]["split_manifest_sha256"])
    if sha256_file(split_path) != expected_split:
        raise ValueError("Paired WSL split-manifest SHA-256 mismatch")

    contract = lock["consumer_training_contract"]
    expected = {
        "image_size": int(contract["input_size"]),
        "model_architecture": "resnet18_unet",
        "no_pretrained_encoder": False,
        "batch_size": int(contract["batch_size"]),
        "lr": float(contract["learning_rate"]),
        "weight_decay": float(contract["weight_decay"]),
        "epochs": int(contract["maximum_epoch"]),
        "seed": int(contract["seed"]),
        "early_stop_patience": int(contract["early_stop_patience"]),
        "checkpoint_dice_tolerance": 0.0001,
        "pos_weight_mode": "manual",
        "pos_weight_value": 10.0,
        "use_clahe": False,
    }
    for attribute, expected_value in expected.items():
        actual = getattr(args, attribute, None)
        if actual != expected_value:
            raise ValueError(
                f"Paired WSL contract fixes {attribute}={expected_value!r}; "
                f"received {actual!r}"
            )
    return {
        "reference_id": lock["reference_id"],
        "reference_lock_canonical_lf_sha256": canonical_text_sha256(lock_path),
        "reference_checkpoint_sha256": lock["artifact_hashes"]["checkpoint_sha256"],
        "split_manifest_sha256": expected_split,
        "allowed_training_difference": "train mask source only",
        "test_evaluated": False,
    }
