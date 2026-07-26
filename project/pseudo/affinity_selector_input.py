from __future__ import annotations

"""Fail-closed loader for frozen prediction-first affinity selector maps."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_affinity_selector_contract(
    *,
    manifest_path: Path | None,
    package_metadata_path: Path | None,
    prediction_freeze_path: Path | None,
    expected_manifest_sha256: str | None,
    expected_package_metadata_sha256: str | None,
    expected_prediction_freeze_sha256: str | None,
    expected_source_commit: str | None,
    expected_protocol_sha256: str | None,
    expected_checkpoint_sha256: str | None,
    split: str,
    split_manifest_sha256: str | None,
    image_size: int,
) -> tuple[dict[str, dict[str, str]], dict[str, object] | None]:
    supplied = (
        manifest_path,
        package_metadata_path,
        prediction_freeze_path,
        expected_manifest_sha256,
        expected_package_metadata_sha256,
        expected_prediction_freeze_sha256,
        expected_source_commit,
        expected_protocol_sha256,
        expected_checkpoint_sha256,
    )
    if not any(value is not None for value in supplied):
        return {}, None
    if any(value is None for value in supplied):
        raise ValueError(
            "Affinity selector input requires manifest, package metadata, "
            "prediction freeze and every expected provenance hash"
        )
    assert manifest_path is not None
    assert package_metadata_path is not None
    assert prediction_freeze_path is not None
    assert expected_manifest_sha256 is not None
    assert expected_package_metadata_sha256 is not None
    assert expected_prediction_freeze_sha256 is not None
    assert expected_source_commit is not None
    assert expected_protocol_sha256 is not None
    assert expected_checkpoint_sha256 is not None

    manifest_path = manifest_path.resolve()
    package_metadata_path = package_metadata_path.resolve()
    prediction_freeze_path = prediction_freeze_path.resolve()
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Affinity selector prediction-manifest SHA-256 mismatch")
    if sha256_file(package_metadata_path) != expected_package_metadata_sha256:
        raise ValueError("Affinity selector package-metadata SHA-256 mismatch")
    if sha256_file(prediction_freeze_path) != expected_prediction_freeze_sha256:
        raise ValueError("Affinity selector prediction-freeze SHA-256 mismatch")

    package = json.loads(package_metadata_path.read_text(encoding="utf-8"))
    freeze = json.loads(prediction_freeze_path.read_text(encoding="utf-8"))
    if package.get("stage") != "prediction-first affinity selector input":
        raise ValueError("Affinity selector package has an unexpected stage")
    if package.get("supervision") != "images and binary image-level labels only":
        raise ValueError("Affinity selector package supervision contract mismatch")
    if package.get("contains_validation_gt_derived_metrics") is not False:
        raise ValueError("Affinity selector package contains validation-GT-derived metrics")
    expected_bindings = {
        "source_commit": expected_source_commit,
        "protocol_sha256": expected_protocol_sha256,
        "split": split,
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "prediction_manifest_sha256": expected_manifest_sha256,
        "prediction_freeze_sha256": expected_prediction_freeze_sha256,
    }
    for key, expected in expected_bindings.items():
        if package.get(key) != expected:
            raise ValueError(f"Affinity selector package binding mismatch: {key}")
    if package.get("image_size") != image_size:
        raise ValueError("Affinity selector map grid differs from --image-size")
    if package.get("validation_gt_read_at_prediction_freeze") is not False:
        raise ValueError("Affinity selector maps were not frozen before validation GT")
    if package.get("consumer_trained") is not False:
        raise ValueError("Affinity selector package unexpectedly trained a consumer")
    if package.get("test_evaluated") is not False:
        raise ValueError("Affinity selector package accessed test")

    freeze_bindings = {
        "source_commit": expected_source_commit,
        "protocol_sha256": expected_protocol_sha256,
        "split_sha256": split_manifest_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "prediction_manifest_sha256": expected_manifest_sha256,
    }
    for key, expected in freeze_bindings.items():
        if freeze.get(key) != expected:
            raise ValueError(f"Affinity selector prediction-freeze mismatch: {key}")
    if freeze.get("validation_gt_read") is not False:
        raise ValueError("Affinity selector prediction freeze reports GT access")
    if freeze.get("consumer_trained") is not False or freeze.get("test_evaluated") is not False:
        raise ValueError("Affinity selector freeze violates consumer/test locks")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "group_id",
        "tumor",
        "map_path",
        "map_sha256",
        "raw_mean",
        "raw_p99",
        "raw_max",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Affinity selector prediction manifest schema mismatch")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = str(row["image_id"])
        if not image_id or image_id in indexed:
            raise ValueError("Affinity selector manifest has duplicate/empty image IDs")
        if row["tumor"] not in {"0", "1"}:
            raise ValueError(f"Affinity selector tumor flag is invalid: {image_id}")
        expected_relative = Path("maps") / f"{Path(image_id).stem}.npy"
        if Path(row["map_path"]) != expected_relative:
            raise ValueError(f"Affinity selector map path mismatch: {image_id}")
        if len(row["map_sha256"]) != 64:
            raise ValueError(f"Affinity selector map hash is invalid: {image_id}")
        indexed[image_id] = row

    cohort = package.get("cohort", {})
    if (
        int(cohort.get("validation", -1)) != len(rows)
        or int(cohort.get("validation_tumor", -1))
        != sum(int(row["tumor"]) for row in rows)
        or int(cohort.get("validation_normal", -1))
        != sum(1 - int(row["tumor"]) for row in rows)
        or int(freeze.get("validation_predictions", -1)) != len(rows)
        or int(package.get("maps", {}).get("count", -1)) != len(rows)
    ):
        raise ValueError("Affinity selector package cohort/count mismatch")

    contract = {
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "package_metadata": str(package_metadata_path),
        "package_metadata_sha256": expected_package_metadata_sha256,
        "prediction_freeze": str(prediction_freeze_path),
        "prediction_freeze_sha256": expected_prediction_freeze_sha256,
        "source_commit": expected_source_commit,
        "protocol_sha256": expected_protocol_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "image_size": image_size,
        "validation_gt_read_at_prediction_freeze": False,
        "contains_validation_gt_derived_metrics": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    return indexed, contract


def load_affinity_selector_map(
    record: dict[str, str],
    *,
    root: Path,
    expected_image_id: str,
    expected_group_id: str,
    expected_image_label: int,
    image_size: int,
) -> np.ndarray:
    if record.get("image_id") != expected_image_id:
        raise ValueError("Affinity selector image identity mismatch")
    if record.get("group_id") != expected_group_id:
        raise ValueError(f"Affinity selector group identity mismatch: {expected_image_id}")
    if int(record.get("tumor", "-1")) != expected_image_label:
        raise ValueError(f"Affinity selector image-label mismatch: {expected_image_id}")
    expected_relative = Path("maps") / f"{Path(expected_image_id).stem}.npy"
    relative = Path(record.get("map_path", ""))
    if relative != expected_relative:
        raise ValueError(f"Affinity selector map path mismatch: {expected_image_id}")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Affinity selector map escapes artifact root") from error
    if sha256_file(path) != record.get("map_sha256"):
        raise ValueError(f"Affinity selector map SHA-256 mismatch: {expected_image_id}")
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float16 or values.shape != (image_size, image_size):
        raise ValueError(f"Affinity selector map dtype/shape mismatch: {expected_image_id}")
    values = values.astype(np.float32)
    if (
        not np.isfinite(values).all()
        or float(values.min()) < 0.0
        or float(values.max()) > 1.0
    ):
        raise ValueError(f"Affinity selector map value range mismatch: {expected_image_id}")
    if expected_image_label == 1 and float(values.max() - values.min()) <= 1e-6:
        raise ValueError(f"Tumor affinity selector map is constant: {expected_image_id}")
    return values
