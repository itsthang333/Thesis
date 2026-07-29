from __future__ import annotations

"""Hash-bound I/O for the GT-blind mask-bag selector-development cache."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PackedCandidateMasks:
    packed: np.ndarray
    candidate_count: int
    height: int
    width: int


MANIFEST_FIELDS = (
    "image_id",
    "group_id",
    "tumor",
    "split",
    "candidate_payload_sha256",
    "candidate_count",
    "descriptor_dim",
    "affinity_dim",
    "packed_masks_included",
    "cache_path",
    "cache_sha256",
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_square_matrix(name: str, values: np.ndarray, count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite candidate-square matrix")
    if not np.allclose(result, result.T, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{name} must be symmetric")
    return result


def save_selector_cache_record(
    path: Path,
    *,
    descriptors: np.ndarray,
    flipped_descriptors: np.ndarray,
    affinity_features: np.ndarray,
    flipped_affinity_features: np.ndarray,
    candidate_indices: np.ndarray,
    family_ids: np.ndarray,
    component_ids: np.ndarray,
    prompt_modes: np.ndarray,
    proposal_source_ids: np.ndarray,
    fallback_flags: np.ndarray,
    shape_features: np.ndarray,
    pairwise_iou: np.ndarray,
    pairwise_containment: np.ndarray,
    pairwise_distance: np.ndarray,
    packed_masks: PackedCandidateMasks | None,
) -> dict[str, int | str]:
    """Save one selector record without images, labels or segmentation targets."""

    original = np.asarray(descriptors, dtype=np.float16)
    flipped = np.asarray(flipped_descriptors, dtype=np.float16)
    affinity = np.asarray(affinity_features, dtype=np.float16)
    flipped_affinity = np.asarray(flipped_affinity_features, dtype=np.float16)
    indices = np.asarray(candidate_indices, dtype=np.int32)
    families = np.asarray(family_ids, dtype=np.int32)
    components = np.asarray(component_ids, dtype=np.int32)
    modes = np.asarray(prompt_modes, dtype="U64")
    sources = np.asarray(proposal_source_ids, dtype="U64")
    fallback = np.asarray(fallback_flags, dtype=np.uint8)
    shapes = np.asarray(shape_features, dtype=np.float32)
    if (
        original.ndim != 2
        or original.shape != flipped.shape
        or original.shape[0] == 0
        or original.shape[1] == 0
        or not np.isfinite(original).all()
        or not np.isfinite(flipped).all()
    ):
        raise ValueError("original/flip descriptors must be aligned finite matrices")
    count, descriptor_dim = original.shape
    if (
        affinity.ndim != 2
        or affinity.shape != flipped_affinity.shape
        or affinity.shape[0] != count
        or affinity.shape[1] <= 0
        or not np.isfinite(affinity).all()
        or not np.isfinite(flipped_affinity).all()
    ):
        raise ValueError("original/flip affinity features must align with descriptors")
    affinity_dim = int(affinity.shape[1])
    if (
        indices.shape != (count,)
        or np.any(indices < 0)
        or np.any(np.diff(indices) <= 0)
        or families.shape != (count,)
        or np.any(families < 0)
        or components.shape != (count,)
        or modes.shape != (count,)
        or sources.shape != (count,)
        or fallback.shape != (count,)
        or np.any(modes == "")
        or np.any(sources == "")
        or np.any(fallback > 1)
        or shapes.shape != (count, 4)
        or not np.isfinite(shapes).all()
    ):
        raise ValueError("candidate index/family/shape cache arrays are invalid")
    iou = _validate_square_matrix("pairwise_iou", pairwise_iou, count)
    containment = _validate_square_matrix(
        "pairwise_containment", pairwise_containment, count
    )
    distance = _validate_square_matrix("pairwise_distance", pairwise_distance, count)
    if (
        np.any(iou < 0)
        or np.any(iou > 1)
        or np.any(containment < 0)
        or np.any(containment > 1)
        or np.any(distance < 0)
    ):
        raise ValueError("pairwise geometry lies outside its valid range")

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "descriptors": original,
        "flipped_descriptors": flipped,
        "affinity_features": affinity,
        "flipped_affinity_features": flipped_affinity,
        "candidate_indices": indices,
        "family_ids": families,
        "component_ids": components,
        "prompt_modes": modes,
        "proposal_source_ids": sources,
        "fallback_flags": fallback,
        "shape_features": shapes,
        "pairwise_iou": iou,
        "pairwise_containment": containment,
        "pairwise_distance": distance,
        "packed_masks_included": np.asarray(
            int(packed_masks is not None), dtype=np.uint8
        ),
    }
    if packed_masks is not None:
        packed = np.asarray(packed_masks.packed, dtype=np.uint8)
        if (
            packed_masks.candidate_count != count
            or packed_masks.height <= 0
            or packed_masks.width <= 0
            or packed.ndim != 2
        ):
            raise ValueError("packed masks do not align with candidate descriptors")
        required_bytes = (
            int(packed_masks.height) * int(packed_masks.width) + 7
        ) // 8
        if packed.shape != (count, required_bytes):
            raise ValueError("packed mask byte count is inconsistent")
        payload.update(
            {
                "packed_masks": packed,
                "mask_height": np.asarray(packed_masks.height, dtype=np.int32),
                "mask_width": np.asarray(packed_masks.width, dtype=np.int32),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "candidate_count": int(count),
        "descriptor_dim": int(descriptor_dim),
        "affinity_dim": affinity_dim,
        "packed_masks_included": int(packed_masks is not None),
        "cache_sha256": sha256_file(path),
    }


def load_selector_cache_record(
    path: Path,
    *,
    expected_sha256: str,
    require_packed_masks: bool,
) -> dict[str, np.ndarray | PackedCandidateMasks]:
    """Load one physically verified selector cache record."""

    if sha256_file(path) != expected_sha256:
        raise ValueError("selector cache record SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["schema_version"]) != 2:
            raise ValueError("selector cache schema mismatch")
        result: dict[str, np.ndarray | PackedCandidateMasks] = {
            key: payload[key]
            for key in (
                "descriptors",
                "flipped_descriptors",
                "affinity_features",
                "flipped_affinity_features",
                "candidate_indices",
                "family_ids",
                "component_ids",
                "prompt_modes",
                "proposal_source_ids",
                "fallback_flags",
                "shape_features",
                "pairwise_iou",
                "pairwise_containment",
                "pairwise_distance",
            )
        }
        masks_included = bool(int(payload["packed_masks_included"]))
        if masks_included:
            result["packed_masks"] = PackedCandidateMasks(
                packed=payload["packed_masks"],
                candidate_count=int(payload["descriptors"].shape[0]),
                height=int(payload["mask_height"]),
                width=int(payload["mask_width"]),
            )
    if require_packed_masks and "packed_masks" not in result:
        raise ValueError("selector cache record omits required candidate masks")
    if not require_packed_masks and "packed_masks" in result:
        raise ValueError("training cache must discard candidate masks")
    # Reuse the writer's complete structural checks without rewriting the file.
    original = np.asarray(result["descriptors"])
    flipped = np.asarray(result["flipped_descriptors"])
    affinity = np.asarray(result["affinity_features"])
    flipped_affinity = np.asarray(result["flipped_affinity_features"])
    indices = np.asarray(result["candidate_indices"])
    families = np.asarray(result["family_ids"])
    components = np.asarray(result["component_ids"])
    modes = np.asarray(result["prompt_modes"])
    sources = np.asarray(result["proposal_source_ids"])
    fallback = np.asarray(result["fallback_flags"])
    shapes = np.asarray(result["shape_features"])
    count = int(original.shape[0])
    if (
        original.dtype != np.float16
        or original.ndim != 2
        or count == 0
        or original.shape != flipped.shape
        or flipped.dtype != np.float16
        or not np.isfinite(original).all()
        or not np.isfinite(flipped).all()
        or affinity.dtype != np.float16
        or affinity.ndim != 2
        or affinity.shape[0] != count
        or affinity.shape[1] <= 0
        or affinity.shape != flipped_affinity.shape
        or flipped_affinity.dtype != np.float16
        or not np.isfinite(affinity).all()
        or not np.isfinite(flipped_affinity).all()
        or indices.dtype != np.int32
        or indices.shape != (count,)
        or np.any(indices < 0)
        or np.any(np.diff(indices) <= 0)
        or families.dtype != np.int32
        or families.shape != (count,)
        or np.any(families < 0)
        or components.dtype != np.int32
        or components.shape != (count,)
        or modes.shape != (count,)
        or modes.dtype.kind != "U"
        or np.any(modes == "")
        or sources.shape != (count,)
        or sources.dtype.kind != "U"
        or np.any(sources == "")
        or fallback.dtype != np.uint8
        or fallback.shape != (count,)
        or np.any(fallback > 1)
        or shapes.dtype != np.float32
        or shapes.shape != (count, 4)
        or not np.isfinite(shapes).all()
    ):
        raise ValueError("selector cache record dtype/shape mismatch")
    iou = _validate_square_matrix(
        "pairwise_iou", np.asarray(result["pairwise_iou"]), count
    )
    containment = _validate_square_matrix(
        "pairwise_containment", np.asarray(result["pairwise_containment"]), count
    )
    distance = _validate_square_matrix(
        "pairwise_distance", np.asarray(result["pairwise_distance"]), count
    )
    if (
        np.any(iou < 0)
        or np.any(iou > 1)
        or np.any(containment < 0)
        or np.any(containment > 1)
        or np.any(distance < 0)
    ):
        raise ValueError("selector cache geometry lies outside its valid range")
    if "packed_masks" in result:
        packed_record = result["packed_masks"]
        assert isinstance(packed_record, PackedCandidateMasks)
        required_bytes = (packed_record.height * packed_record.width + 7) // 8
        if (
            packed_record.candidate_count != count
            or packed_record.height <= 0
            or packed_record.width <= 0
            or np.asarray(packed_record.packed).dtype != np.uint8
            or np.asarray(packed_record.packed).shape != (count, required_bytes)
        ):
            raise ValueError("selector cache packed-mask content mismatch")
    return result


def write_selector_cache_manifest(
    root: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | str]:
    """Freeze the complete train/validation selector cache inventory."""

    if not rows:
        raise ValueError("selector cache manifest cannot be empty")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "selector_cache_manifest.csv"
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["split"]), str(row["image_id"]))
        if key in seen or key[0] not in {"train", "val"} or not key[1]:
            raise ValueError("selector cache identities are invalid or duplicated")
        seen.add(key)
        normalized_row = {field: row[field] for field in MANIFEST_FIELDS}
        cache_path = root / str(normalized_row["cache_path"])
        if not cache_path.is_file() or sha256_file(cache_path) != str(
            normalized_row["cache_sha256"]
        ):
            raise ValueError(f"selector cache payload mismatch: {key}")
        if (key[0] == "val") != bool(int(normalized_row["packed_masks_included"])):
            raise ValueError("only validation selector cache may retain WTA masks")
        normalized.append(normalized_row)
    normalized.sort(key=lambda row: (str(row["split"]), str(row["image_id"])))
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(normalized)
    return {
        "schema_version": 2,
        "records": len(normalized),
        "train_records": sum(row["split"] == "train" for row in normalized),
        "validation_records": sum(row["split"] == "val" for row in normalized),
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_selector_cache_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_images: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, list[dict[str, str]]]:
    """Verify the manifest and every physical cache record before arm fitting."""

    if set(expected_images) != {"train", "val"}:
        raise ValueError("expected selector-cache splits must be train and val")
    manifest_path = root / "selector_cache_manifest.csv"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("selector cache manifest SHA-256 mismatch")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["split"], row["image_id"]): row for row in rows}
    expected_keys = {
        (split, image_id)
        for split, images in expected_images.items()
        for image_id in images
    }
    if len(rows) != len(indexed) or set(indexed) != expected_keys:
        raise ValueError("selector cache manifest cohort mismatch")

    validated: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for image_id, expected in expected_images[split].items():
            row = indexed[(split, image_id)]
            if (
                row["group_id"] != str(expected["group_id"])
                or row["tumor"] != str(expected["tumor"])
                or row["candidate_payload_sha256"]
                != str(expected["candidate_payload_sha256"])
            ):
                raise ValueError(f"selector cache provenance mismatch: {split}/{image_id}")
            payload = load_selector_cache_record(
                root / row["cache_path"],
                expected_sha256=row["cache_sha256"],
                require_packed_masks=split == "val",
            )
            descriptors = np.asarray(payload["descriptors"])
            affinity = np.asarray(payload["affinity_features"])
            if (
                int(row["candidate_count"]) != descriptors.shape[0]
                or int(row["descriptor_dim"]) != descriptors.shape[1]
                or int(row["affinity_dim"]) != affinity.shape[1]
                or affinity.shape[0] != descriptors.shape[0]
                or bool(int(row["packed_masks_included"])) != (split == "val")
            ):
                raise ValueError(
                    f"selector cache manifest/content mismatch: {split}/{image_id}"
                )
            validated[split].append(row)
    return validated


__all__ = [
    "PackedCandidateMasks",
    "load_selector_cache_record",
    "save_selector_cache_record",
    "sha256_file",
    "validate_selector_cache_manifest",
    "write_selector_cache_manifest",
]
