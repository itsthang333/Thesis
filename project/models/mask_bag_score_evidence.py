from __future__ import annotations

"""Hash-bound all-candidate score evidence for post-freeze selector audits."""

import csv
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


FIELDS = (
    "image_id",
    "group_id",
    "tumor",
    "candidate_payload_sha256",
    "candidate_count",
    "selected_candidate_index",
    "selected_candidate_logit",
    "score_path",
    "score_sha256",
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_candidate_score_evidence(
    path: Path,
    *,
    candidate_indices: Sequence[int] | np.ndarray,
    candidate_logits: Sequence[float] | np.ndarray,
) -> dict[str, int | float | str]:
    """Save every valid candidate score in immutable gallery order."""

    indices = np.asarray(candidate_indices, dtype=np.int64)
    logits = np.asarray(candidate_logits, dtype=np.float32)
    if indices.ndim != 1 or logits.ndim != 1 or indices.shape != logits.shape:
        raise ValueError("candidate indices and logits must be matching 1D arrays")
    if len(indices) == 0:
        raise ValueError("candidate score evidence cannot be empty")
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("candidate indices must be unique and nonnegative")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("candidate indices must preserve ascending gallery order")
    if not np.isfinite(logits).all():
        raise ValueError("candidate logits must be finite")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        candidate_indices=indices,
        candidate_logits=logits,
    )
    selected_position = int(np.argmax(logits))
    return {
        "candidate_count": int(len(indices)),
        "selected_candidate_index": int(indices[selected_position]),
        "selected_candidate_logit": float(logits[selected_position]),
        "score_sha256": sha256_file(path),
    }


def write_candidate_score_manifest(
    root: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | str]:
    """Write the score manifest after every per-image payload already exists."""

    if not rows:
        raise ValueError("candidate score manifest cannot be empty")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "candidate_score_manifest.csv"
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        image_id = str(row["image_id"])
        if not image_id or image_id in seen:
            raise ValueError("image IDs must be unique and nonempty")
        seen.add(image_id)
        normalized_row = {field: row[field] for field in FIELDS}
        score_path = root / str(normalized_row["score_path"])
        if not score_path.is_file():
            raise ValueError(f"candidate score payload is missing: {image_id}")
        if sha256_file(score_path) != str(normalized_row["score_sha256"]):
            raise ValueError(f"candidate score payload hash mismatch: {image_id}")
        normalized.append(normalized_row)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(normalized)
    return {
        "schema_version": 1,
        "images": len(normalized),
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_candidate_score_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_images: Mapping[str, Mapping[str, object]],
) -> list[dict[str, str]]:
    """Verify all score evidence before any post-freeze quality is computed."""

    manifest_path = root / "candidate_score_manifest.csv"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("candidate score manifest SHA-256 mismatch")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != set(expected_images):
        raise ValueError("candidate score manifest image cohort mismatch")

    validated: list[dict[str, str]] = []
    for image_id in expected_images:
        row = indexed[image_id]
        expected = expected_images[image_id]
        if (
            row["group_id"] != str(expected["group_id"])
            or row["tumor"] != str(expected["tumor"])
            or row["candidate_payload_sha256"]
            != str(expected["candidate_payload_sha256"])
            or int(row["candidate_count"]) != int(expected["candidate_count"])
        ):
            raise ValueError(f"candidate score provenance mismatch: {image_id}")
        score_path = root / row["score_path"]
        if not score_path.is_file() or sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"candidate score payload hash mismatch: {image_id}")
        with np.load(score_path, allow_pickle=False) as payload:
            if int(payload["schema_version"]) != 1:
                raise ValueError(f"candidate score schema mismatch: {image_id}")
            indices = payload["candidate_indices"]
            logits = payload["candidate_logits"]
        if (
            indices.dtype != np.int64
            or logits.dtype != np.float32
            or indices.ndim != 1
            or logits.ndim != 1
            or indices.shape != logits.shape
            or len(indices) != int(row["candidate_count"])
            or np.any(indices < 0)
            or np.any(np.diff(indices) <= 0)
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"candidate score content mismatch: {image_id}")
        selected_position = int(np.argmax(logits))
        if (
            int(indices[selected_position]) != int(row["selected_candidate_index"])
            or float(logits[selected_position])
            != float(row["selected_candidate_logit"])
        ):
            raise ValueError(f"candidate score winner mismatch: {image_id}")
        validated.append(row)
    return validated


__all__ = [
    "save_candidate_score_evidence",
    "sha256_file",
    "validate_candidate_score_manifest",
    "write_candidate_score_manifest",
]
