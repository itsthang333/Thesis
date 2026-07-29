from __future__ import annotations

"""GT-free cache helpers for immutable mask-bag selector development."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedCandidateMasks:
    packed: np.ndarray
    candidate_count: int
    height: int
    width: int


def encode_candidate_families(
    component_ids: np.ndarray,
    prompt_modes: np.ndarray,
    proposal_source_ids: np.ndarray,
    *,
    kept_indices: np.ndarray | None = None,
    fallback_flags: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Encode stable family IDs from immutable proposal provenance."""

    components = np.asarray(component_ids, dtype=np.int64).reshape(-1)
    modes = np.asarray(prompt_modes, dtype="U64").reshape(-1)
    sources = np.asarray(proposal_source_ids, dtype="U64").reshape(-1)
    if not (len(components) == len(modes) == len(sources)):
        raise ValueError("candidate provenance arrays must align")
    if len(components) == 0:
        raise ValueError("candidate provenance must be nonempty")
    if np.any(modes == "") or np.any(sources == ""):
        raise ValueError("candidate provenance strings must be nonempty")

    original_indices = np.arange(len(components), dtype=np.int64)
    if kept_indices is not None:
        kept = np.asarray(kept_indices, dtype=np.int64).reshape(-1)
        if kept.size == 0 or np.any(kept < 0) or np.any(kept >= len(components)):
            raise ValueError("kept_indices lie outside candidate provenance")
        if len(np.unique(kept)) != len(kept):
            raise ValueError("kept_indices must be unique")
        components = components[kept]
        modes = modes[kept]
        sources = sources[kept]
        original_indices = kept

    if fallback_flags is None:
        fallback = np.zeros(len(component_ids), dtype=bool)
    else:
        fallback = np.asarray(fallback_flags).reshape(-1).astype(bool)
        if fallback.shape != (len(component_ids),):
            raise ValueError("fallback_flags must align with original candidates")
    fallback = fallback[original_indices]

    keys: list[tuple[str, str, int]] = []
    for component, mode, source, is_fallback in zip(
        components,
        modes,
        sources,
        fallback,
        strict=True,
    ):
        if is_fallback:
            keys.append(("fallback", "fallback", -1))
        else:
            keys.append((str(source), str(mode), int(component)))

    ordered_keys = sorted(set(keys))
    key_to_id = {key: index for index, key in enumerate(ordered_keys)}
    family_ids = np.asarray([key_to_id[key] for key in keys], dtype=np.int32)
    table: list[dict[str, object]] = []
    for family_id, (source, mode, component) in enumerate(ordered_keys):
        members = np.flatnonzero(family_ids == family_id)
        table.append(
            {
                "family_id": family_id,
                "proposal_source": source,
                "prompt_mode": mode,
                "component_id": component,
                "candidate_count": int(len(members)),
                "original_candidate_indices": original_indices[members].tolist(),
            }
        )
    return family_ids, table


def candidate_shape_features(candidate_masks: np.ndarray) -> np.ndarray:
    """Return area, box area, fill ratio and log aspect without position."""

    masks = np.asarray(candidate_masks)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    binary = masks > 0.5
    height, width = binary.shape[1:]
    image_area = float(height * width)
    rows: list[list[float]] = []
    for mask in binary:
        y, x = np.nonzero(mask)
        if not len(y):
            raise ValueError("candidate masks must be nonempty")
        box_height = int(y.max() - y.min() + 1)
        box_width = int(x.max() - x.min() + 1)
        area = int(mask.sum())
        box_area = box_height * box_width
        rows.append(
            [
                area / image_area,
                box_area / image_area,
                area / float(box_area),
                float(np.log((box_width / width) / (box_height / height))),
            ]
        )
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("candidate shape features are non-finite")
    return result


def pairwise_overlap_geometry(
    candidate_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU, containment and normalized centroid-distance matrices."""

    masks = np.asarray(candidate_masks)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    binary = masks > 0.5
    flat = binary.reshape(binary.shape[0], -1).astype(np.float32)
    areas = flat.sum(axis=1)
    if np.any(areas <= 0):
        raise ValueError("candidate masks must be nonempty")
    intersections = flat @ flat.T
    unions = areas[:, None] + areas[None, :] - intersections
    iou = intersections / np.maximum(unions, 1.0)
    containment = intersections / np.maximum(
        np.minimum(areas[:, None], areas[None, :]),
        1.0,
    )

    coordinates = np.indices(binary.shape[1:], dtype=np.float32)
    centroid_y = np.einsum("nhw,hw->n", binary, coordinates[0]) / areas
    centroid_x = np.einsum("nhw,hw->n", binary, coordinates[1]) / areas
    delta_y = (centroid_y[:, None] - centroid_y[None, :]) / max(
        1.0,
        float(binary.shape[1] - 1),
    )
    delta_x = (centroid_x[:, None] - centroid_x[None, :]) / max(
        1.0,
        float(binary.shape[2] - 1),
    )
    distance = np.sqrt(delta_y**2 + delta_x**2)
    return (
        iou.astype(np.float32),
        containment.astype(np.float32),
        distance.astype(np.float32),
    )


def pack_candidate_masks(candidate_masks: np.ndarray) -> PackedCandidateMasks:
    """Bit-pack immutable binary masks without changing candidate order."""

    masks = np.asarray(candidate_masks)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    binary = masks > 0.5
    packed = np.packbits(binary.reshape(binary.shape[0], -1), axis=1)
    return PackedCandidateMasks(
        packed=packed,
        candidate_count=int(binary.shape[0]),
        height=int(binary.shape[1]),
        width=int(binary.shape[2]),
    )


def unpack_candidate_masks(payload: PackedCandidateMasks) -> np.ndarray:
    """Restore an exact uint8 candidate-mask tensor from a packed payload."""

    if (
        payload.candidate_count <= 0
        or payload.height <= 0
        or payload.width <= 0
        or np.asarray(payload.packed).ndim != 2
    ):
        raise ValueError("packed candidate-mask metadata are invalid")
    required_bits = payload.height * payload.width
    expected_bytes = (required_bits + 7) // 8
    packed = np.asarray(payload.packed, dtype=np.uint8)
    if packed.shape != (payload.candidate_count, expected_bytes):
        raise ValueError("packed candidate-mask byte count is inconsistent")
    unpacked = np.unpackbits(packed, axis=1, count=required_bits)
    return unpacked.reshape(
        payload.candidate_count,
        payload.height,
        payload.width,
    ).astype(np.uint8)


__all__ = [
    "PackedCandidateMasks",
    "candidate_shape_features",
    "encode_candidate_families",
    "pack_candidate_masks",
    "pairwise_overlap_geometry",
    "unpack_candidate_masks",
]
