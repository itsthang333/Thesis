from __future__ import annotations

"""Pure, annotation-free primitives for the exact G4 E5 gallery study."""

from collections import defaultdict
from typing import Mapping

import numpy as np

from merge_frozen_candidate_galleries import resize_binary_masks_nearest


ALIGNED_FIELDS = (
    "sam_masks",
    "sam_scores",
    "selection_scores",
    "classifier_causal_scores",
    "component_ids",
    "prompt_modes",
    "proposal_source_ids",
)
EXACT_FIELDS = ("cam_levels", "prompt_ids", "multimask_indices")


def _vector(payload: Mapping[str, np.ndarray], field: str, count: int) -> np.ndarray:
    if field not in payload:
        if field == "classifier_causal_scores":
            return np.zeros(count, dtype=np.float32)
        raise ValueError(f"candidate payload is missing {field}")
    value = np.asarray(payload[field])
    if field != "sam_masks":
        value = value.reshape(-1)
    if len(value) != count:
        raise ValueError(f"candidate field {field} is not aligned")
    return value


def normalized_payload(
    payload: Mapping[str, np.ndarray],
    *,
    namespace: str | None = None,
) -> dict[str, np.ndarray]:
    """Copy a candidate payload and apply the frozen addition namespace."""

    masks = np.asarray(payload["sam_masks"], dtype=np.uint8)
    if masks.ndim != 3:
        raise ValueError("sam_masks must have shape [N,H,W]")
    count = len(masks)
    result = {field: _vector(payload, field, count).copy() for field in ALIGNED_FIELDS}
    result.update(
        {
            "sam_masks": masks.copy(),
            "sam_scores": np.asarray(result["sam_scores"], dtype=np.float32),
            "selection_scores": np.asarray(result["selection_scores"], dtype=np.float32),
            "classifier_causal_scores": np.asarray(
                result["classifier_causal_scores"], dtype=np.float32
            ),
            "component_ids": np.asarray(result["component_ids"], dtype=np.int32),
            "prompt_modes": np.asarray(result["prompt_modes"], dtype="U32"),
            "proposal_source_ids": np.asarray(
                result["proposal_source_ids"], dtype="U96"
            ),
        }
    )
    for field in EXACT_FIELDS:
        if field in payload:
            value = np.asarray(payload[field]).reshape(-1)
            if len(value) != count:
                raise ValueError(f"candidate field {field} is not aligned")
            result[field] = value.copy()
    exact_present = [field in result for field in EXACT_FIELDS]
    if any(exact_present) and not all(exact_present):
        raise ValueError("exact candidate provenance is incomplete")
    if all(exact_present):
        result["cam_levels"] = np.asarray(result["cam_levels"], dtype=np.float32)
        result["prompt_ids"] = np.asarray(result["prompt_ids"], dtype="U192")
        result["multimask_indices"] = np.asarray(
            result["multimask_indices"], dtype=np.int16
        )
    if namespace:
        prefix = f"{namespace}:"
        result["proposal_source_ids"] = np.asarray(
            [prefix + value for value in result["proposal_source_ids"].astype(str)],
            dtype="U96",
        )
        if "prompt_ids" in result:
            result["prompt_ids"] = np.asarray(
                [prefix + value for value in result["prompt_ids"].astype(str)],
                dtype="U192",
            )
    return result


def prompt_key(payload: Mapping[str, np.ndarray], index: int) -> tuple[str, int, str]:
    return (
        str(np.asarray(payload["proposal_source_ids"]).reshape(-1)[index]),
        int(np.asarray(payload["component_ids"]).reshape(-1)[index]),
        str(np.asarray(payload["prompt_modes"]).reshape(-1)[index]),
    )


def attach_exact_multimask_provenance(
    multimask_payload: Mapping[str, np.ndarray],
    single_mask_payload: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bind each old 3-mask SAM group to one newly frozen exact prompt ID."""

    multi = normalized_payload(multimask_payload)
    single = normalized_payload(single_mask_payload)
    if not all(field in single for field in EXACT_FIELDS):
        raise ValueError("single-mask payload lacks exact prompt provenance")
    single_groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    multi_groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index in range(len(single["sam_masks"])):
        single_groups[prompt_key(single, index)].append(index)
    for index in range(len(multi["sam_masks"])):
        multi_groups[prompt_key(multi, index)].append(index)
    if set(single_groups) != set(multi_groups):
        missing = sorted(set(multi_groups) - set(single_groups))[:5]
        extra = sorted(set(single_groups) - set(multi_groups))[:5]
        raise ValueError(f"single/multimask prompt groups differ: missing={missing}, extra={extra}")

    cam_levels = np.empty(len(multi["sam_masks"]), dtype=np.float32)
    prompt_ids = np.empty(len(multi["sam_masks"]), dtype="U192")
    multimask_indices = np.empty(len(multi["sam_masks"]), dtype=np.int16)
    for key in sorted(multi_groups):
        single_indices = single_groups[key]
        multi_indices = multi_groups[key]
        if len(single_indices) != 1 or len(multi_indices) != 3:
            raise ValueError(
                f"exact E5 expects one single mask and three multimasks per prompt: "
                f"{key} has {len(single_indices)}/{len(multi_indices)}"
            )
        single_index = single_indices[0]
        prompt_id = str(single["prompt_ids"][single_index])
        cam_level = float(single["cam_levels"][single_index])
        if not prompt_id or not np.isfinite(cam_level) or cam_level < 0:
            raise ValueError(f"invalid exact prompt provenance for {key}")
        for occurrence, multi_index in enumerate(multi_indices):
            cam_levels[multi_index] = cam_level
            prompt_ids[multi_index] = prompt_id
            multimask_indices[multi_index] = occurrence
    multi.update(
        {
            "cam_levels": cam_levels,
            "prompt_ids": prompt_ids,
            "multimask_indices": multimask_indices,
        }
    )
    return multi


def concatenate_payloads(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    left = normalized_payload(first)
    right = normalized_payload(second)
    if tuple(left["sam_masks"].shape[1:]) != tuple(right["sam_masks"].shape[1:]):
        raise ValueError("candidate banks use different grids")
    exact = all(field in left for field in EXACT_FIELDS) and all(
        field in right for field in EXACT_FIELDS
    )
    if exact != all(field in left for field in EXACT_FIELDS) or exact != all(
        field in right for field in EXACT_FIELDS
    ):
        raise ValueError("candidate banks disagree on exact provenance")
    fields = ALIGNED_FIELDS + (EXACT_FIELDS if exact else ())
    result = {
        field: np.concatenate([np.asarray(left[field]), np.asarray(right[field])])
        for field in fields
    }
    return result


def project_payload_masks_to_grid(
    payload: Mapping[str, np.ndarray],
    target_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Project only candidate masks onto the frozen anchor grid.

    The final rich-gallery baseline merges the 448-pixel addition bank into
    the 320-pixel anchor bank with ``resize_binary_masks_nearest`` before
    duplicate removal.  E5 must replay that exact alignment before it can
    reconstruct the pre-dedup gallery; otherwise the two banks cannot be
    concatenated and, more importantly, duplicate identity would be defined
    in two incompatible coordinate systems.
    """

    normalized = normalized_payload(payload)
    if len(target_shape) != 2 or min(int(value) for value in target_shape) <= 0:
        raise ValueError("target_shape must contain two positive dimensions")
    normalized["sam_masks"] = resize_binary_masks_nearest(
        normalized["sam_masks"],
        (int(target_shape[0]), int(target_shape[1])),
    )
    return normalized


def first_unique_mask_indices(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N,H,W]")
    seen: set[bytes] = set()
    indices: list[int] = []
    for index, mask in enumerate(masks):
        key = np.ascontiguousarray(mask).tobytes()
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def verify_post_dedup_reproduction(
    raw_payload: Mapping[str, np.ndarray],
    post_payload: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return raw first-occurrence indices after proving exact post-dedup replay."""

    raw = normalized_payload(raw_payload)
    post = normalized_payload(post_payload)
    kept = first_unique_mask_indices(raw["sam_masks"])
    if len(kept) != len(post["sam_masks"]):
        raise ValueError("post-dedup candidate count is not the exact raw unique count")
    for field in ALIGNED_FIELDS:
        actual = np.asarray(raw[field])[kept]
        expected = np.asarray(post[field])
        if actual.dtype.kind in "f":
            equal = np.array_equal(actual, expected)
        else:
            equal = np.array_equal(actual, expected)
        if not equal:
            raise ValueError(f"post-dedup payload does not reproduce raw field {field}")
    return kept


__all__ = [
    "ALIGNED_FIELDS",
    "EXACT_FIELDS",
    "attach_exact_multimask_provenance",
    "concatenate_payloads",
    "first_unique_mask_indices",
    "normalized_payload",
    "project_payload_masks_to_grid",
    "prompt_key",
    "verify_post_dedup_reproduction",
]
