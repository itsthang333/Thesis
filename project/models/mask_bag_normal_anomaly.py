from __future__ import annotations

"""Normal-only direct anomaly ranking for immutable candidate bags.

The caller supplies candidate descriptors from image-label-normal training
bags.  This module has no dataset, annotation, subgroup, evaluator or consumer
interface and never learns from positive bags.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from models.mask_bag_normal_prototypes import (
    fit_weighted_spherical_prototypes,
    hierarchical_image_family_weights,
)


@dataclass(frozen=True)
class DirectNormalAnomalyConfig:
    prototype_count: int = 32
    seed: int = 42
    maximum_iterations: int = 100
    convergence_tolerance: float = 1.0e-6


def _normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("descriptors must have shape [candidates, features]")
    if not np.isfinite(array).all():
        raise ValueError("descriptors must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("descriptors must have nonzero norm")
    return array / norms


def normal_bank_training_arrays(
    records: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build equal image/family/candidate/view weighted nominal instances."""

    if not records:
        raise ValueError("normal bank needs at least one training image")
    descriptors: list[np.ndarray] = []
    image_ids: list[np.ndarray] = []
    family_ids: list[np.ndarray] = []
    dimension: int | None = None
    total_candidates = 0
    seen: set[str] = set()
    for record in records:
        image_id = str(record["image_id"])
        if not image_id or image_id in seen:
            raise ValueError("normal-bank image IDs must be unique and nonempty")
        seen.add(image_id)
        if int(record["image_label"]) != 0:
            raise ValueError("normal bank may contain only image-label-normal records")
        original = np.asarray(record["descriptors"], dtype=np.float32)
        flipped = np.asarray(record["flipped_descriptors"], dtype=np.float32)
        families = np.asarray(record["family_ids"])
        if (
            original.ndim != 2
            or flipped.shape != original.shape
            or families.shape != (original.shape[0],)
            or original.shape[0] == 0
            or not np.isfinite(original).all()
            or not np.isfinite(flipped).all()
        ):
            raise ValueError(f"invalid normal-bank record: {image_id}")
        if dimension is None:
            dimension = int(original.shape[1])
        elif original.shape[1] != dimension:
            raise ValueError("normal-bank descriptor dimensions differ")
        count = int(original.shape[0])
        total_candidates += count
        descriptors.extend((original, flipped))
        image_ids.extend(
            (
                np.full(count, image_id, dtype="U128"),
                np.full(count, image_id, dtype="U128"),
            )
        )
        family_ids.extend((families.copy(), families.copy()))

    stacked = np.concatenate(descriptors, axis=0)
    images = np.concatenate(image_ids, axis=0)
    families = np.concatenate(family_ids, axis=0)
    weights = hierarchical_image_family_weights(images, families)
    audit = {
        "normal_images": len(records),
        "normal_candidates": total_candidates,
        "normal_candidate_views": int(stacked.shape[0]),
        "descriptor_dimension": int(stacked.shape[1]),
        "weight_sum": float(weights.astype(np.float64).sum()),
        "all_training_image_labels_normal": True,
        "view_multiplicity": 2,
    }
    return stacked, weights, audit


def fit_direct_normal_anomaly_bank(
    records: Sequence[Mapping[str, object]],
    *,
    config: DirectNormalAnomalyConfig = DirectNormalAnomalyConfig(),
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one fixed spherical nominal bank with no positive-bag optimizer."""

    if config.prototype_count != 32 or config.seed != 42:
        raise ValueError("N1 requires the frozen K=32 and seed=42 controls")
    values, weights, audit = normal_bank_training_arrays(records)
    prototypes, assignments = fit_weighted_spherical_prototypes(
        values,
        weights,
        prototype_count=config.prototype_count,
        seed=config.seed,
        maximum_iterations=config.maximum_iterations,
        convergence_tolerance=config.convergence_tolerance,
    )
    cluster_counts = np.bincount(assignments, minlength=config.prototype_count)
    if prototypes.shape != (config.prototype_count, values.shape[1]) or np.any(cluster_counts == 0):
        raise RuntimeError("normal anomaly bank contains an empty prototype")
    return prototypes, {
        **audit,
        "prototype_count": config.prototype_count,
        "seed": config.seed,
        "cluster_counts": cluster_counts.astype(int).tolist(),
        "positive_bags_used": 0,
        "learned_residual": False,
    }


def score_direct_normal_anomaly(
    descriptors: np.ndarray,
    flipped_descriptors: np.ndarray,
    prototypes: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Average aligned-view nearest-normal distances in candidate order."""

    original = _normalized(descriptors)
    flipped = _normalized(flipped_descriptors)
    centers = _normalized(prototypes)
    if flipped.shape != original.shape or centers.shape[1] != original.shape[1]:
        raise ValueError("N1 descriptor/prototype shapes differ")
    original_distance = (1.0 - np.max(original @ centers.T, axis=1)).astype(np.float32)
    flipped_distance = (1.0 - np.max(flipped @ centers.T, axis=1)).astype(np.float32)
    candidate_scores = (0.5 * (original_distance + flipped_distance)).astype(np.float32)
    if not np.array_equal(
        candidate_scores,
        (0.5 * (original_distance + flipped_distance)).astype(np.float32),
    ):
        raise RuntimeError("N1 aligned-view averaging identity failed")
    return {
        "original_normal_distance": original_distance,
        "flipped_normal_distance": flipped_distance,
        "candidate_scores": candidate_scores,
        "selected_candidate_position": int(np.argmax(candidate_scores)),
        "view_selected_agreement": int(
            np.argmax(original_distance) == np.argmax(flipped_distance)
        ),
    }


__all__ = [
    "DirectNormalAnomalyConfig",
    "fit_direct_normal_anomaly_bank",
    "normal_bank_training_arrays",
    "score_direct_normal_anomaly",
]
