from __future__ import annotations

"""Dataset-agnostic normal-prototype features for immutable proposal bags.

The caller is responsible for supplying descriptors from image-label-normal
training bags.  This module has no dataset, annotation or subgroup interface.
"""

import numpy as np


def _normalize_rows(values: np.ndarray, *, epsilon: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("values must have shape [instances,features]")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("values must not contain zero-norm rows")
    return values / norms


def hierarchical_image_family_weights(
    image_ids: np.ndarray,
    family_ids: np.ndarray,
) -> np.ndarray:
    """Give each image, then each family in that image, equal total weight."""

    images = np.asarray(image_ids)
    families = np.asarray(family_ids)
    if images.ndim != 1 or families.shape != images.shape or images.size == 0:
        raise ValueError("image_ids and family_ids must be aligned vectors")
    if np.issubdtype(images.dtype, np.number) and not np.isfinite(images).all():
        raise ValueError("image_ids must be finite")
    if np.issubdtype(families.dtype, np.number) and not np.isfinite(
        families
    ).all():
        raise ValueError("family_ids must be finite")

    unique_images = np.unique(images)
    weights = np.zeros(images.size, dtype=np.float64)
    image_weight = 1.0 / float(unique_images.size)
    for image in unique_images:
        in_image = images == image
        image_families = np.unique(families[in_image])
        family_weight = image_weight / float(image_families.size)
        for family in image_families:
            members = in_image & (families == family)
            weights[members] = family_weight / float(np.count_nonzero(members))
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-12):
        raise RuntimeError("hierarchical weights do not sum to one")
    return weights.astype(np.float32)


def fit_weighted_spherical_prototypes(
    descriptors: np.ndarray,
    weights: np.ndarray,
    *,
    prototype_count: int,
    seed: int,
    maximum_iterations: int = 100,
    convergence_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic-seeded weighted spherical k-means bank."""

    values = _normalize_rows(descriptors)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if sample_weights.shape != (values.shape[0],):
        raise ValueError("weights must align with descriptors")
    if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
        raise ValueError("weights must be finite and positive")
    sample_weights /= sample_weights.sum()
    if not 1 <= prototype_count <= values.shape[0]:
        raise ValueError("prototype_count lies outside the instance count")
    if maximum_iterations < 1 or convergence_tolerance <= 0:
        raise ValueError("iteration controls must be positive")

    rng = np.random.default_rng(seed)
    selected: list[int] = [
        int(rng.choice(values.shape[0], p=sample_weights))
    ]
    nearest_distance = np.clip(
        1.0 - values @ values[selected[0]],
        0.0,
        2.0,
    )
    while len(selected) < prototype_count:
        probabilities = sample_weights * nearest_distance**2
        probabilities[np.asarray(selected, dtype=np.int64)] = 0.0
        total = float(probabilities.sum())
        if total <= 0:
            remaining = np.setdiff1d(
                np.arange(values.shape[0]),
                np.asarray(selected),
                assume_unique=False,
            )
            selected.append(int(remaining[0]))
        else:
            selected.append(
                int(rng.choice(values.shape[0], p=probabilities / total))
            )
        nearest_distance = np.minimum(
            nearest_distance,
            np.clip(1.0 - values @ values[selected[-1]], 0.0, 2.0),
        )

    prototypes = values[np.asarray(selected)].copy()
    assignments = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(maximum_iterations):
        assignments = np.argmax(values @ prototypes.T, axis=1)
        updated = prototypes.copy()
        for cluster in range(prototype_count):
            members = assignments == cluster
            if not np.any(members):
                residual = 1.0 - np.max(values @ updated.T, axis=1)
                replacement = int(np.argmax(sample_weights * residual))
                updated[cluster] = values[replacement]
                continue
            centroid = np.sum(
                values[members] * sample_weights[members, None],
                axis=0,
            )
            norm = float(np.linalg.norm(centroid))
            if norm <= 1.0e-12:
                raise RuntimeError("prototype update produced a zero vector")
            updated[cluster] = centroid / norm
        shift = float(np.max(np.linalg.norm(updated - prototypes, axis=1)))
        prototypes = updated
        if shift <= convergence_tolerance:
            break

    assignments = np.argmax(values @ prototypes.T, axis=1).astype(np.int64)
    return prototypes.astype(np.float32), assignments


def normal_prototype_features(
    descriptors: np.ndarray,
    prototypes: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Return distance, soft distance, assignment entropy and top-two margin."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = _normalize_rows(descriptors)
    centers = _normalize_rows(prototypes)
    if values.shape[1] != centers.shape[1]:
        raise ValueError("descriptor and prototype dimensions must match")

    similarities = np.clip(values @ centers.T, -1.0, 1.0)
    distances = 1.0 - similarities
    nearest_distance = distances.min(axis=1)
    scaled = -distances / float(temperature)
    maximum = scaled.max(axis=1, keepdims=True)
    log_mean_exp = (
        maximum[:, 0]
        + np.log(np.exp(scaled - maximum).mean(axis=1))
    )
    soft_distance = -float(temperature) * log_mean_exp

    assignment_logits = similarities / float(temperature)
    assignment_logits -= assignment_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(assignment_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if centers.shape[0] == 1:
        entropy = np.zeros(values.shape[0], dtype=np.float32)
        margin = np.zeros(values.shape[0], dtype=np.float32)
    else:
        entropy = -np.sum(
            probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0)),
            axis=1,
        ) / np.log(float(centers.shape[0]))
        top_two = np.partition(similarities, -2, axis=1)[:, -2:]
        margin = top_two.max(axis=1) - top_two.min(axis=1)

    return np.stack(
        (nearest_distance, soft_distance, entropy, margin),
        axis=1,
    ).astype(np.float32)


__all__ = [
    "fit_weighted_spherical_prototypes",
    "hierarchical_image_family_weights",
    "normal_prototype_features",
]
