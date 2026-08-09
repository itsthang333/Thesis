from __future__ import annotations

"""Disease-specific late-localization (DSLL) utilities.

The image classifier remains the frozen ten-class DenseNet.  Disease classes
are selected only from its prediction, never from the image label or spatial
annotation.  Each contrastive LayerCAM is normalized before posterior-weighted
late fusion, which is the key distinction from early logit collapse.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DSLLSource:
    source_id: str
    class_id: int
    probability: float
    rank: int
    cam: np.ndarray


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    values = np.asarray(cam, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("DSLL CAM must be a finite 2-D array")
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1.0e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def conditional_disease_topk(logits: torch.Tensor, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 10:
        raise ValueError("DSLL requires one ten-class logit vector")
    if not 1 <= k <= 9:
        raise ValueError("Disease top-k must be in [1, 9]")
    posterior = torch.softmax(logits[0, 1:], dim=0)
    probabilities, local_indices = torch.topk(posterior, k=k, largest=True, sorted=True)
    return (
        (local_indices + 1).detach().cpu().numpy().astype(np.int16),
        probabilities.detach().cpu().numpy().astype(np.float32),
    )


def _cam_with_optional_flip(layercam, image_tensor: torch.Tensor, *, class_id: int | None, flip: bool) -> np.ndarray:
    def one(tensor: torch.Tensor) -> np.ndarray:
        if class_id is None:
            output = layercam.cam_for_tumor_log_odds(tensor)
        else:
            output = layercam.cam_for_class_contrast(tensor, int(class_id), reference_index=0)
        return output.cam[0].detach().cpu().numpy().astype(np.float32)

    original = one(image_tensor)
    if not flip:
        return normalize_cam(original)
    mirrored = np.fliplr(one(torch.flip(image_tensor, dims=[3])))
    return normalize_cam(0.5 * (original + mirrored))


def generate_dsll_sources(
    layercam,
    image_tensor: torch.Tensor,
    classifier_logits: torch.Tensor,
    *,
    flip_tta: bool,
) -> list[DSLLSource]:
    class_ids, probabilities = conditional_disease_topk(classifier_logits, k=3)
    generic = _cam_with_optional_flip(
        layercam, image_tensor, class_id=None, flip=flip_tta
    )
    disease_maps = [
        _cam_with_optional_flip(
            layercam, image_tensor, class_id=int(class_id), flip=flip_tta
        )
        for class_id in class_ids
    ]
    normalized_weights = probabilities / max(float(probabilities.sum()), 1.0e-8)
    late_fusion = normalize_cam(
        np.sum(
            np.stack(disease_maps, axis=0)
            * normalized_weights[:, None, None],
            axis=0,
        )
    )
    sources = [DSLLSource("tumor_logodds_320", -1, 1.0, 0, generic)]
    sources.extend(
        DSLLSource(
            f"disease_top{rank}_320",
            int(class_id),
            float(probability),
            rank,
            cam,
        )
        for rank, (class_id, probability, cam) in enumerate(
            zip(class_ids, probabilities, disease_maps, strict=True), start=1
        )
    )
    sources.append(DSLLSource("disease_latefusion_320", -2, 1.0, 0, late_fusion))
    return sources


def average_percentile_ranks(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if len(values) != len(groups):
        raise ValueError("Rank values and groups differ")
    result = np.zeros(len(values), dtype=np.float32)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        local = values[indices]
        if len(indices) == 1:
            result[indices] = 1.0
            continue
        order = np.argsort(local, kind="mergesort")
        sorted_values = local[order]
        ranks = np.empty(len(indices), dtype=np.float64)
        start = 0
        while start < len(indices):
            end = start + 1
            while end < len(indices) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        result[indices] = (ranks / float(len(indices) - 1)).astype(np.float32)
    return result


def source_specific_candidate_features(
    masks: np.ndarray,
    component_ids: np.ndarray,
    component_maps: dict[int, np.ndarray],
    sam_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = np.asarray(masks).astype(bool)
    component_ids = np.asarray(component_ids, dtype=np.int32).reshape(-1)
    sam_scores = np.asarray(sam_scores, dtype=np.float32).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(component_ids) or len(masks) != len(sam_scores):
        raise ValueError("DSLL candidates, component IDs and SAM scores differ")
    mean_scores = np.zeros(len(masks), dtype=np.float32)
    mass_coverages = np.zeros(len(masks), dtype=np.float32)
    densities = np.zeros(len(masks), dtype=np.float32)
    for index, (mask, component_id) in enumerate(zip(masks, component_ids, strict=True)):
        source_map = np.asarray(component_maps[int(component_id)], dtype=np.float32)
        if source_map.shape != mask.shape:
            raise ValueError("Source map and candidate mask geometry differ")
        if not mask.any():
            continue
        values = source_map[mask]
        mean_scores[index] = float(values.mean())
        mass_coverages[index] = float(values.sum() / max(float(source_map.sum()), 1.0e-8))
        densities[index] = float((values > 0.5).mean())
    sam_ranks = average_percentile_ranks(sam_scores, component_ids)
    source_scores = 0.60 * densities + 0.25 * mass_coverages + 0.15 * sam_ranks
    return mean_scores, mass_coverages, densities, source_scores.astype(np.float32)
