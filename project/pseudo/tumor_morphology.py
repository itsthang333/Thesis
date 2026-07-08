from __future__ import annotations

"""Tumor-specific morphology for BTXRD radiographs.

Unlike RAM-H1200 (hand-only label, so CAM only anchors the whole-hand
silhouette and the "bone_likelihood" prior in bone_morphology.py leans on
radiopaque intensity + cortical edges to separate bone from soft tissue),
BTXRD's classifier is trained directly on a tumor-vs-normal label. Its CAM is
therefore a much stronger localization signal for *where the lesion is*, so
this module inverts the weighting used in bone_morphology.py: CAM dominates,
and the radiographic term looks for local anomalies (both lytic/dark and
sclerotic/bright) instead of assuming the target region is always radiopaque.
"""

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass(frozen=True)
class TumorComponent:
    """A complete morphology proposal selected by semantic CAM evidence."""

    component_id: int
    mask: np.ndarray
    score: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    positive_points: tuple[tuple[int, int], ...]  # row, col


def _normalise_percentile(
    values: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _enhance_grayscale(image_rgb: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    if image_rgb.ndim == 3:
        gray = (
            0.299 * image_rgb[..., 0]
            + 0.587 * image_rgb[..., 1]
            + 0.114 * image_rgb[..., 2]
        ).astype(np.uint8)
    else:
        gray = image_rgb.astype(np.uint8)

    if use_clahe and cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return _normalise_percentile(gray)


def _edge_response(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy)
    return _normalise_percentile(magnitude, low=5.0, high=99.0)


def _local_mean(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Box-filtered local mean, used as a cheap regional-intensity baseline."""
    if cv2 is not None:
        return cv2.blur(values.astype(np.float32), (kernel_size, kernel_size))
    pad = kernel_size // 2
    padded = np.pad(values.astype(np.float32), pad, mode="reflect")
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
    h, w = values.shape
    total = (
        cumsum[kernel_size:kernel_size + h, kernel_size:kernel_size + w]
        - cumsum[0:h, kernel_size:kernel_size + w]
        - cumsum[kernel_size:kernel_size + h, 0:w]
        + cumsum[0:h, 0:w]
    )
    return total / float(kernel_size * kernel_size)


def _local_anomaly_response(gray: np.ndarray, kernel_size: int = 41) -> np.ndarray:
    """Absolute deviation from the local regional mean.

    Bone tumors can be lytic (locally darker than surrounding bone) or
    sclerotic (locally brighter), so this looks for deviation magnitude in
    either direction rather than assuming "bright == target" like
    bone_morphology.py does for plain bone-vs-soft-tissue separation.
    """
    local_mean = _local_mean(gray, kernel_size)
    deviation = np.abs(gray - local_mean)
    return _normalise_percentile(deviation, low=2.0, high=99.0)


def _dilate3x3(mask: np.ndarray) -> np.ndarray:
    """8-connected one-pixel dilation. Uses cv2 when available, else numpy shifts."""
    mask = mask.astype(np.uint8)
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr : 1 + dr + mask.shape[0], 1 + dc : 1 + dc + mask.shape[1]]
    return out.astype(np.uint8)


def morphological_reconstruction(
    seed: np.ndarray,
    support: np.ndarray,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Grow seed pixels inside support via iterative dilation-and-intersect."""
    support_bool = support.astype(bool)
    current = seed.astype(bool) & support_bool
    if not current.any():
        return current.astype(np.uint8)

    iterations = 0
    limit = max_iterations if max_iterations is not None else max(current.shape)
    while iterations < limit:
        grown = _dilate3x3(current).astype(bool) & support_bool
        if np.array_equal(grown, current):
            break
        current = grown
        iterations += 1
    return current.astype(np.uint8)


def _connected_components(binary: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected binary component masks. Uses cv2 when available, else BFS."""
    binary = binary.astype(np.uint8)
    if cv2 is not None:
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        return [(labels == label_id).astype(np.uint8) for label_id in range(1, num_labels)]

    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for row in range(h):
        for col in range(w):
            if not binary[row, col] or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and binary[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            component = np.zeros((h, w), dtype=np.uint8)
            rr, cc = zip(*coords)
            component[np.asarray(rr), np.asarray(cc)] = 1
            components.append(component)
    return components


def _component_bbox(mask: np.ndarray, padding_ratio: float = 0.05) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask > 0)
    h, w = mask.shape
    if rows.size == 0:
        return 0, 0, w - 1, h - 1
    x0, x1 = int(cols.min()), int(cols.max())
    y0, y1 = int(rows.min()), int(rows.max())
    pad_x = max(2, int((x1 - x0 + 1) * padding_ratio))
    pad_y = max(2, int((y1 - y0 + 1) * padding_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w - 1, x1 + pad_x),
        min(h - 1, y1 + pad_y),
    )


def _structured_component_points(
    component: np.ndarray,
    tumor_likelihood: np.ndarray,
    cam: np.ndarray,
    max_points: int = 3,
) -> tuple[tuple[int, int], ...]:
    rows, cols = np.where(component > 0)
    if rows.size == 0:
        return ()

    response = 0.45 * tumor_likelihood + 0.55 * cam
    values = response[rows, cols]
    peak_index = int(np.argmax(values))
    candidates: list[tuple[float, int, int]] = [
        (float(values[peak_index]) + 1.0, int(rows[peak_index]), int(cols[peak_index]))
    ]

    centroid_r = float(rows.mean())
    centroid_c = float(cols.mean())
    centroid_index = int(np.argmin((rows - centroid_r) ** 2 + (cols - centroid_c) ** 2))
    candidates.append(
        (float(values[centroid_index]) + 0.5, int(rows[centroid_index]), int(cols[centroid_index]))
    )

    coords = np.stack([rows, cols], axis=1).astype(np.float32)
    if coords.shape[0] >= 3:
        centered = coords - coords.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, coords.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projections = centered @ major_axis
        for quantile in (0.25, 0.75):
            target = float(np.quantile(projections, quantile))
            axis_index = int(np.argmin(np.abs(projections - target)))
            candidates.append(
                (float(values[axis_index]) + 0.25, int(rows[axis_index]), int(cols[axis_index]))
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, int]] = []
    min_distance = max(4.0, min(component.shape) * 0.025)
    for _, row, col in candidates:
        if all((row - pr) ** 2 + (col - pc) ** 2 >= min_distance ** 2 for pr, pc in selected):
            selected.append((row, col))
        if len(selected) >= max_points:
            break
    return tuple(selected)


def select_tumor_components(
    tumor_support: np.ndarray,
    fused_cam: np.ndarray,
    tumor_likelihood: np.ndarray,
    min_component_area: int = 40,
    max_components: int = 6,
    points_per_component: int = 3,
    bbox_padding_ratio: float = 0.05,
) -> list[TumorComponent]:
    """Rank full morphology components using CAM, without trimming their shape."""
    cam = _normalise_percentile(fused_cam, low=0.0, high=100.0)
    cam_seed = cam >= float(np.percentile(cam, 85.0))
    ranked: list[tuple[float, np.ndarray]] = []

    for component in _connected_components(tumor_support):
        area = int(component.sum())
        if area < min_component_area:
            continue
        region = component.astype(bool)
        intersection = float((region & cam_seed).sum())
        cam_recall = intersection / max(1.0, float(cam_seed.sum()))
        cam_precision = intersection / float(area)
        cam_energy = float(cam[region].mean())
        tumor_energy = float(tumor_likelihood[region].mean())
        score = 0.40 * cam_recall + 0.20 * cam_precision + 0.25 * cam_energy + 0.15 * tumor_energy
        if intersection > 0 or cam_energy >= 0.08:
            ranked.append((score, component))

    if not ranked:
        fallback = _connected_components(tumor_support)
        ranked = [
            (float(tumor_likelihood[c.astype(bool)].mean()), c)
            for c in fallback
            if int(c.sum()) >= min_component_area
        ]

    ranked.sort(key=lambda item: item[0], reverse=True)
    components: list[TumorComponent] = []
    for component_id, (score, component) in enumerate(ranked[:max_components]):
        components.append(
            TumorComponent(
                component_id=component_id,
                mask=component.astype(np.uint8),
                score=float(score),
                bbox=_component_bbox(component, padding_ratio=bbox_padding_ratio),
                positive_points=_structured_component_points(
                    component,
                    tumor_likelihood=tumor_likelihood,
                    cam=cam,
                    max_points=points_per_component,
                ),
            )
        )
    return components


def build_class_conditioned_components(
    image_rgb: np.ndarray,
    per_class_cams: list[np.ndarray],
    class_weights: list[float] | np.ndarray,
    seed_percentile: float = 82.0,
    support_percentile: float = 55.0,
    min_component_area: int = 40,
    max_components: int = 6,
    points_per_component: int = 3,
    bbox_padding_ratio: float = 0.05,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[TumorComponent]]:
    """Build candidates per active class CAM, then merge non-duplicates.

    Signature matches pseudo.bone_morphology.build_class_conditioned_components
    so generate_pseudo_masks.py/inference.py/visualize_pipeline.py can select
    between bone- and tumor-morphology purely via --dataset.
    """
    if not per_class_cams:
        h, w = image_rgb.shape[:2]
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.uint8),
            [],
        )

    weights = np.asarray(class_weights, dtype=np.float32)
    if weights.size != len(per_class_cams):
        weights = np.ones(len(per_class_cams), dtype=np.float32)
    weights = weights / (weights.max() + 1e-8)

    combined_likelihood = np.zeros_like(per_class_cams[0], dtype=np.float32)
    combined_support = np.zeros_like(per_class_cams[0], dtype=np.uint8)
    ranked_components: list[TumorComponent] = []

    for class_index, (cam, class_weight) in enumerate(zip(per_class_cams, weights)):
        class_debug_dir = Path(debug_dir) / f"class_{class_index}" if debug_dir is not None else None
        likelihood, support = build_tumor_guidance(
            image_rgb,
            cam,
            seed_percentile=seed_percentile,
            support_percentile=support_percentile,
            min_component_area=min_component_area,
            max_components=max_components,
            debug_dir=class_debug_dir,
        )
        components = select_tumor_components(
            support,
            cam,
            likelihood,
            min_component_area=min_component_area,
            max_components=max_components,
            points_per_component=points_per_component,
            bbox_padding_ratio=bbox_padding_ratio,
        )
        combined_likelihood = np.maximum(combined_likelihood, likelihood * float(class_weight))
        combined_support |= support
        ranked_components.extend(
            replace(component, score=component.score * float(class_weight))
            for component in components
        )

    ranked_components.sort(key=lambda component: component.score, reverse=True)
    selected: list[TumorComponent] = []
    for candidate in ranked_components:
        candidate_mask = candidate.mask.astype(bool)
        duplicate = False
        for existing in selected:
            existing_mask = existing.mask.astype(bool)
            intersection = float((candidate_mask & existing_mask).sum())
            union = float((candidate_mask | existing_mask).sum())
            if intersection / max(1.0, union) >= 0.65:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_components:
            break

    selected = [
        replace(component, component_id=index)
        for index, component in enumerate(selected)
    ]
    selected_support = np.zeros_like(combined_support, dtype=np.uint8)
    for component in selected:
        selected_support |= component.mask
    if selected_support.any():
        combined_support = selected_support

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray((combined_likelihood * 255).astype(np.uint8), mode="L").save(
            debug_path / "class_conditioned_tumor_likelihood.png"
        )
        Image.fromarray(combined_support * 255, mode="L").save(
            debug_path / "class_conditioned_tumor_support.png"
        )
        for component in selected:
            Image.fromarray(component.mask * 255, mode="L").save(
                debug_path / f"selected_tumor_component_{component.component_id}.png"
            )

    return combined_likelihood, combined_support, selected


def _select_cam_supported_components(
    reconstructed: np.ndarray,
    cam: np.ndarray,
    tumor_likelihood: np.ndarray,
    min_component_area: int,
    max_components: int,
) -> np.ndarray:
    cam_seed = cam >= np.percentile(cam, 85.0)
    ranked: list[tuple[float, np.ndarray]] = []

    for component in _connected_components(reconstructed):
        area = int(component.sum())
        if area < min_component_area:
            continue
        region = component.astype(bool)
        cam_recall = float((region & cam_seed).sum()) / max(1.0, float(cam_seed.sum()))
        cam_precision = float((region & cam_seed).sum()) / float(area)
        cam_energy = float(cam[region].mean())
        tumor_energy = float(tumor_likelihood[region].mean())
        score = 0.35 * cam_recall + 0.20 * cam_precision + 0.30 * cam_energy + 0.15 * tumor_energy
        if cam_energy > 0.05 or cam_precision > 0.0:
            ranked.append((score, component))

    if not ranked:
        return reconstructed.astype(np.uint8)

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(reconstructed, dtype=np.uint8)
    for _, component in ranked[:max_components]:
        selected |= component
    return selected


def build_tumor_guidance(
    image_rgb: np.ndarray,
    fused_cam: np.ndarray,
    seed_percentile: float = 82.0,
    support_percentile: float = 55.0,
    min_component_area: int = 40,
    max_components: int = 8,
    use_clahe: bool = True,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous tumor likelihood and reconstructed binary support.

    CAM is the dominant cue here (unlike bone_morphology.py's hand-silhouette
    CAM, BTXRD's classifier is trained directly on tumor presence, so its CAM
    already localizes the lesion). Local intensity anomaly and edge response
    refine CAM toward focal abnormal regions rather than assuming radiopaque
    intensity means "target", since lytic lesions are locally darker than
    surrounding bone while sclerotic lesions are locally brighter.
    """
    gray = _enhance_grayscale(image_rgb, use_clahe=use_clahe)
    edge = _edge_response(gray)
    anomaly = _local_anomaly_response(gray)
    cam = _normalise_percentile(fused_cam, low=0.0, high=100.0)

    tumor_likelihood = 0.55 * cam + 0.25 * anomaly + 0.20 * edge
    tumor_likelihood = _normalise_percentile(tumor_likelihood, low=1.0, high=99.0)

    seed_threshold = float(np.percentile(tumor_likelihood, seed_percentile))
    support_threshold = float(
        min(
            np.percentile(tumor_likelihood, support_percentile),
            np.percentile(tumor_likelihood, 55.0),
        )
    )
    cam_gate = cam >= float(np.percentile(cam, 60.0))
    relaxed_cam_gate = cam >= float(np.percentile(cam, 30.0))
    strong_anomaly_evidence = (
        (anomaly >= float(np.percentile(anomaly, 70.0)))
        | (edge >= float(np.percentile(edge, 80.0)))
    )

    seeds = (
        (tumor_likelihood >= seed_threshold) & cam_gate
    ).astype(np.uint8)
    support = (
        (tumor_likelihood >= support_threshold)
        & (relaxed_cam_gate | strong_anomaly_evidence)
    ).astype(np.uint8)
    reconstructed = morphological_reconstruction(seeds, support)
    reconstructed = _select_cam_supported_components(
        reconstructed,
        cam=cam,
        tumor_likelihood=tumor_likelihood,
        min_component_area=min_component_area,
        max_components=max_components,
    )

    # If morphology produces no reliable support, retain the original CAM path.
    if not reconstructed.any():
        reconstructed = (cam >= np.percentile(cam, 85.0)).astype(np.uint8)

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray((gray * 255).astype(np.uint8), mode="L").save(debug_path / "tumor_gray_enhanced.png")
        Image.fromarray((edge * 255).astype(np.uint8), mode="L").save(debug_path / "tumor_edge_response.png")
        Image.fromarray((anomaly * 255).astype(np.uint8), mode="L").save(debug_path / "tumor_anomaly_response.png")
        Image.fromarray((tumor_likelihood * 255).astype(np.uint8), mode="L").save(debug_path / "tumor_likelihood.png")
        Image.fromarray(seeds * 255, mode="L").save(debug_path / "tumor_seeds.png")
        Image.fromarray(reconstructed * 255, mode="L").save(debug_path / "tumor_support.png")

    return tumor_likelihood.astype(np.float32), reconstructed.astype(np.uint8)


# Alias matching bone_morphology.py's public name so call sites can import
# either module under the same local name based on --dataset.
build_bone_guidance = build_tumor_guidance


def fuse_cam_with_bone_guidance(
    fused_cam: np.ndarray,
    bone_likelihood: np.ndarray,
    bone_support: np.ndarray,
) -> np.ndarray:
    """Create the prompt map while preserving CAM as the semantic anchor."""
    cam = _normalise_percentile(fused_cam, low=0.0, high=100.0)
    support_bonus = bone_support.astype(np.float32)
    prompt_map = 0.60 * cam + 0.30 * bone_likelihood + 0.10 * support_bonus
    return _normalise_percentile(prompt_map, low=0.0, high=100.0).astype(np.float32)
