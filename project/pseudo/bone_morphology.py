from __future__ import annotations

"""Bone-specific morphology for radiographs.

The module produces a conservative bone likelihood map before SAM. It combines
locally enhanced X-ray intensity, cortical edge response, and the semantic CAM,
then reconstructs candidate regions from high-confidence bone seeds.
"""

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass(frozen=True)
class BoneComponent:
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


def _cortical_edge_response(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy)
    return _normalise_percentile(magnitude, low=5.0, high=99.0)


def morphological_reconstruction(
    seed: np.ndarray,
    support: np.ndarray,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Grow seed pixels inside support using a single constrained flood-fill."""
    del max_iterations  # retained for API compatibility
    current = seed.astype(bool) & support.astype(bool)
    support_bool = support.astype(bool)
    queue: deque[tuple[int, int]] = deque(map(tuple, np.argwhere(current)))
    h, w = current.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        row, col = queue.popleft()
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and support_bool[nr, nc]
                and not current[nr, nc]
            ):
                current[nr, nc] = True
                queue.append((nr, nc))
    return current.astype(np.uint8)


def _connected_components(binary: np.ndarray) -> list[np.ndarray]:
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
    bone_likelihood: np.ndarray,
    cam: np.ndarray,
    max_points: int = 3,
) -> tuple[tuple[int, int], ...]:
    rows, cols = np.where(component > 0)
    if rows.size == 0:
        return ()

    response = 0.55 * bone_likelihood + 0.45 * cam
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

    # Approximate the component's major axis and add interior points near both
    # ends. This is useful for elongated long-bone candidates.
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


def select_bone_components(
    bone_support: np.ndarray,
    fused_cam: np.ndarray,
    bone_likelihood: np.ndarray,
    min_component_area: int = 40,
    max_components: int = 6,
    points_per_component: int = 3,
    bbox_padding_ratio: float = 0.05,
) -> list[BoneComponent]:
    """Rank full morphology components using CAM, without trimming their shape."""
    cam = _normalise_percentile(fused_cam, low=0.0, high=100.0)
    cam_seed = cam >= float(np.percentile(cam, 85.0))
    ranked: list[tuple[float, np.ndarray]] = []

    for component in _connected_components(bone_support):
        area = int(component.sum())
        if area < min_component_area:
            continue
        region = component.astype(bool)
        intersection = float((region & cam_seed).sum())
        cam_recall = intersection / max(1.0, float(cam_seed.sum()))
        cam_precision = intersection / float(area)
        cam_energy = float(cam[region].mean())
        bone_energy = float(bone_likelihood[region].mean())
        score = 0.35 * cam_recall + 0.20 * cam_precision + 0.20 * cam_energy + 0.25 * bone_energy
        if intersection > 0 or cam_energy >= 0.08:
            ranked.append((score, component))

    if not ranked:
        fallback = _connected_components(bone_support)
        ranked = [
            (float(bone_likelihood[c.astype(bool)].mean()), c)
            for c in fallback
            if int(c.sum()) >= min_component_area
        ]

    ranked.sort(key=lambda item: item[0], reverse=True)
    components: list[BoneComponent] = []
    for component_id, (score, component) in enumerate(ranked[:max_components]):
        components.append(
            BoneComponent(
                component_id=component_id,
                mask=component.astype(np.uint8),
                score=float(score),
                bbox=_component_bbox(component, padding_ratio=bbox_padding_ratio),
                positive_points=_structured_component_points(
                    component,
                    bone_likelihood=bone_likelihood,
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
    seed_percentile: float = 88.0,
    support_percentile: float = 62.0,
    min_component_area: int = 40,
    max_components: int = 6,
    points_per_component: int = 3,
    bbox_padding_ratio: float = 0.05,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[BoneComponent]]:
    """Build candidates per active anatomy CAM, then merge non-duplicates."""
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
    ranked_components: list[BoneComponent] = []

    for class_index, (cam, class_weight) in enumerate(zip(per_class_cams, weights)):
        class_debug_dir = Path(debug_dir) / f"class_{class_index}" if debug_dir is not None else None
        likelihood, support = build_bone_guidance(
            image_rgb,
            cam,
            seed_percentile=seed_percentile,
            support_percentile=support_percentile,
            min_component_area=min_component_area,
            max_components=max_components,
            debug_dir=class_debug_dir,
        )
        components = select_bone_components(
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
    selected: list[BoneComponent] = []
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
            debug_path / "class_conditioned_bone_likelihood.png"
        )
        Image.fromarray(combined_support * 255, mode="L").save(
            debug_path / "class_conditioned_bone_support.png"
        )
        for component in selected:
            Image.fromarray(component.mask * 255, mode="L").save(
                debug_path / f"selected_bone_component_{component.component_id}.png"
            )

    return combined_likelihood, combined_support, selected


def _select_cam_supported_components(
    reconstructed: np.ndarray,
    cam: np.ndarray,
    bone_likelihood: np.ndarray,
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
        bone_energy = float(bone_likelihood[region].mean())
        score = 0.30 * cam_recall + 0.20 * cam_precision + 0.25 * cam_energy + 0.25 * bone_energy
        if cam_energy > 0.05 or cam_precision > 0.0:
            ranked.append((score, component))

    if not ranked:
        return reconstructed.astype(np.uint8)

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(reconstructed, dtype=np.uint8)
    for _, component in ranked[:max_components]:
        selected |= component
    return selected


def build_bone_guidance(
    image_rgb: np.ndarray,
    fused_cam: np.ndarray,
    seed_percentile: float = 88.0,
    support_percentile: float = 62.0,
    min_component_area: int = 40,
    max_components: int = 8,
    use_clahe: bool = True,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous bone likelihood and reconstructed binary support.

    The CAM remains the semantic cue. Intensity and cortical edges refine it
    toward radiopaque bone structures and away from soft-tissue silhouettes.
    """
    gray = _enhance_grayscale(image_rgb, use_clahe=use_clahe)
    edge = _cortical_edge_response(gray)
    cam = _normalise_percentile(fused_cam, low=0.0, high=100.0)

    # RAM-H1200 only has a hand-level image label, so the classifier CAM often
    # marks the hand silhouette. Weight radiographic evidence more strongly and
    # keep CAM as a weak semantic anchor rather than the main foreground cue.
    bone_likelihood = 0.58 * gray + 0.32 * edge + 0.10 * cam
    bone_likelihood = _normalise_percentile(bone_likelihood, low=1.0, high=99.0)

    seed_threshold = float(np.percentile(bone_likelihood, seed_percentile))
    # min() so that a caller passing support_percentile < 68 actually gets a
    # looser threshold (more support pixels) rather than being silently overridden.
    support_threshold = float(
        min(
            np.percentile(bone_likelihood, support_percentile),
            np.percentile(bone_likelihood, 68.0),
        )
    )
    cam_gate = cam >= float(np.percentile(cam, 65.0))
    relaxed_cam_gate = cam >= float(np.percentile(cam, 25.0))
    strong_bone_evidence = (
        ((gray >= float(np.percentile(gray, 72.0))) & (edge >= float(np.percentile(edge, 45.0))))
        | (edge >= float(np.percentile(edge, 82.0)))
    )
    radiographic_support = (
        ((gray >= float(np.percentile(gray, 66.0))) & (edge >= float(np.percentile(edge, 35.0))))
        | (edge >= float(np.percentile(edge, 78.0)))
        | ((gray >= float(np.percentile(gray, 76.0))) & relaxed_cam_gate)
    )

    seeds = (
        (bone_likelihood >= seed_threshold)
        & (cam_gate | strong_bone_evidence)
        & strong_bone_evidence
    ).astype(np.uint8)
    support = (
        (bone_likelihood >= support_threshold)
        & radiographic_support
        & (relaxed_cam_gate | strong_bone_evidence)
    ).astype(np.uint8)
    reconstructed = morphological_reconstruction(seeds, support)
    reconstructed = _select_cam_supported_components(
        reconstructed,
        cam=cam,
        bone_likelihood=bone_likelihood,
        min_component_area=min_component_area,
        max_components=max_components,
    )

    # If morphology produces no reliable support, retain the original CAM path.
    if not reconstructed.any():
        reconstructed = (cam >= np.percentile(cam, 85.0)).astype(np.uint8)

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray((gray * 255).astype(np.uint8), mode="L").save(debug_path / "bone_gray_enhanced.png")
        Image.fromarray((edge * 255).astype(np.uint8), mode="L").save(debug_path / "bone_edge_response.png")
        Image.fromarray((bone_likelihood * 255).astype(np.uint8), mode="L").save(debug_path / "bone_likelihood.png")
        Image.fromarray(seeds * 255, mode="L").save(debug_path / "bone_seeds.png")
        Image.fromarray(reconstructed * 255, mode="L").save(debug_path / "bone_support.png")

    return bone_likelihood.astype(np.float32), reconstructed.astype(np.uint8)


def fuse_cam_with_bone_guidance(
    fused_cam: np.ndarray,
    bone_likelihood: np.ndarray,
    bone_support: np.ndarray,
) -> np.ndarray:
    """Create the prompt map while preserving CAM as the semantic anchor."""
    cam = _normalise_percentile(fused_cam, low=0.0, high=100.0)
    support_bonus = bone_support.astype(np.float32)
    prompt_map = 0.50 * cam + 0.40 * bone_likelihood + 0.10 * support_bonus
    return _normalise_percentile(prompt_map, low=0.0, high=100.0).astype(np.float32)
