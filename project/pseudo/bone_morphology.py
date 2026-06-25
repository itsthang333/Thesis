from __future__ import annotations

"""Bone-specific morphology for radiographs.

The module produces a conservative bone likelihood map before SAM. It combines
locally enhanced X-ray intensity, cortical edge response, and the semantic CAM,
then reconstructs candidate regions from high-confidence bone seeds.
"""

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from pseudo.morphology import binary_closing

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


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

    # Bright radiopaque structures and paired cortical edges are useful but not
    # sufficient alone; CAM suppresses text markers and unrelated anatomy.
    bone_likelihood = 0.45 * gray + 0.25 * edge + 0.30 * cam
    bone_likelihood = _normalise_percentile(bone_likelihood, low=1.0, high=99.0)

    seed_threshold = float(np.percentile(bone_likelihood, seed_percentile))
    support_threshold = float(np.percentile(bone_likelihood, support_percentile))
    cam_gate = cam >= float(np.percentile(cam, 65.0))
    relaxed_cam_gate = cam >= float(np.percentile(cam, 35.0))
    radiographic_support = (
        (gray >= float(np.percentile(gray, 55.0)))
        | (edge >= float(np.percentile(edge, 75.0)))
    )

    seeds = ((bone_likelihood >= seed_threshold) & cam_gate).astype(np.uint8)
    support = (
        (bone_likelihood >= support_threshold)
        & relaxed_cam_gate
        & radiographic_support
    ).astype(np.uint8)
    support = binary_closing(support, kernel_size=3)
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
