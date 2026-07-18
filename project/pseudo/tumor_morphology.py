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
from dataclasses import dataclass
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
    # Chosen from low-CAM pixels inside the component's support region (see
    # _select_negative_points), not sam_refine.py's bbox-corner heuristic —
    # a wide support mask (by design, see build_tumor_guidance) can have
    # corners that are still close to the lesion, while low-CAM interior
    # points reliably mark "definitely not the lesion" regardless of shape.
    negative_points: tuple[tuple[int, int], ...] = ()


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


def _adaptive_point_count(area: int, base_max_points: int) -> int:
    """Scale positive-point count to component size instead of a fixed count.

    A 20px lesion and a 2000px lesion get the same fixed point budget under a
    constant max_points, which either wastes prompts on tiny lesions or under-
    samples large ones. Small components get fewer points (a single CAM peak
    is usually representative); large components get more, capped by
    base_max_points (the caller's own --points-per-component budget), so this
    never prompts more points than the caller asked for.
    """
    if area < 150:
        target = 2
    elif area < 600:
        target = 3
    elif area < 2000:
        target = 5
    else:
        target = 6
    return max(1, min(base_max_points, target))


def _find_local_maxima(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    max_points: int,
    min_distance: float,
) -> list[tuple[float, int, int]]:
    """Greedy non-maximum suppression over scattered (row, col, value) samples.

    Repeatedly takes the highest remaining value, then removes every
    unselected sample within min_distance of it, so selected peaks are true
    local maxima of the response rather than points from unrelated summary
    statistics (centroid, axis quantiles) that ignore the response entirely.
    """
    order = np.argsort(values)[::-1]
    suppressed = np.zeros(values.shape[0], dtype=bool)
    selected: list[tuple[float, int, int]] = []
    min_distance_sq = min_distance ** 2

    for index in order:
        if suppressed[index]:
            continue
        row, col = int(rows[index]), int(cols[index])
        selected.append((float(values[index]), row, col))
        if len(selected) >= max_points:
            break
        dist_sq = (rows - row) ** 2 + (cols - col) ** 2
        suppressed |= dist_sq < min_distance_sq

    return selected


def _select_negative_points(
    component: np.ndarray,
    cam: np.ndarray,
    positive_points: tuple[tuple[int, int], ...],
    max_points: int = 4,
    low_percentile: float = 20.0,
) -> tuple[tuple[int, int], ...]:
    """Pick negative SAM prompt points at the component's lowest-CAM pixels.

    sam_refine.py's default negative-point heuristic samples the support
    bounding box's corners, which can still land close to the lesion when the
    support region is wide (an intentional tradeoff — see build_tumor_guidance).
    Sampling low-CAM local minima *inside the component itself* instead marks
    "definitely not the lesion, even though it's inside the search region we
    gave SAM" — the exact ambiguity a wide-but-safe support mask creates —
    using the same non-maximum-suppression as the positive points, just on
    inverted CAM values so negatives spread out instead of clustering.
    """
    rows, cols = np.where(component > 0)
    if rows.size == 0 or max_points <= 0:
        return ()

    values = cam[rows, cols]
    threshold = float(np.percentile(values, low_percentile))
    low_mask = values <= threshold
    if not low_mask.any():
        return ()

    low_rows, low_cols, low_values = rows[low_mask], cols[low_mask], values[low_mask]
    min_distance = max(4.0, min(component.shape) * 0.03)
    # Negate values so _find_local_maxima's "highest first" greedy selection
    # picks the lowest-CAM points (local minima) instead of peaks.
    candidates = _find_local_maxima(-low_values, low_rows, low_cols, max_points, min_distance)

    selected: list[tuple[int, int]] = []
    min_sep_sq = (min_distance * 1.5) ** 2
    for _, row, col in candidates:
        if any((row - pr) ** 2 + (col - pc) ** 2 < min_sep_sq for pr, pc in positive_points):
            continue
        selected.append((row, col))
    return tuple(selected)


def _structured_component_points(
    component: np.ndarray,
    tumor_likelihood: np.ndarray,
    cam: np.ndarray,
    max_points: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Pick positive SAM prompt points at true local maxima of the CAM.

    Support masks are intentionally permissive (they only need to avoid
    missing the lesion, per build_tumor_guidance's own design — see that
    function's docstring), so a component can be much larger than the actual
    lesion. Picking points from the component's geometry (centroid, principal-
    axis quantiles) — as an earlier version of this function did — samples
    wherever the *support region* happens to be shaped, not wherever the
    lesion evidence actually peaks, which is a mismatch that shows up as a low
    point-in-lesion hit rate independent of how good the CAM itself is.
    Peaks are also taken from cam alone (not blended with tumor_likelihood):
    the prompt must stay faithful to the classifier's own evidence, while
    tumor_likelihood's intensity/edge terms exist only to shape the support
    region in build_tumor_guidance, not to relocate prompts away from CAM.
    """
    rows, cols = np.where(component > 0)
    if rows.size == 0:
        return ()

    area = int(rows.size)
    adaptive_max_points = _adaptive_point_count(area, max_points)
    values = cam[rows, cols]
    min_distance = max(4.0, min(component.shape) * 0.03)

    peaks = _find_local_maxima(values, rows, cols, adaptive_max_points, min_distance)
    if not peaks:
        return ()

    return tuple((row, col) for _, row, col in peaks)


def build_class_conditioned_components(
    image_rgb: np.ndarray,
    per_class_cams: list[np.ndarray],
    class_weights: list[float] | np.ndarray,
    cam_percentile: float = 85.0,
    min_component_area: int = 40,
    max_components: int = 6,
    points_per_component: int = 3,
    bbox_padding_ratio: float = 0.05,
    negative_points_per_component: int = 4,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[TumorComponent]]:
    """Build tumor prompt components from a single CAM threshold.

    An earlier version of this function ran a much more elaborate morphology
    stage (seed+support percentile pair, morphological_reconstruction,
    lytic/sclerotic anomaly + edge response fusion, CAM-recall-weighted
    multi-component ranking) under the assumption that CAM alone was too
    weak/noisy a signal to threshold directly. This project's own oracle
    diagnostics (support_loss_dice ~ 0 on average, meaning the reconstructed
    support mask essentially never changed Dice once SAM candidates were
    clipped to it) showed that safety net rarely did anything useful on
    BTXRD, so it was removed in favor of this: threshold the fused CAM at a
    single percentile, keep the largest connected component by default, and
    optionally retain a deterministic top-N set for ablation.

    A single connected component is deliberately kept (not top-N by area)
    since BTXRD lesions are typically one contiguous region, and this
    project saw repeatedly, across both SAM mask-selection and CAM-
    refinement debugging, that keeping/scoring multiple small components let
    spurious high-activation noise blobs compete with the real lesion.

    Args:
        image_rgb:      [H, W, 3] uint8, unused directly here but kept for
                         signature parity with bone_morphology.py's function
                         of the same name (--dataset dispatch relies on it).
        per_class_cams: list of [H, W] float32 CAMs, one per active class.
        class_weights:  classifier confidence per active class.
        cam_percentile: Single threshold on the (weighted-max-fused) CAM
                         defining the seed/support region.
        min_component_area, max_components, points_per_component,
        bbox_padding_ratio, negative_points_per_component: same meaning as
                         before.

    Returns:
        (likelihood, support, components): likelihood is the fused CAM
        itself (float32 [H, W]), support is the selected component union,
        components is a list of at most ``max_components`` TumorComponents.
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

    fused_cam = np.zeros_like(per_class_cams[0], dtype=np.float32)
    for cam, weight in zip(per_class_cams, weights):
        fused_cam = np.maximum(fused_cam, _normalise_percentile(cam, low=0.0, high=100.0) * float(weight))

    threshold = float(np.percentile(fused_cam, cam_percentile))
    support = (fused_cam >= threshold).astype(np.uint8)

    components_raw = [c for c in _connected_components(support) if int(c.sum()) >= min_component_area]
    if not components_raw:
        # Nothing survives the area filter -- fall back to whatever CAM
        # produced, same spirit as build_tumor_guidance's empty-support fallback.
        support = (fused_cam >= np.percentile(fused_cam, 85.0)).astype(np.uint8)
        components_raw = _connected_components(support)

    components: list[TumorComponent] = []
    if components_raw:
        # Keep the historical largest-component behavior when max_components=1,
        # but expose a deterministic top-N ablation for cases where CAM breaks
        # a single lesion into several disconnected high-activation islands.
        ranked_components = sorted(components_raw, key=lambda c: int(c.sum()), reverse=True)
        selected_components = ranked_components[: max(1, int(max_components))]
        support = np.zeros_like(support, dtype=np.uint8)
        for component_id, component in enumerate(selected_components):
            support |= component.astype(np.uint8)
            positive_points = _structured_component_points(
                component, tumor_likelihood=fused_cam, cam=fused_cam, max_points=points_per_component
            )
            components.append(
                TumorComponent(
                    component_id=component_id,
                    mask=component,
                    score=float(fused_cam[component.astype(bool)].mean()),
                    bbox=_component_bbox(component, padding_ratio=bbox_padding_ratio),
                    positive_points=positive_points,
                    negative_points=_select_negative_points(
                        component, cam=fused_cam, positive_points=positive_points,
                        max_points=negative_points_per_component,
                    ),
                )
            )

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray((fused_cam * 255).astype(np.uint8), mode="L").save(
            debug_path / "simple_tumor_likelihood.png"
        )
        Image.fromarray(support * 255, mode="L").save(debug_path / "simple_tumor_support.png")

    return fused_cam, support, components[:max_components]


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
    support_threshold = float(np.percentile(tumor_likelihood, support_percentile))
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
