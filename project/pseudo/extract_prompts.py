from __future__ import annotations

"""Extract SAM point prompts from a fused bone CAM.

Pipeline (per pipeline.md Stage 3):
  1. Adaptive percentile threshold → binary foreground mask
  2. Connected-components labelling (4-connectivity BFS)
  3. Peak extraction: argmax(CAM) inside each component
  4. Optional cap at max_points (sorted by CAM value, highest first)
"""

from collections import deque

import numpy as np


def _connected_components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connectivity BFS over foreground pixels."""
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if binary[r, c] == 0 or visited[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and binary[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def extract_point_prompts(
    bone_cam: np.ndarray,
    cam_percentile: float = 85.0,
    max_points: int = 5,
    min_component_area: int = 100,
    support_mask: np.ndarray | None = None,
    debug_dir: str | None = None,
    image_pil=None,
) -> list[tuple[int, int]]:
    """Return (row, col) peak points for SAM prompts.

    Args:
        bone_cam:            [H, W] float32 in [0, 1].
        cam_percentile:      Threshold percentile (85 / 90 / 95 per pipeline.md).
        max_points:          Cap on number of prompt points.
        min_component_area:  Ignore components smaller than this.
        debug_dir:           If set, saves foreground.png, component_*.png, layercam_with_points.png.
        image_pil:           PIL Image for the CAM+points overlay.

    Returns:
        List of (row, col) tuples, sorted by CAM value descending.
    """
    threshold = float(np.percentile(bone_cam, cam_percentile))
    fg = (bone_cam > threshold).astype(np.uint8)
    if support_mask is not None and support_mask.any():
        supported = (fg.astype(bool) & support_mask.astype(bool)).astype(np.uint8)
        if supported.any():
            fg = supported

    components = _connected_components(fg)

    peaks: list[tuple[float, int, int]] = []  # (priority, row, col)
    for comp in components:
        if len(comp) < min_component_area:
            continue
        rows = np.array([p[0] for p in comp])
        cols = np.array([p[1] for p in comp])
        cam_vals = bone_cam[rows, cols]

        # Semantic point: strongest CAM/bone-likelihood response.
        best_idx = int(np.argmax(cam_vals))
        r, c = int(rows[best_idx]), int(cols[best_idx])
        peaks.append((float(cam_vals[best_idx]), r, c))

        # Geometric point: component pixel nearest its centroid. This is more
        # likely to lie inside bone than an edge-only activation.
        centroid_r = float(rows.mean())
        centroid_c = float(cols.mean())
        distances = (rows - centroid_r) ** 2 + (cols - centroid_c) ** 2
        center_idx = int(np.argmin(distances))
        center_r, center_c = int(rows[center_idx]), int(cols[center_idx])
        if (center_r, center_c) != (r, c):
            peaks.append((float(cam_vals[center_idx]) + 0.05, center_r, center_c))

    # Prefer high-confidence points while suppressing near-duplicates.
    peaks.sort(key=lambda x: x[0], reverse=True)
    result: list[tuple[int, int]] = []
    min_distance = max(6.0, min(bone_cam.shape) * 0.04)
    for _, r, c in peaks:
        if all((r - pr) ** 2 + (c - pc) ** 2 >= min_distance ** 2 for pr, pc in result):
            result.append((r, c))
        if len(result) >= max_points:
            break

    # Robust fallback for sparse wrist/finger candidates or aggressive
    # morphology thresholds.
    if not result:
        candidate = support_mask.astype(bool) if support_mask is not None and support_mask.any() else np.ones_like(bone_cam, dtype=bool)
        masked_values = np.where(candidate, bone_cam, -np.inf)
        flat_index = int(np.argmax(masked_values))
        r, c = np.unravel_index(flat_index, bone_cam.shape)
        result = [(int(r), int(c))]

    if debug_dir is not None:
        _save_prompt_debug(debug_dir, bone_cam, fg, components, min_component_area, result, image_pil)

    return result


def _save_prompt_debug(
    debug_dir,
    bone_cam: np.ndarray,
    fg: np.ndarray,
    components: list,
    min_component_area: int,
    point_prompts: list[tuple[int, int]],
    image_pil,
) -> None:
    """Save foreground mask, per-component masks, and CAM+points overlay."""
    from pathlib import Path
    from PIL import Image as _Image, ImageDraw

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # foreground mask
    _Image.fromarray(fg * 255, mode="L").save(debug_dir / "foreground.png")

    # per-component masks (only those above min_component_area)
    h, w = bone_cam.shape
    comp_idx = 0
    for comp in components:
        if len(comp) < min_component_area:
            continue
        comp_mask = np.zeros((h, w), dtype=np.uint8)
        for r, c in comp:
            comp_mask[r, c] = 255
        _Image.fromarray(comp_mask, mode="L").save(debug_dir / f"component_{comp_idx}.png")
        comp_idx += 1

    # CAM + prompt points overlay — jet colormap
    red = np.clip(1.5 - np.abs(4.0 * bone_cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * bone_cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * bone_cam - 1.0), 0.0, 1.0)
    cam_rgb = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)

    if image_pil is not None:
        base = np.array(image_pil.convert("RGB")).astype(np.float32)
        cam_rgb_f = cam_rgb.astype(np.float32)
        blended = (0.55 * base + 0.45 * cam_rgb_f).clip(0, 255).astype(np.uint8)
        overlay_img = _Image.fromarray(blended)
    else:
        overlay_img = _Image.fromarray(cam_rgb)

    draw = ImageDraw.Draw(overlay_img)
    for i, (r, c) in enumerate(point_prompts):
        radius = max(6, min(h, w) // 60)
        draw.ellipse(
            [c - radius, r - radius, c + radius, r + radius],
            fill=(255, 0, 0),
            outline=(255, 255, 255),
        )
        draw.text((c + radius + 2, r - radius), str(i), fill=(255, 255, 0))

    overlay_img.save(debug_dir / "layercam_with_points.png")
