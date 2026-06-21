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
) -> list[tuple[int, int]]:
    """Return (row, col) peak points for SAM prompts.

    Args:
        bone_cam:            [H, W] float32 in [0, 1].
        cam_percentile:      Threshold percentile (85 / 90 / 95 per pipeline.md).
        max_points:          Cap on number of prompt points.
        min_component_area:  Ignore components smaller than this.

    Returns:
        List of (row, col) tuples, sorted by CAM value descending.
    """
    threshold = float(np.percentile(bone_cam, cam_percentile))
    fg = (bone_cam > threshold).astype(np.uint8)

    components = _connected_components(fg)

    peaks: list[tuple[float, int, int]] = []  # (cam_value, row, col)
    for comp in components:
        if len(comp) < min_component_area:
            continue
        rows = np.array([p[0] for p in comp])
        cols = np.array([p[1] for p in comp])
        cam_vals = bone_cam[rows, cols]
        best_idx = int(np.argmax(cam_vals))
        r, c = int(rows[best_idx]), int(cols[best_idx])
        peaks.append((float(cam_vals[best_idx]), r, c))

    # sort by cam value descending, cap at max_points
    peaks.sort(key=lambda x: x[0], reverse=True)
    peaks = peaks[:max_points]

    return [(r, c) for _, r, c in peaks]
