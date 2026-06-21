from __future__ import annotations

"""Morphological refinement of binary pseudo masks (pipeline.md Stage 6).

Operations applied in order:
  1. binary_closing   — disk(5)  — fill small holes
  2. binary_opening   — disk(3)  — remove thin noise
  3. fill_holes       — flood-fill enclosed background regions
  4. remove_small_objects — remove components < min_size pixels
"""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# low-level morphology via max-pool (torch, no scipy dependency)
# ---------------------------------------------------------------------------

def _morph_op(mask: np.ndarray, kernel_size: int, operation: str) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd to preserve spatial dimensions, got {kernel_size}")
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    pad = kernel_size // 2
    if operation == "dilate":
        out = F.max_pool2d(t, kernel_size=kernel_size, stride=1, padding=pad)
    else:  # erode
        out = 1.0 - F.max_pool2d(1.0 - t, kernel_size=kernel_size, stride=1, padding=pad)
    return (out[0, 0].numpy() > 0.5).astype(np.uint8)


def binary_closing(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Dilate then erode — fills small holes."""
    return _morph_op(_morph_op(mask, kernel_size, "dilate"), kernel_size, "erode")


def binary_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Erode then dilate — removes thin noise."""
    return _morph_op(_morph_op(mask, kernel_size, "erode"), kernel_size, "dilate")


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood-fill enclosed background regions (4-connectivity from border)."""
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))

    filled = mask.copy()
    filled[(mask == 0) & (~visited)] = 1
    return filled


def remove_small_objects(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Remove connected components (8-connectivity) smaller than min_size pixels."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    output = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 or visited[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h and 0 <= nc < w
                        and not visited[nr, nc]
                        and mask[nr, nc] > 0
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if len(comp) >= min_size:
                for pr, pc in comp:
                    output[pr, pc] = 1

    return output


def morphological_refinement(
    mask: np.ndarray,
    closing_kernel: int = 5,
    opening_kernel: int = 3,
    min_size: int = 200,
) -> np.ndarray:
    """Full refinement pipeline per pipeline.md Stage 6.

    Returns [H, W] uint8 binary mask.
    """
    mask = binary_closing(mask, kernel_size=closing_kernel)
    mask = binary_opening(mask, kernel_size=opening_kernel)
    mask = fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)
    return mask
