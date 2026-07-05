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

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


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


def _label_components_bfs(mask: np.ndarray, offsets: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    """Pure-Python BFS fallback labeling. Returns (labels, num_components), 0 = background."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 or labels[r, c] != 0:
                continue
            next_label += 1
            queue: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_label
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h and 0 <= nc < w
                        and labels[nr, nc] == 0
                        and mask[nr, nc] > 0
                    ):
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
    return labels, next_label


def _label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label connected components. Uses cv2 when available (much faster), else BFS.

    Returns (labels, num_components) where labels is [H, W] int32 with 0 = background
    and components numbered 1..num_components.
    """
    mask = mask.astype(np.uint8)
    if cv2 is not None:
        num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)
        return labels.astype(np.int32), num_labels - 1

    offsets_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets_8 = offsets_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offsets = offsets_8 if connectivity == 8 else offsets_4
    return _label_components_bfs(mask, offsets)


def fill_holes(mask: np.ndarray, max_hole_area: int | None = None) -> np.ndarray:
    """Fill enclosed holes, optionally only when they are sufficiently small."""
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    background = (mask == 0).astype(np.uint8)

    # Label background components (4-connectivity, matching the original flood-fill)
    # then mark labels touching the border as "outside" — everything else is a hole.
    labels, num_labels = _label_components(background, connectivity=4)
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border_labels.discard(0)

    holes = background.astype(bool) & ~np.isin(labels, list(border_labels))

    if max_hole_area is None:
        filled = mask.copy()
        filled[holes] = 1
        return filled

    filled = mask.copy()
    for component in _component_masks(holes.astype(np.uint8)):
        if int(component.sum()) <= max_hole_area:
            filled[component.astype(bool)] = 1
    return filled


def remove_small_objects(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Remove connected components (8-connectivity) smaller than min_size pixels."""
    labels, num_labels = _label_components(mask, connectivity=8)
    if num_labels == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = np.bincount(labels.ravel(), minlength=num_labels + 1)
    keep = sizes >= min_size
    keep[0] = False  # background never kept
    return keep[labels].astype(np.uint8)


def morphological_refinement(
    mask: np.ndarray,
    closing_kernel: int = 0,
    opening_kernel: int = 0,
    min_size: int = 200,
    guidance_map: np.ndarray | None = None,
    guidance_threshold: float = 0.40,
    max_hole_area: int = 0,
) -> np.ndarray:
    """Full refinement pipeline per pipeline.md Stage 6.

    Returns [H, W] uint8 binary mask.
    """
    if closing_kernel > 1:
        mask = binary_closing(mask, kernel_size=closing_kernel)
    if opening_kernel > 1:
        mask = binary_opening(mask, kernel_size=opening_kernel)
    mask = fill_holes(mask, max_hole_area=max_hole_area)
    mask = remove_small_objects(mask, min_size=min_size)

    if guidance_map is not None and mask.any():
        filtered = np.zeros_like(mask, dtype=np.uint8)
        for component in _component_masks(mask):
            region = component.astype(bool)
            if float(guidance_map[region].mean()) >= guidance_threshold:
                filtered |= component
        if filtered.any():
            mask = filtered
    return mask


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected binary component masks, in label order."""
    labels, num_labels = _label_components(mask, connectivity=8)
    return [(labels == label_id).astype(np.uint8) for label_id in range(1, num_labels + 1)]
