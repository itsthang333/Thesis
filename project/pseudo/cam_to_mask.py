from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


def normalize_min_max(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = array.astype(np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    return (array - minimum) / (maximum - minimum + eps)


def aggregate_cam_heatmaps(cams: list[np.ndarray], weights: list[float] | np.ndarray | None = None, mode: str = "weighted_mean") -> np.ndarray:
    if not cams:
        raise ValueError("aggregate_cam_heatmaps requires at least one CAM.")

    stacked = np.stack([normalize_min_max(cam) for cam in cams], axis=0)
    if mode == "max":
        return stacked.max(axis=0)

    if weights is None:
        weights_array = np.ones(stacked.shape[0], dtype=np.float32)
    else:
        weights_array = np.asarray(weights, dtype=np.float32)
        if weights_array.shape[0] != stacked.shape[0]:
            raise ValueError("weights must match the number of CAMs.")

    weights_array = np.clip(weights_array, 0.0, None)
    if float(weights_array.sum()) == 0.0:
        weights_array = np.ones_like(weights_array)
    weights_array = weights_array / weights_array.sum()
    aggregated = np.tensordot(weights_array, stacked, axes=(0, 0))
    return normalize_min_max(aggregated)


def _binary_morphology(mask: np.ndarray, kernel_size: int, operation: str) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    padding = kernel_size // 2
    if operation == "dilate":
        result = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=padding)
    elif operation == "erode":
        result = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=kernel_size, stride=1, padding=padding)
    else:
        raise ValueError(f"Unsupported morphology operation: {operation}")
    return (result[0, 0].numpy() > 0.5).astype(np.uint8)


def binary_opening(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return _binary_morphology(_binary_morphology(mask, kernel_size, "erode"), kernel_size, "dilate")


def binary_closing(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return _binary_morphology(_binary_morphology(mask, kernel_size, "dilate"), kernel_size, "erode")


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for row in range(height):
        for col in (0, width - 1):
            if mask[row, col] == 0 and not visited[row, col]:
                visited[row, col] = True
                queue.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if mask[row, col] == 0 and not visited[row, col]:
                visited[row, col] = True
                queue.append((row, col))

    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in neighbor_offsets:
            next_row = row + delta_row
            next_col = col + delta_col
            if 0 <= next_row < height and 0 <= next_col < width and not visited[next_row, next_col] and mask[next_row, next_col] == 0:
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

    holes = (mask == 0) & (~visited)
    filled = mask.copy()
    filled[holes] = 1
    return filled


def _component_filter(mask: np.ndarray, min_area: int, keep_largest: bool = True) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for row in range(height):
        for col in range(width):
            if mask[row, col] == 0 or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for delta_row, delta_col in neighbor_offsets:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and mask[next_row, next_col] > 0
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            components.append(component)

    if not components:
        return np.zeros_like(mask, dtype=np.uint8)

    if keep_largest:
        components = [max(components, key=len)]

    output = np.zeros_like(mask, dtype=np.uint8)
    for component in components:
        if len(component) < min_area:
            continue
        for row, col in component:
            output[row, col] = 1
    return output


def cam_to_pseudo_mask(
    cam: np.ndarray,
    percentile: float = 80.0,
    min_area: int = 200,
    kernel_size: int = 5,
    keep_largest_component: bool = True,
) -> np.ndarray:
    cam = normalize_min_max(cam)
    threshold = np.percentile(cam, percentile)
    binary = (cam >= threshold).astype(np.uint8)
    binary = binary_closing(binary, kernel_size=kernel_size)
    binary = binary_opening(binary, kernel_size=kernel_size)
    binary = fill_holes(binary)
    binary = _component_filter(binary, min_area=min_area, keep_largest=keep_largest_component)
    return binary.astype(np.uint8)
