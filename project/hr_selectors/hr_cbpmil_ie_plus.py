from __future__ import annotations

"""Deterministic, annotation-free IE+ selector over one frozen candidate gallery."""

from dataclasses import dataclass

import numpy as np
import torch


def _validate_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != (320, 320):
        raise ValueError("Candidate masks must have shape [N,320,320]")
    if not len(masks) or np.any(masks.reshape(len(masks), -1).sum(axis=1) == 0):
        raise ValueError("Every candidate mask must be non-empty")
    return masks


def pairwise_intersection(masks: np.ndarray, *, block_size: int = 32) -> np.ndarray:
    """Exact intersections using bounded-memory float64 GEMM (exact for binary 320 masks)."""

    masks = _validate_masks(masks)
    flat = torch.from_numpy(masks.reshape(len(masks), -1).astype(np.float64, copy=False))
    output = np.empty((len(masks), len(masks)), dtype=np.int64)
    for start in range(0, len(masks), block_size):
        stop = min(start + block_size, len(masks))
        output[start:stop] = torch.matmul(flat[start:stop], flat.T).numpy().astype(np.int64)
    return output


def duplicate_cluster_ids(masks: np.ndarray, *, threshold: float = 0.90) -> np.ndarray:
    masks = _validate_masks(masks)
    areas = masks.reshape(len(masks), -1).sum(axis=1, dtype=np.int64)
    parent = np.arange(len(masks), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    # IoU >= t implies min(area)/max(area) >= t.  This necessary filter removes
    # most pairs before an exact native-resolution intersection is evaluated.
    for left in range(len(masks)):
        ratios = np.minimum(areas[left], areas[left + 1 :]) / np.maximum(
            np.maximum(areas[left], areas[left + 1 :]), 1
        )
        for offset in np.flatnonzero(ratios >= threshold).tolist():
            right = left + 1 + offset
            intersection = int(np.count_nonzero(masks[left] & masks[right]))
            union_area = int(areas[left] + areas[right] - intersection)
            if intersection / max(union_area, 1) >= threshold:
                union(left, right)
    root_to_cluster: dict[int, int] = {}
    result = np.empty(len(masks), dtype=np.int32)
    for index in range(len(masks)):
        root = find(index)
        result[index] = root_to_cluster.setdefault(root, len(root_to_cluster))
    return result


def relation_matrices(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU, containment and same-location matrices at native 320 geometry."""

    masks = _validate_masks(masks)
    intersections = pairwise_intersection(masks)
    areas = masks.reshape(len(masks), -1).sum(axis=1, dtype=np.int64)
    iou = intersections / np.maximum(areas[:, None] + areas[None, :] - intersections, 1)
    containment = intersections / np.maximum(np.minimum(areas[:, None], areas[None, :]), 1)
    return iou, containment, (iou >= 0.15) | (containment >= 0.65)


def _adaptive_ring(mask: np.ndarray) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    area = int(mask.sum())
    width = int(np.clip(np.round(0.15 * np.sqrt(area / np.pi)), 2, 8))
    dilated = torch.nn.functional.max_pool2d(
        mask_tensor, kernel_size=2 * width + 1, stride=1, padding=width
    )[0, 0].numpy() > 0
    ring = dilated & ~mask
    if not ring.any():
        ring = ~mask
    if not ring.any():
        ring = np.ones_like(mask, dtype=bool)
    return ring


@dataclass(frozen=True)
class SelectionResult:
    selected_index: int
    selected_cluster: int
    top3_clusters: tuple[int, ...]
    cluster_identity: np.ndarray
    candidate_extent: np.ndarray
    family_score: np.ndarray


def select_ie_plus(
    masks: np.ndarray,
    classification_logits: np.ndarray,
    detection_logits: np.ndarray,
    dense_logits: np.ndarray,
    cluster_ids: np.ndarray,
) -> SelectionResult:
    masks = _validate_masks(masks)
    a = np.asarray(classification_logits, dtype=np.float64).reshape(-1)
    b = np.asarray(detection_logits, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids, dtype=np.int32).reshape(-1)
    evidence_logits = np.asarray(dense_logits, dtype=np.float64)
    if len(a) != len(masks) or len(b) != len(masks) or len(clusters) != len(masks):
        raise ValueError("Candidate score arrays do not align")
    if evidence_logits.shape != (160, 160) or not np.isfinite(evidence_logits).all():
        raise ValueError("dense_logits must be one finite 160x160 map")
    if np.any(clusters < 0):
        raise ValueError("cluster_ids must be non-negative")

    q = 1.0 / (1.0 + np.exp(-np.clip(a, -60.0, 60.0)))
    unique = np.unique(clusters)
    balanced = []
    members_by_cluster: dict[int, np.ndarray] = {}
    within = np.zeros(len(masks), dtype=np.float64)
    for cluster in unique.tolist():
        members = np.flatnonzero(clusters == cluster)
        members_by_cluster[cluster] = members
        local = b[members]
        maximum = float(local.max())
        balanced.append(maximum + np.log(np.exp(local - maximum).sum()) - np.log(len(members)))
        shifted = local - maximum
        within[members] = np.exp(shifted) / np.exp(shifted).sum()
    balanced_array = np.asarray(balanced)
    balanced_array -= balanced_array.max()
    pi_values = np.exp(balanced_array) / np.exp(balanced_array).sum()
    detection = np.zeros(len(masks), dtype=np.float64)
    identity = np.zeros(int(unique.max()) + 1, dtype=np.float64)
    representatives: dict[int, int] = {}
    for cluster, pi in zip(unique.tolist(), pi_values.tolist(), strict=True):
        members = members_by_cluster[cluster]
        detection[members] = pi * within[members]
        identity[cluster] = pi * float(np.sum(within[members] * q[members]))
        representatives[cluster] = int(members[np.argmax(q[members] * detection[members])])

    _, _, same_location = relation_matrices(masks)
    ranked_clusters = sorted(unique.tolist(), key=lambda c: (-identity[c], c))
    top3: list[int] = []
    for cluster in ranked_clusters:
        representative = representatives[cluster]
        if all(not same_location[representative, representatives[kept]] for kept in top3):
            top3.append(cluster)
        if len(top3) == 3:
            break

    evidence = 1.0 / (1.0 + np.exp(-np.clip(evidence_logits, -60.0, 60.0)))
    fractional = torch.nn.functional.avg_pool2d(
        torch.from_numpy(masks.astype(np.float32))[:, None], kernel_size=2, stride=2
    )[:, 0].numpy()
    survival = fractional > 0
    extent = np.full(len(masks), -np.inf, dtype=np.float64)
    family_score = np.full(len(masks), -np.inf, dtype=np.float64)
    winners: list[tuple[float, float, int, int]] = []
    for cluster in top3:
        representative = representatives[cluster]
        family = np.flatnonzero(same_location[representative])
        union = np.max(fractional[family], axis=0)
        union_evidence = float((evidence * union).sum())
        for index in family.tolist():
            capture = float((evidence * fractional[index]).sum()) / max(union_evidence, 1.0e-12)
            ring = _adaptive_ring(survival[index])
            inside_mean = float((evidence * fractional[index]).sum()) / max(
                float(fractional[index].sum()), 1.0e-12
            )
            ring_mean = float(evidence[ring].mean())
            purity = inside_mean / max(inside_mean + ring_mean, 1.0e-12)
            extent[index] = 2.0 * capture * purity / max(capture + purity, 1.0e-12)
        best_extent = float(np.max(extent[family]))
        tied = family[np.isclose(extent[family], best_extent, rtol=0.0, atol=1.0e-12)]
        best = int(tied[np.argmax(q[tied] * detection[tied])])
        if len(tied) > 1:
            best_identity = float(np.max(q[tied] * detection[tied]))
            tied = tied[np.isclose(q[tied] * detection[tied], best_identity, rtol=0.0, atol=1.0e-12)]
            best = int(np.min(tied))
        score = float(identity[cluster] * extent[best])
        family_score[best] = score
        winners.append((score, float(identity[cluster]), -best, cluster))
    if not winners:
        raise RuntimeError("IE+ produced no eligible cluster")
    _score, _identity, neg_index, selected_cluster = max(winners)
    return SelectionResult(
        selected_index=-int(neg_index),
        selected_cluster=int(selected_cluster),
        top3_clusters=tuple(int(value) for value in top3),
        cluster_identity=identity,
        candidate_extent=extent,
        family_score=family_score,
    )
