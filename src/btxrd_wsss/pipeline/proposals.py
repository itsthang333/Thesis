from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from btxrd_wsss.config import ProposalConfig
from btxrd_wsss.types import Proposal

SOURCES = ("hrnet_full", "hrnet_tile", "biomedclip")


@dataclass
class ComponentNode:
    mask: np.ndarray
    threshold: float
    quality: float
    peak: tuple[int, int]


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _points(
    evidence: np.ndarray, component: np.ndarray, count: int, high: bool
) -> tuple[tuple[int, int], ...]:
    coordinates = np.argwhere(component)
    order = np.argsort(evidence[component], kind="stable")
    if high:
        order = order[::-1]
    selected: list[tuple[int, int]] = []
    distance = max(2.0, np.sqrt(component.sum()) / (count + 1))
    for index in order:
        y, x = coordinates[index]
        if all((x - px) ** 2 + (y - py) ** 2 >= distance**2 for px, py in selected):
            selected.append((int(x), int(y)))
            if len(selected) == count:
                break
    return tuple(selected)


def _negative_ring(
    evidence: np.ndarray, component: np.ndarray, count: int
) -> tuple[tuple[int, int], ...]:
    radius = max(3, round(np.sqrt(component.sum()) * 0.15))
    ring = ndimage.binary_dilation(component, iterations=radius) & ~component
    return _points(evidence, ring, count, high=False) if ring.any() else ()


def _box(component: np.ndarray, padding: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(component)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = max(2, round((x1 - x0) * padding)), max(2, round((y1 - y0) * padding))
    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(component.shape[1], x1 + px),
        min(component.shape[0], y1 + py),
    )


def _component_quality(evidence: np.ndarray, component: np.ndarray) -> float:
    inside = float(evidence[component].mean())
    radius = max(2, round(np.sqrt(component.sum()) * 0.1))
    ring = ndimage.binary_dilation(component, iterations=radius) & ~component
    outside = float(evidence[ring].mean()) if ring.any() else 0.0
    peak = float(evidence[component].max())
    return 0.55 * inside + 0.30 * max(0.0, inside - outside) + 0.15 * peak


def _same_branch(first: ComponentNode, second: ComponentNode) -> bool:
    if first.peak == second.peak:
        return True
    intersection = np.logical_and(first.mask, second.mask).sum()
    containment = intersection / max(1, min(first.mask.sum(), second.mask.sum()))
    return containment >= 0.70


def component_tree(
    evidence: np.ndarray,
    thresholds: list[float],
    *,
    minimum_area: int,
    per_threshold: int,
) -> list[tuple[ComponentNode, int, tuple[float, ...]]]:
    """Collapse nested threshold components into stable branches."""
    nodes: list[ComponentNode] = []
    for threshold in sorted(set(thresholds)):
        labels, count = ndimage.label(evidence >= threshold, np.ones((3, 3), np.uint8))
        local: list[ComponentNode] = []
        for index in range(1, count + 1):
            mask = labels == index
            if int(mask.sum()) < minimum_area:
                continue
            peak_y, peak_x = np.unravel_index(
                np.argmax(np.where(mask, evidence, -np.inf)), evidence.shape
            )
            local.append(
                ComponentNode(
                    mask,
                    float(threshold),
                    _component_quality(evidence, mask),
                    (int(peak_x), int(peak_y)),
                )
            )
        nodes.extend(sorted(local, key=lambda item: item.quality, reverse=True)[:per_threshold])
    branches: list[list[ComponentNode]] = []
    for node in sorted(nodes, key=lambda item: (-item.threshold, -item.quality)):
        branch = next(
            (group for group in branches if any(_same_branch(node, old) for old in group)), None
        )
        if branch is None:
            branches.append([node])
        else:
            branch.append(node)
    result: list[tuple[ComponentNode, int, tuple[float, ...]]] = []
    for branch in branches:
        # Higher thresholds preserve compact lesions; quality breaks ties.
        representative = max(
            branch, key=lambda item: (item.quality, item.threshold, -item.mask.sum())
        )
        stability = len({item.threshold for item in branch})
        result.append(
            (representative, stability, tuple(sorted({item.threshold for item in branch})))
        )
    result.sort(key=lambda item: item[0].quality + 0.05 * item[1], reverse=True)
    return result


class ProposalGenerator:
    def __init__(self, config: ProposalConfig) -> None:
        self.config = config

    def from_map(
        self,
        evidence: np.ndarray,
        *,
        image_id: str,
        source: str,
        source_view: str,
        thresholds: list[float] | None = None,
        source_confidence: float = 1.0,
    ) -> list[Proposal]:
        if source not in SOURCES:
            raise ValueError(f"Unknown proposal source: {source}")
        evidence = np.asarray(evidence, dtype=np.float32)
        if evidence.ndim != 2 or not np.isfinite(evidence).all():
            raise ValueError("Proposal evidence must be a finite 2D map")
        thresholds = (
            thresholds
            or {
                "hrnet_full": self.config.hrnet_full_percentiles,
                "hrnet_tile": self.config.hrnet_tile_percentiles,
                "biomedclip": self.config.biomedclip_percentiles,
            }[source]
        )
        branches = component_tree(
            evidence,
            thresholds,
            minimum_area=self.config.minimum_native_area,
            per_threshold=self.config.max_components_per_threshold,
        )
        proposals: list[Proposal] = []
        for branch_index, (node, stability, branch_thresholds) in enumerate(
            branches[: self.config.source_quotas[source]]
        ):
            peak_x, peak_y = node.peak
            padding = min(self.config.box_padding)
            point_count = max(self.config.positive_point_counts)
            proposals.append(
                Proposal(
                    proposal_id=_stable_id(image_id, source, source_view, branch_index, node.peak),
                    source=source,
                    source_view=source_view,
                    native_box=_box(node.mask, padding),
                    positive_points=_points(evidence, node.mask, point_count, high=True),
                    negative_points=_negative_ring(
                        evidence, node.mask, self.config.negative_points
                    ),
                    score=float(node.quality + 0.05 * stability),
                    component_mask=node.mask,
                    metadata={
                        "threshold": node.threshold,
                        "branch_thresholds": branch_thresholds,
                        "threshold_stability": stability,
                        "padding": padding,
                        "peak_x": peak_x,
                        "peak_y": peak_y,
                        "source_confidence": float(np.clip(source_confidence, 0, 1)),
                    },
                )
            )
        return proposals

    def generate_all(
        self, maps: dict[str, np.ndarray], *, image_id: str, confidences: dict[str, float]
    ) -> list[Proposal]:
        if set(maps) != set(SOURCES) or set(confidences) != set(SOURCES):
            raise ValueError(f"Expected maps/confidences for {SOURCES}")
        return [
            proposal
            for source in SOURCES
            for proposal in self.from_map(
                maps[source],
                image_id=image_id,
                source=source,
                source_view=source,
                source_confidence=confidences[source],
            )
        ]
