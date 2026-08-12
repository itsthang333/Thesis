from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Tile:
    image_id: str
    scale: int
    box: tuple[int, int, int, int]
    pixels: np.ndarray


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    source: str
    source_view: str
    native_box: tuple[int, int, int, int]
    positive_points: tuple[tuple[int, int], ...]
    negative_points: tuple[tuple[int, int], ...]
    score: float
    component_mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateMask:
    candidate_id: str
    mask: np.ndarray
    proposal_id: str
    proposal_source: str
    sam_backend: str
    prompt_type: str
    predicted_iou: float
    stability: float
    roi_scale: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    image_id: str
    candidate_id: str
    mask: np.ndarray
    probability: float
    bag_probability: float
    uncertainty: float
    evidence: dict[str, float]
