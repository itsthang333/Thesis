from __future__ import annotations

"""Matched-normal candidate transplantation primitives.

The module contains no dataset or annotation reader.  It operates on frozen
image-label metadata, images and candidate masks supplied by a prediction-first
runner.  A positive transplant and its normal-to-normal sham always share the
same recipient and mask, so geometry and paste-boundary effects cancel in the
signed classifier-logit difference.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NormalReferencePair:
    recipient_image_id: str
    recipient_group_id: str
    sham_image_id: str
    sham_group_id: str


def _text(row: Mapping[str, object], key: str) -> str:
    return str(row.get(key, "")).strip().lower()


def _aspect_ratio(row: Mapping[str, object]) -> float:
    width = float(row.get("width", 0.0))
    height = float(row.get("height", 0.0))
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Reference metadata require positive width and height")
    return width / height


def _reference_key(
    query: Mapping[str, object],
    donor: Mapping[str, object],
) -> tuple[int, int, int, float, str]:
    query_id = _text(query, "image_id")
    donor_id = _text(donor, "image_id")
    tie = hashlib.sha256(f"{query_id}|{donor_id}".encode("utf-8")).hexdigest()
    return (
        int(_text(query, "anatomy") != _text(donor, "anatomy")),
        int(_text(query, "view") != _text(donor, "view")),
        int(_text(query, "center") != _text(donor, "center")),
        abs(math.log(_aspect_ratio(query)) - math.log(_aspect_ratio(donor))),
        tie,
    )


def select_normal_reference_pairs(
    query: Mapping[str, object],
    normal_rows: Sequence[Mapping[str, object]],
    *,
    pair_count: int = 2,
) -> list[NormalReferencePair]:
    """Choose deterministic, metadata-matched, group-disjoint normal pairs."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    query_group = _text(query, "group_id")
    query_id = _text(query, "image_id")
    eligible: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for row in normal_rows:
        image_id = _text(row, "image_id")
        group_id = _text(row, "group_id")
        if str(row.get("tumor", "")) not in {"0", "0.0", "false", "False"}:
            raise ValueError("normal_rows contain a tumor-positive record")
        if not image_id or not group_id:
            raise ValueError("Reference metadata omit image_id/group_id")
        if image_id == query_id or group_id == query_group:
            continue
        if image_id in seen_ids:
            raise ValueError("Duplicate normal image_id in reference metadata")
        seen_ids.add(image_id)
        eligible.append(row)
    ordered = sorted(eligible, key=lambda row: _reference_key(query, row))
    selected: list[Mapping[str, object]] = []
    selected_groups: set[str] = set()
    for row in ordered:
        group_id = _text(row, "group_id")
        if group_id in selected_groups:
            continue
        selected.append(row)
        selected_groups.add(group_id)
        if len(selected) == 2 * pair_count:
            break
    if len(selected) != 2 * pair_count:
        raise ValueError("Insufficient group-distinct normal references")
    return [
        NormalReferencePair(
            recipient_image_id=_text(selected[2 * index], "image_id"),
            recipient_group_id=_text(selected[2 * index], "group_id"),
            sham_image_id=_text(selected[2 * index + 1], "image_id"),
            sham_group_id=_text(selected[2 * index + 1], "group_id"),
        )
        for index in range(pair_count)
    ]


def reference_manifest_rows(
    queries: Sequence[Mapping[str, object]],
    normal_rows: Sequence[Mapping[str, object]],
    *,
    pair_count: int = 2,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query in queries:
        for pair_index, pair in enumerate(
            select_normal_reference_pairs(query, normal_rows, pair_count=pair_count)
        ):
            rows.append(
                {
                    "image_id": _text(query, "image_id"),
                    "group_id": _text(query, "group_id"),
                    "pair_index": pair_index,
                    **asdict(pair),
                }
            )
    return rows


def _validate_image(image: torch.Tensor, name: str) -> None:
    if image.ndim != 3 or image.shape[0] not in {1, 3}:
        raise ValueError(f"{name} must have shape [1|3,H,W]")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} must be finite")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError(f"{name} must be in [0,1]")


def robust_affine_match(
    source: torch.Tensor,
    recipient: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    recipient_valid: torch.Tensor | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> torch.Tensor:
    """Match source intensity location/scale to recipient on valid content."""

    _validate_image(source, "source")
    _validate_image(recipient, "recipient")
    if source.shape != recipient.shape:
        raise ValueError("source and recipient shapes differ")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("invalid robust quantiles")
    height, width = source.shape[-2:]
    if source_valid is None:
        source_valid = torch.ones((height, width), dtype=torch.bool, device=source.device)
    if recipient_valid is None:
        recipient_valid = torch.ones(
            (height, width), dtype=torch.bool, device=recipient.device
        )
    if source_valid.shape != (height, width) or recipient_valid.shape != (height, width):
        raise ValueError("valid-content masks must match image shape")
    if not bool(source_valid.any()) or not bool(recipient_valid.any()):
        raise ValueError("valid-content masks must be nonempty")
    output = torch.empty_like(source)
    for channel in range(source.shape[0]):
        source_values = source[channel][source_valid]
        recipient_values = recipient[channel][recipient_valid]
        source_quantiles = torch.quantile(
            source_values.float(),
            torch.tensor(
                [lower_quantile, 0.5, upper_quantile],
                device=source.device,
            ),
        )
        recipient_quantiles = torch.quantile(
            recipient_values.float(),
            torch.tensor(
                [lower_quantile, 0.5, upper_quantile],
                device=recipient.device,
            ),
        )
        source_span = (source_quantiles[2] - source_quantiles[0]).clamp_min(1.0e-4)
        recipient_span = (recipient_quantiles[2] - recipient_quantiles[0]).clamp_min(
            1.0e-4
        )
        output[channel] = (
            (source[channel] - source_quantiles[1])
            * (recipient_span / source_span)
            + recipient_quantiles[1]
        )
    return output.clamp(0.0, 1.0)


def feather_candidate_masks(
    candidate_masks: torch.Tensor,
    *,
    kernel_size: int = 7,
) -> torch.Tensor:
    if candidate_masks.ndim != 3:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be positive and odd")
    if not torch.isfinite(candidate_masks).all():
        raise ValueError("candidate masks must be finite")
    if bool((candidate_masks < 0.0).any()) or bool((candidate_masks > 1.0).any()):
        raise ValueError("candidate masks must be in [0,1]")
    return F.avg_pool2d(
        candidate_masks[:, None].float(),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )[:, 0].clamp(0.0, 1.0)


def build_matched_transplants(
    source: torch.Tensor,
    recipient: torch.Tensor,
    sham_donor: torch.Tensor,
    candidate_masks: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    recipient_valid: torch.Tensor | None = None,
    sham_valid: torch.Tensor | None = None,
    feather_kernel: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive and sham batches sharing every geometric operation."""

    _validate_image(source, "source")
    _validate_image(recipient, "recipient")
    _validate_image(sham_donor, "sham_donor")
    if source.shape != recipient.shape or source.shape != sham_donor.shape:
        raise ValueError("all transplant images must share shape")
    if candidate_masks.shape[-2:] != source.shape[-2:]:
        raise ValueError("candidate-mask grid differs from transplant image")
    source_matched = robust_affine_match(
        source,
        recipient,
        source_valid=source_valid,
        recipient_valid=recipient_valid,
    )
    sham_matched = robust_affine_match(
        sham_donor,
        recipient,
        source_valid=sham_valid,
        recipient_valid=recipient_valid,
    )
    alpha = feather_candidate_masks(candidate_masks, kernel_size=feather_kernel)[:, None]
    recipient_batch = recipient[None].expand(len(candidate_masks), -1, -1, -1)
    positive = recipient_batch * (1.0 - alpha) + source_matched[None] * alpha
    sham = recipient_batch * (1.0 - alpha) + sham_matched[None] * alpha
    return positive, sham


def _classifier_logits(
    classifier: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    *,
    batch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    channels = images.shape[1]
    norm_mean = torch.tensor(mean[:channels], device=images.device)[None, :, None, None]
    norm_std = torch.tensor(std[:channels], device=images.device)[None, :, None, None]
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            output = classifier((images[start : start + batch_size] - norm_mean) / norm_std)
            if output.ndim == 2 and output.shape[1] == 1:
                output = output[:, 0]
            if output.ndim != 1:
                raise ValueError("classifier must return [B] or [B,1] logits")
            logits.append(output.float())
    return torch.cat(logits)


def matched_transplant_scores(
    classifier: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    candidate_masks: torch.Tensor,
    reference_images: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int = 32,
    feather_kernel: int = 7,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> dict[str, torch.Tensor]:
    if not reference_images:
        raise ValueError("reference_images must be nonempty")
    deltas: list[torch.Tensor] = []
    positive_logits: list[torch.Tensor] = []
    sham_logits: list[torch.Tensor] = []
    for recipient, sham_donor in reference_images:
        positive, sham = build_matched_transplants(
            source,
            recipient,
            sham_donor,
            candidate_masks,
            feather_kernel=feather_kernel,
        )
        paired = _classifier_logits(
            classifier,
            torch.cat((positive, sham), dim=0),
            batch_size=batch_size,
            mean=imagenet_mean,
            std=imagenet_std,
        )
        candidate_count = len(candidate_masks)
        pos = paired[:candidate_count]
        neg = paired[candidate_count:]
        positive_logits.append(pos)
        sham_logits.append(neg)
        deltas.append(pos - neg)
    delta_stack = torch.stack(deltas)
    return {
        "score": delta_stack.mean(dim=0),
        "recipient_std": delta_stack.std(dim=0, unbiased=False),
        "positive_logit_mean": torch.stack(positive_logits).mean(dim=0),
        "sham_logit_mean": torch.stack(sham_logits).mean(dim=0),
        "positive_recipient_fraction": (delta_stack > 0.0).float().mean(dim=0),
    }


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be one finite nonempty vector")
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1)
        result[order[start:stop]] = rank / max(len(values) - 1, 1)
        start = stop
    return result


def frozen_selector_panel(
    baseline_scores: np.ndarray,
    transplant_scores: np.ndarray,
    random_control_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline = percentile_ranks(baseline_scores)
    transplant = percentile_ranks(transplant_scores)
    random_control = percentile_ranks(random_control_scores)
    return {
        "g1_upstream_baseline": baseline,
        "transplant_only": transplant,
        "baseline_transplant_equal": 0.5 * baseline + 0.5 * transplant,
        "baseline_transplant_three_to_one": 0.75 * baseline + 0.25 * transplant,
        "baseline_random_control_three_to_one": 0.75 * baseline
        + 0.25 * random_control,
    }


__all__ = [
    "NormalReferencePair",
    "build_matched_transplants",
    "feather_candidate_masks",
    "frozen_selector_panel",
    "matched_transplant_scores",
    "percentile_ranks",
    "reference_manifest_rows",
    "robust_affine_match",
    "select_normal_reference_pairs",
]
