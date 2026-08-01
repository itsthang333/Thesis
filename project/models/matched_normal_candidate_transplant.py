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


DENSENET_DIAGNOSTIC_STAGES = (
    "pool0",
    "transition1",
    "transition2",
    "transition3",
    "norm5",
)


@dataclass(frozen=True)
class NormalReferencePair:
    recipient_image_id: str
    recipient_group_id: str
    sham_image_id: str
    sham_group_id: str


def _text(row: Mapping[str, object], key: str) -> str:
    return str(row.get(key, "")).strip().lower()


def _identity(row: Mapping[str, object], key: str) -> str:
    """Return canonical identifiers without changing filesystem-significant case."""

    return str(row.get(key, "")).strip()


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
    query_group = _identity(query, "group_id")
    query_id = _identity(query, "image_id")
    eligible: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for row in normal_rows:
        image_id = _identity(row, "image_id")
        group_id = _identity(row, "group_id")
        if str(row.get("tumor", "")) not in {"0", "0.0", "false", "False"}:
            raise ValueError("normal_rows contain a tumor-positive record")
        if not image_id or not group_id:
            raise ValueError("Reference metadata omit image_id/group_id")
        image_key = image_id.casefold()
        group_key = group_id.casefold()
        if image_key == query_id.casefold() or group_key == query_group.casefold():
            continue
        if image_key in seen_ids:
            raise ValueError("Duplicate normal image_id in reference metadata")
        seen_ids.add(image_key)
        eligible.append(row)
    ordered = sorted(eligible, key=lambda row: _reference_key(query, row))
    selected: list[Mapping[str, object]] = []
    selected_groups: set[str] = set()
    for row in ordered:
        group_id = _identity(row, "group_id")
        group_key = group_id.casefold()
        if group_key in selected_groups:
            continue
        selected.append(row)
        selected_groups.add(group_key)
        if len(selected) == 2 * pair_count:
            break
    if len(selected) != 2 * pair_count:
        raise ValueError("Insufficient group-distinct normal references")
    return [
        NormalReferencePair(
            recipient_image_id=_identity(selected[2 * index], "image_id"),
            recipient_group_id=_identity(selected[2 * index], "group_id"),
            sham_image_id=_identity(selected[2 * index + 1], "image_id"),
            sham_group_id=_identity(selected[2 * index + 1], "group_id"),
        )
        for index in range(pair_count)
    ]


def select_random_normal_reference_pairs(
    query: Mapping[str, object],
    normal_rows: Sequence[Mapping[str, object]],
    *,
    pair_count: int = 2,
    seed: int = 20260802,
) -> list[NormalReferencePair]:
    """Deterministic metadata-blind donor control with identical exclusions."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    query_id = _identity(query, "image_id")
    query_group = _identity(query, "group_id")
    eligible: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for row in normal_rows:
        image_id = _identity(row, "image_id")
        group_id = _identity(row, "group_id")
        if str(row.get("tumor", "")) not in {"0", "0.0", "false", "False"}:
            raise ValueError("normal_rows contain a tumor-positive record")
        if not image_id or not group_id:
            raise ValueError("Reference metadata omit image_id/group_id")
        image_key = image_id.casefold()
        group_key = group_id.casefold()
        if image_key == query_id.casefold() or group_key == query_group.casefold():
            continue
        if image_key in seen_ids:
            raise ValueError("Duplicate normal image_id in reference metadata")
        seen_ids.add(image_key)
        eligible.append(row)
    ordered = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{seed}|{query_id.casefold()}|{_identity(row, 'image_id').casefold()}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[Mapping[str, object]] = []
    selected_groups: set[str] = set()
    for row in ordered:
        group_id = _identity(row, "group_id")
        group_key = group_id.casefold()
        if group_key in selected_groups:
            continue
        selected.append(row)
        selected_groups.add(group_key)
        if len(selected) == 2 * pair_count:
            break
    if len(selected) != 2 * pair_count:
        raise ValueError("Insufficient group-distinct random normal references")
    return [
        NormalReferencePair(
            recipient_image_id=_identity(selected[2 * index], "image_id"),
            recipient_group_id=_identity(selected[2 * index], "group_id"),
            sham_image_id=_identity(selected[2 * index + 1], "image_id"),
            sham_group_id=_identity(selected[2 * index + 1], "group_id"),
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
    return build_transplants_from_matched_contents(
        source_matched,
        recipient,
        sham_matched,
        candidate_masks,
        feather_kernel=feather_kernel,
    )


def build_transplants_from_matched_contents(
    source_matched: torch.Tensor,
    recipient: torch.Tensor,
    sham_matched: torch.Tensor,
    candidate_masks: torch.Tensor,
    *,
    feather_kernel: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste already intensity-matched contents without repeating quantiles."""

    _validate_image(source_matched, "source_matched")
    _validate_image(recipient, "recipient")
    _validate_image(sham_matched, "sham_matched")
    if source_matched.shape != recipient.shape or source_matched.shape != sham_matched.shape:
        raise ValueError("all transplant images must share shape")
    if candidate_masks.shape[-2:] != source_matched.shape[-2:]:
        raise ValueError("candidate-mask grid differs from transplant image")
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


def _weighted_spatial_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if values.ndim != 3 or weights.shape != values.shape:
        raise ValueError("values and weights must both have shape [N,H,W]")
    numerator = (values * weights).sum(dim=(-2, -1))
    denominator = weights.sum(dim=(-2, -1))
    return torch.where(
        denominator > eps,
        numerator / denominator.clamp_min(eps),
        torch.zeros_like(numerator),
    )


def _stage_mask_and_ring(
    candidate_masks: torch.Tensor,
    spatial_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = F.interpolate(
        candidate_masks[:, None].float(),
        size=spatial_shape,
        mode="area",
    )[:, 0].clamp(0.0, 1.0)
    dilated = F.max_pool2d(mask[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    ring = (dilated - mask).clamp(0.0, 1.0)
    return mask, ring


def _dense_feature_modules(
    classifier: torch.nn.Module,
    stage_names: Sequence[str],
) -> tuple[torch.nn.Module, dict[str, torch.nn.Module]]:
    base = classifier.module if hasattr(classifier, "module") else classifier
    features = getattr(base, "features", None)
    if features is None or not hasattr(features, "named_children"):
        raise ValueError("classifier does not expose DenseNet .features")
    available = dict(features.named_children())
    missing = [name for name in stage_names if name not in available]
    if missing:
        raise ValueError(f"DenseNet diagnostic stages are missing: {missing}")
    head = getattr(base, "classifier", None)
    if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
        raise ValueError("layerwise diagnostic requires a one-logit linear head")
    return base, {name: available[name] for name in stage_names}


def paired_dense_layer_diagnostics(
    classifier: torch.nn.Module,
    positive: torch.Tensor,
    sham: torch.Tensor,
    candidate_masks: torch.Tensor,
    *,
    batch_size: int = 8,
    stage_names: Sequence[str] = DENSENET_DIAGNOSTIC_STAGES,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> dict[str, torch.Tensor | tuple[str, ...]]:
    """Decompose a positive-vs-sham intervention through DenseNet stages.

    All returned tensors are compact per-candidate statistics.  No feature
    tensor is retained after its batch.  The same candidate mask is projected
    to every feature grid, and the final signed class-response decomposition is
    checked against the exact classifier-logit difference.
    """

    if classifier.training:
        raise ValueError("layerwise diagnostic requires classifier.eval()")
    if positive.shape != sham.shape or positive.ndim != 4:
        raise ValueError("positive and sham must share shape [N,C,H,W]")
    if len(positive) != len(candidate_masks):
        raise ValueError("candidate count differs from transplant batches")
    if candidate_masks.shape[-2:] != positive.shape[-2:]:
        raise ValueError("candidate-mask grid differs from transplant images")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stage_names = tuple(stage_names)
    if not stage_names:
        raise ValueError("stage_names must be nonempty")
    base, stage_modules = _dense_feature_modules(classifier, stage_names)
    class_weight = base.classifier.weight[0].detach().float()
    channels = positive.shape[1]
    norm_mean = torch.tensor(
        imagenet_mean[:channels], device=positive.device, dtype=positive.dtype
    )[None, :, None, None]
    norm_std = torch.tensor(
        imagenet_std[:channels], device=positive.device, dtype=positive.dtype
    )[None, :, None, None]

    metric_names = (
        "feature_l2_inside",
        "feature_l2_ring",
        "feature_l2_contrast",
        "relative_feature_l2_inside",
        "relative_feature_l2_ring",
        "relative_feature_l2_contrast",
        "cosine_inside",
        "cosine_ring",
        "delta_energy_inside_fraction",
        "mask_mass",
        "ring_mass",
    )
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in metric_names}
    logit_delta_chunks: list[torch.Tensor] = []
    class_response_chunks: dict[str, list[torch.Tensor]] = {
        "class_response_delta_inside": [],
        "class_response_delta_ring": [],
        "class_response_delta_contrast": [],
        "class_response_delta_global": [],
        "class_response_logit_residual": [],
    }

    for start in range(0, len(positive), batch_size):
        stop = min(start + batch_size, len(positive))
        pair_count = stop - start
        captures: dict[str, torch.Tensor] = {}
        handles = []
        for name, module in stage_modules.items():
            def hook(_module, _inputs, output, *, _name=name):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"DenseNet stage {_name} did not return a tensor")
                captures[_name] = output.float()

            handles.append(module.register_forward_hook(hook))
        paired = torch.cat((positive[start:stop], sham[start:stop]), dim=0)
        normalized = (paired - norm_mean) / norm_std
        try:
            with torch.no_grad():
                logits = classifier(normalized)
        finally:
            for handle in handles:
                handle.remove()
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.shape != (2 * pair_count,):
            raise ValueError("classifier must return [B] or [B,1] logits")
        logit_delta = logits[:pair_count].float() - logits[pair_count:].float()
        logit_delta_chunks.append(logit_delta)
        mask_chunk = candidate_masks[start:stop]

        stage_metric_chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in metric_names
        }
        final_positive: torch.Tensor | None = None
        final_sham: torch.Tensor | None = None
        final_mask: torch.Tensor | None = None
        final_ring: torch.Tensor | None = None
        for stage_name in stage_names:
            feature = captures.get(stage_name)
            if feature is None or feature.shape[0] != 2 * pair_count:
                raise RuntimeError(f"invalid capture for DenseNet stage {stage_name}")
            pos_feature = feature[:pair_count]
            sham_feature = feature[pair_count:]
            if stage_name == "norm5":
                pos_feature = torch.relu(pos_feature)
                sham_feature = torch.relu(sham_feature)
            mask, ring = _stage_mask_and_ring(mask_chunk, pos_feature.shape[-2:])
            difference = pos_feature - sham_feature
            feature_l2 = difference.square().mean(dim=1).clamp_min(0.0).sqrt()
            feature_scale = 0.5 * (
                pos_feature.square().mean(dim=1).clamp_min(0.0).sqrt()
                + sham_feature.square().mean(dim=1).clamp_min(0.0).sqrt()
            )
            relative_l2 = feature_l2 / feature_scale.clamp_min(1.0e-6)
            cosine = F.cosine_similarity(pos_feature, sham_feature, dim=1, eps=1.0e-6)
            inside_l2 = _weighted_spatial_mean(feature_l2, mask)
            ring_l2 = _weighted_spatial_mean(feature_l2, ring)
            inside_relative = _weighted_spatial_mean(relative_l2, mask)
            ring_relative = _weighted_spatial_mean(relative_l2, ring)
            energy_fraction = (feature_l2 * mask).sum(dim=(-2, -1)) / feature_l2.sum(
                dim=(-2, -1)
            ).clamp_min(1.0e-8)
            values = {
                "feature_l2_inside": inside_l2,
                "feature_l2_ring": ring_l2,
                "feature_l2_contrast": inside_l2 - ring_l2,
                "relative_feature_l2_inside": inside_relative,
                "relative_feature_l2_ring": ring_relative,
                "relative_feature_l2_contrast": inside_relative - ring_relative,
                "cosine_inside": _weighted_spatial_mean(cosine, mask),
                "cosine_ring": _weighted_spatial_mean(cosine, ring),
                "delta_energy_inside_fraction": energy_fraction,
                "mask_mass": mask.mean(dim=(-2, -1)),
                "ring_mass": ring.mean(dim=(-2, -1)),
            }
            for name in metric_names:
                stage_metric_chunks[name].append(values[name])
            if stage_name == "norm5":
                final_positive = pos_feature
                final_sham = sham_feature
                final_mask = mask
                final_ring = ring
        for name in metric_names:
            chunks[name].append(torch.stack(stage_metric_chunks[name], dim=1).cpu())

        if final_positive is None or final_sham is None or final_mask is None or final_ring is None:
            raise RuntimeError("norm5 must be included in diagnostic stages")
        if final_positive.shape[1] != class_weight.numel():
            raise ValueError("final DenseNet feature dimension differs from classifier head")
        class_delta_map = torch.einsum(
            "c,nchw->nhw", class_weight.to(final_positive.device), final_positive - final_sham
        )
        class_inside = _weighted_spatial_mean(class_delta_map, final_mask)
        class_ring = _weighted_spatial_mean(class_delta_map, final_ring)
        class_global = class_delta_map.mean(dim=(-2, -1))
        class_values = {
            "class_response_delta_inside": class_inside,
            "class_response_delta_ring": class_ring,
            "class_response_delta_contrast": class_inside - class_ring,
            "class_response_delta_global": class_global,
            "class_response_logit_residual": logit_delta - class_global,
        }
        for name, value in class_values.items():
            class_response_chunks[name].append(value.cpu())

    result: dict[str, torch.Tensor | tuple[str, ...]] = {
        "stage_names": stage_names,
        "classifier_logit_delta": torch.cat(logit_delta_chunks).cpu(),
    }
    for name, values in chunks.items():
        result[name] = torch.cat(values, dim=0)
    for name, values in class_response_chunks.items():
        result[name] = torch.cat(values, dim=0)
    return result


def matched_transplant_layerwise_scores(
    classifier: torch.nn.Module,
    source: torch.Tensor,
    candidate_masks: torch.Tensor,
    reference_images: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int = 8,
    candidate_chunk_size: int = 8,
    feather_kernel: int = 7,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> dict[str, torch.Tensor | tuple[str, ...]]:
    """Aggregate compact layerwise causal diagnostics across recipients."""

    if not reference_images:
        raise ValueError("reference_images must be nonempty")
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    recipient_results: list[dict[str, torch.Tensor | tuple[str, ...]]] = []
    for recipient, sham_donor in reference_images:
        source_matched = robust_affine_match(source, recipient)
        sham_matched = robust_affine_match(sham_donor, recipient)
        chunk_results: list[dict[str, torch.Tensor | tuple[str, ...]]] = []
        for start in range(0, len(candidate_masks), candidate_chunk_size):
            masks = candidate_masks[start : start + candidate_chunk_size]
            positive, sham = build_transplants_from_matched_contents(
                source_matched,
                recipient,
                sham_matched,
                masks,
                feather_kernel=feather_kernel,
            )
            chunk_results.append(
                paired_dense_layer_diagnostics(
                    classifier,
                    positive,
                    sham,
                    masks,
                    batch_size=batch_size,
                    imagenet_mean=imagenet_mean,
                    imagenet_std=imagenet_std,
                )
            )
        stage_names = chunk_results[0]["stage_names"]
        combined: dict[str, torch.Tensor | tuple[str, ...]] = {"stage_names": stage_names}
        for key in chunk_results[0]:
            if key == "stage_names":
                continue
            values = [item[key] for item in chunk_results]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"non-tensor layerwise value for {key}")
            combined[key] = torch.cat(values, dim=0)  # type: ignore[arg-type]
        recipient_results.append(combined)
    stage_names = recipient_results[0]["stage_names"]
    if not isinstance(stage_names, tuple):
        raise TypeError("invalid stage_names payload")
    output: dict[str, torch.Tensor | tuple[str, ...]] = {"stage_names": stage_names}
    tensor_keys = [key for key in recipient_results[0] if key != "stage_names"]
    for key in tensor_keys:
        values = [item[key] for item in recipient_results]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"non-tensor layerwise value for {key}")
        stack = torch.stack(values)  # type: ignore[arg-type]
        output[key + "_mean"] = stack.mean(dim=0)
        output[key + "_recipient_std"] = stack.std(dim=0, unbiased=False)
    output["score"] = output["classifier_logit_delta_mean"]
    output["recipient_std"] = output["classifier_logit_delta_recipient_std"]
    return output


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
    "DENSENET_DIAGNOSTIC_STAGES",
    "NormalReferencePair",
    "build_matched_transplants",
    "build_transplants_from_matched_contents",
    "feather_candidate_masks",
    "frozen_selector_panel",
    "matched_transplant_scores",
    "matched_transplant_layerwise_scores",
    "paired_dense_layer_diagnostics",
    "percentile_ranks",
    "reference_manifest_rows",
    "robust_affine_match",
    "select_normal_reference_pairs",
    "select_random_normal_reference_pairs",
]
