from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Protocol

import numpy as np

from btxrd_wsss.config import SAMConfig, SelectionConfig
from btxrd_wsss.pipeline.selection import percentile_ranks, score_and_gate
from btxrd_wsss.types import CandidateMask, Proposal


class SAMBackend(Protocol):
    name: str

    def predict_roi(
        self,
        image: np.ndarray,
        proposal: Proposal,
        *,
        roi_scale: float,
        multimask: bool,
    ) -> list[tuple[np.ndarray, float, float]]: ...


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(np.logical_and(first, second).sum() / union)


def expanded_roi(
    box: tuple[int, int, int, int], shape: tuple[int, int], scale: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width, height = max(2, (x1 - x0) * scale), max(2, (y1 - y0) * scale)
    return (
        max(0, round(cx - width / 2)),
        max(0, round(cy - height / 2)),
        min(shape[1], round(cx + width / 2)),
        min(shape[0], round(cy + height / 2)),
    )


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    elif values.ndim == 3 and values.shape[0] == 3:
        values = np.moveaxis(values, 0, -1)
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 1)
        values = np.round(values * 255).astype(np.uint8)
    return values


def _stability(logits: np.ndarray, threshold: float = 0.0, offset: float = 1.0) -> float:
    high, low = logits > threshold + offset, logits > threshold - offset
    denominator = low.sum()
    return 1.0 if denominator == 0 else float(high.sum() / denominator)


class SAMViTBROIBackend:
    name = "sam_vit_b_roi"

    def __init__(self, checkpoint: str, device: str = "cuda", model_type: str = "vit_b") -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError("Install the 'sam' extra to run SAM") from exc
        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.predictor = SamPredictor(model)

    def predict_roi(
        self, image: np.ndarray, proposal: Proposal, *, roi_scale: float, multimask: bool
    ) -> list[tuple[np.ndarray, float, float]]:
        native_shape = (
            image.shape[-2:] if image.ndim == 2 or image.shape[0] == 3 else image.shape[:2]
        )
        rx0, ry0, rx1, ry1 = expanded_roi(proposal.native_box, native_shape, roi_scale)
        rgb = _as_rgb_u8(image)
        crop = rgb[ry0:ry1, rx0:rx1]
        self.predictor.set_image(crop)
        x0, y0, x1, y1 = proposal.native_box
        box = np.asarray([x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0], np.float32)
        positive = np.asarray([(x - rx0, y - ry0) for x, y in proposal.positive_points], np.float32)
        negative = np.asarray([(x - rx0, y - ry0) for x, y in proposal.negative_points], np.float32)
        if len(positive):
            positive = positive[
                (positive[:, 0] >= 0)
                & (positive[:, 0] < crop.shape[1])
                & (positive[:, 1] >= 0)
                & (positive[:, 1] < crop.shape[0])
            ]
        if len(negative):
            negative = negative[
                (negative[:, 0] >= 0)
                & (negative[:, 0] < crop.shape[1])
                & (negative[:, 1] >= 0)
                & (negative[:, 1] < crop.shape[0])
            ]
        points = np.concatenate([positive, negative]) if len(negative) else positive
        labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))]).astype(np.int32)
        logits, scores, _ = self.predictor.predict(
            point_coords=points if len(points) else None,
            point_labels=labels if len(points) else None,
            box=box,
            multimask_output=multimask,
            return_logits=True,
        )
        results: list[tuple[np.ndarray, float, float]] = []
        threshold = float(self.predictor.model.mask_threshold)
        for local_logits, score in zip(logits, scores, strict=True):
            local_mask = local_logits > threshold
            native = np.zeros(native_shape, dtype=bool)
            native[ry0:ry1, rx0:rx1] = local_mask
            results.append((native, float(score), _stability(local_logits, threshold)))
        return results


class SAMMed2DROIBackend(SAMViTBROIBackend):
    """Official SAM-Med2D ViT-B adapter, isolated from the original SAM namespace."""

    name = "sam_med2d_vit_b_roi"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        model_type: str = "vit_b",
        *,
        image_size: int = 256,
        encoder_adapter: bool = True,
    ) -> None:
        from btxrd_wsss.vendor.sam_med2d import SamPredictor, sam_model_registry

        arguments = SimpleNamespace(
            image_size=image_size,
            sam_checkpoint=checkpoint,
            encoder_adapter=encoder_adapter,
        )
        model = sam_model_registry[model_type](arguments)
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.predictor = SamPredictor(model)


def create_sam_backend(config: SAMConfig, device: str) -> SAMBackend:
    if config.backend == "sam_med2d":
        return SAMMed2DROIBackend(
            config.checkpoint,
            device,
            config.model_type,
            image_size=config.image_size,
            encoder_adapter=config.encoder_adapter,
        )
    if config.backend == "sam_vit_b":
        return SAMViTBROIBackend(config.checkpoint, device, config.model_type)
    raise ValueError(f"Unsupported SAM backend: {config.backend}")


def _candidate_id(proposal: Proposal, backend: str, roi_scale: float, index: int) -> str:
    value = f"{proposal.proposal_id}|{backend}|{roi_scale}|{index}"
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def _append_predictions(
    gallery: list[CandidateMask],
    predictions: list[tuple[np.ndarray, float, float]],
    proposal: Proposal,
    backend_name: str,
    roi_scale: float,
    duplicate_iou: float,
) -> None:
    for index, (mask, predicted_iou, stability) in enumerate(predictions):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any() or any(mask_iou(mask, old.mask) >= duplicate_iou for old in gallery):
            continue
        gallery.append(
            CandidateMask(
                candidate_id=_candidate_id(proposal, backend_name, roi_scale, index),
                mask=mask,
                proposal_id=proposal.proposal_id,
                proposal_source=proposal.source,
                sam_backend=backend_name,
                prompt_type="box+positive+negative",
                predicted_iou=float(predicted_iou),
                stability=float(stability),
                roi_scale=float(roi_scale),
                metadata={
                    "source_score": proposal.score,
                    "source_component": proposal.component_mask,
                    **proposal.metadata,
                },
            )
        )


def build_adaptive_gallery(
    image: np.ndarray,
    proposals: list[Proposal],
    backend: SAMBackend,
    *,
    config: SAMConfig,
) -> list[CandidateMask]:
    """One cheap mask per component, then multimask only for uncertain/novel ROIs."""
    gallery: list[CandidateMask] = []
    for proposal in sorted(proposals, key=lambda item: item.score, reverse=True):
        predictions = backend.predict_roi(
            image,
            proposal,
            roi_scale=config.initial_roi_scale,
            multimask=False,
        )
        _append_predictions(
            gallery,
            predictions,
            proposal,
            backend.name,
            config.initial_roi_scale,
            config.duplicate_iou,
        )
        if len(gallery) >= config.maximum_raw_candidates:
            break
    initial_by_proposal = {item.proposal_id: item for item in gallery}
    initial_quality = [
        float(np.clip((item.predicted_iou + item.stability) / 2, 0, 1)) for item in gallery
    ]
    quality_ranks = percentile_ranks(initial_quality)
    quality_by_candidate = {
        item.candidate_id: float(rank)
        for item, rank in zip(gallery, quality_ranks, strict=True)
    }
    expansion: list[tuple[float, Proposal]] = []
    for proposal in proposals:
        candidate = initial_by_proposal.get(proposal.proposal_id)
        if candidate is None:
            priority = 10 + proposal.score
        else:
            component = proposal.component_mask
            coverage = np.logical_and(candidate.mask, component).sum() / max(1, component.sum())
            quality = quality_by_candidate[candidate.candidate_id]
            small_bonus = float(component.sum() / component.size <= config.small_area_ratio)
            alternatives = [old for old in gallery if old.proposal_id != candidate.proposal_id]
            novelty = 1 - max(
                (mask_iou(candidate.mask, old.mask) for old in alternatives), default=0
            )
            priority = (
                proposal.score
                + 0.30 * (1 - quality)
                + 0.20 * (1 - coverage)
                + 0.15 * small_bonus
                + 0.10 * novelty
            )
        expansion.append((priority, proposal))
    for source, quota in config.expansion_roi_quotas.items():
        selected = sorted(
            (item for item in expansion if item[1].source == source),
            reverse=True,
            key=lambda item: item[0],
        )[:quota]
        for _, proposal in selected:
            predictions = backend.predict_roi(
                image,
                proposal,
                roi_scale=config.expansion_roi_scale,
                multimask=config.multimask,
            )
            _append_predictions(
                gallery,
                predictions,
                proposal,
                backend.name,
                config.expansion_roi_scale,
                config.duplicate_iou,
            )
            if len(gallery) >= config.maximum_raw_candidates:
                return gallery[: config.maximum_raw_candidates]
    return gallery


def _area_bucket(candidate: CandidateMask, config: SAMConfig) -> str:
    ratio = candidate.mask.sum() / candidate.mask.size
    if ratio < config.tiny_area_ratio:
        return "tiny"
    if ratio < config.small_area_ratio:
        return "small"
    return "large"


def select_diverse_gallery(
    candidates: list[CandidateMask],
    source_maps: dict[str, np.ndarray],
    *,
    sam_config: SAMConfig,
    selection_config: SelectionConfig,
) -> list[CandidateMask]:
    """Source/size-aware greedy diversity selection before expensive RAD-DINO."""
    candidates = score_and_gate(candidates, source_maps, selection_config, sam_config)
    selected: list[CandidateMask] = []
    selected_ids: set[str] = set()

    def utility(candidate: CandidateMask) -> float:
        novelty = 1 - max((mask_iou(candidate.mask, old.mask) for old in selected), default=0)
        return float(candidate.metadata["upstream_score"]) + sam_config.diversity_weight * novelty

    def fill(pool: list[CandidateMask], target_count: int) -> None:
        pool_ids = {item.candidate_id for item in pool}
        while len(selected) < sam_config.maximum_selected_candidates:
            available = [item for item in pool if item.candidate_id not in selected_ids]
            if (
                not available
                or sum(item.candidate_id in pool_ids for item in selected) >= target_count
            ):
                return
            winner = max(available, key=utility)
            selected.append(winner)
            selected_ids.add(winner.candidate_id)

    fill(
        [item for item in candidates if _area_bucket(item, sam_config) == "tiny"],
        sam_config.minimum_tiny_candidates,
    )
    fill(
        [item for item in candidates if _area_bucket(item, sam_config) == "small"],
        sam_config.minimum_small_candidates,
    )
    for source, quota in sam_config.gallery_minimum_quotas.items():
        pool = [item for item in candidates if item.proposal_source == source]
        while (
            sum(item.proposal_source == source for item in selected) < quota
            and len(selected) < sam_config.maximum_selected_candidates
        ):
            available = [item for item in pool if item.candidate_id not in selected_ids]
            if not available:
                break
            winner = max(available, key=utility)
            selected.append(winner)
            selected_ids.add(winner.candidate_id)
    while len(selected) < min(sam_config.maximum_selected_candidates, len(candidates)):
        available = [item for item in candidates if item.candidate_id not in selected_ids]
        if not available:
            break
        winner = max(available, key=utility)
        selected.append(winner)
        selected_ids.add(winner.candidate_id)
    return selected
