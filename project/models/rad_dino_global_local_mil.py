"""Global-local MIL primitives for image-label-only RAD-DINO WSSS.

The frozen global decoder and frozen RAD-DINO encoder live in the runner.  This
module contains the trainable local decoder plus deterministic proposal,
pooling, stitching, and fusion operations.  It deliberately has no dataset or
segmentation-annotation dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.rad_dino_multilayer_soft_region_decoder import (
    MultiLayerSoftRegionConfig,
    RadDinoMultiLayerSoftRegionDecoder,
)


@dataclass(frozen=True)
class GlobalLocalMILConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    affinity_dim: int = 64
    decoder_scale: int = 2
    train_patches: int = 6
    inference_patches: int = 3
    proposal_size: int = 160
    proposal_stride: int = 8
    proposal_iou_limit: float = 0.25
    top_fraction: float = 0.01
    negative_dense_weight: float = 0.5
    positive_sparsity_weight: float = 0.05
    flip_consistency_weight: float = 0.2
    local_keep_fraction: float = 0.02
    residual_weight: float = 0.35
    confidence_temperature: float = 0.10

    def validate(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.affinity_dim <= 0:
            raise ValueError("Decoder dimensions must be positive")
        if self.decoder_scale < 1:
            raise ValueError("decoder_scale must be positive")
        if not 0 < self.inference_patches <= self.train_patches:
            raise ValueError("Patch counts must satisfy 0 < inference <= train")
        if self.proposal_size <= 0 or self.proposal_stride <= 0:
            raise ValueError("Proposal geometry must be positive")
        if not 0 <= self.proposal_iou_limit < 1:
            raise ValueError("proposal_iou_limit must be in [0, 1)")
        for name in ("top_fraction", "local_keep_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "negative_dense_weight",
            "positive_sparsity_weight",
            "flip_consistency_weight",
            "residual_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.confidence_temperature <= 0:
            raise ValueError("confidence_temperature must be positive")


class RadDinoGlobalLocalMILDecoder(nn.Module):
    """Apply one shared spatial decoder to every proposal in a MIL bag."""

    def __init__(self, config: GlobalLocalMILConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        base_config = MultiLayerSoftRegionConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            affinity_dim=config.affinity_dim,
            decoder_scale=config.decoder_scale,
        )
        self.spatial_decoder = RadDinoMultiLayerSoftRegionDecoder(base_config)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits/features shaped ``[B,K,C,H,W]``."""

        tokens = torch.as_tensor(patch_tokens)
        guide = torch.as_tensor(guidance)
        if tokens.ndim != 6:
            raise ValueError("patch_tokens must have shape [B,K,L,H,W,D]")
        if guide.ndim != 5:
            raise ValueError("guidance must have shape [B,K,C,H,W]")
        batch, patches = tokens.shape[:2]
        if guide.shape[:2] != (batch, patches):
            raise ValueError("Token and guidance bag dimensions differ")
        flat_tokens = tokens.flatten(0, 1)
        flat_guidance = guide.flatten(0, 1)
        logits, features, _layer_weights = self.spatial_decoder(
            flat_tokens,
            flat_guidance,
        )
        return (
            logits.unflatten(0, (batch, patches)),
            features.unflatten(0, (batch, patches)),
        )


def top_fraction_pool(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Pool the highest valid proposal pixels into one logit per image."""

    values = torch.as_tensor(logits)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    if values.ndim != 5 or values.shape[2] != 1:
        raise ValueError("logits must have shape [B,K,1,H,W]")
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.shape != values.shape:
        raise ValueError("valid must match logits, with optional singleton channel")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    pooled: list[torch.Tensor] = []
    for image_logits, image_valid in zip(values, mask):
        selected = image_logits[image_valid]
        if selected.numel() == 0:
            raise ValueError("Every MIL bag must contain valid pixels")
        count = max(1, int(np.ceil(float(selected.numel()) * fraction)))
        pooled.append(selected.topk(count, sorted=False).values.mean())
    return torch.stack(pooled)


def local_mil_loss(
    logits: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_fraction: float,
    negative_dense_weight: float,
    positive_sparsity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Image BCE plus normal dense suppression and positive-map sparsity."""

    values = torch.as_tensor(logits)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    targets = torch.as_tensor(labels, dtype=values.dtype, device=values.device)
    if targets.shape != (values.shape[0],):
        raise ValueError("labels must have shape [B]")
    bag_logits = top_fraction_pool(values, mask, fraction=top_fraction)
    image_bce = F.binary_cross_entropy_with_logits(bag_logits, targets)
    zero = values.sum() * 0.0
    negative_terms: list[torch.Tensor] = []
    positive_terms: list[torch.Tensor] = []
    for index, target in enumerate(targets):
        selected = values[index][mask[index]]
        if float(target.detach()) < 0.5:
            negative_terms.append(F.softplus(selected).mean())
        else:
            positive_terms.append(torch.sigmoid(selected).mean())
    negative_dense = (
        torch.stack(negative_terms).mean() if negative_terms else zero
    )
    positive_sparsity = (
        torch.stack(positive_terms).mean() if positive_terms else zero
    )
    total = (
        image_bce
        + negative_dense_weight * negative_dense
        + positive_sparsity_weight * positive_sparsity
    )
    return total, {
        "image_bce": image_bce,
        "negative_dense": negative_dense,
        "positive_sparsity": positive_sparsity,
        "bag_logits": bag_logits,
    }


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (
        by1 - by0
    ) - intersection
    return float(intersection) / max(float(union), 1.0)


@lru_cache(maxsize=32)
def _window_geometry(
    height: int,
    width: int,
    window_size: int,
    stride: int,
    iou_limit: float,
) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    tuple[int, ...],
]:
    """Cache candidate boxes and pairwise-compatible bitsets."""

    xs = list(range(0, width - window_size + 1, stride))
    ys = list(range(0, height - window_size + 1, stride))
    if xs[-1] != width - window_size:
        xs.append(width - window_size)
    if ys[-1] != height - window_size:
        ys.append(height - window_size)
    boxes = tuple(
        (x0, y0, x0 + window_size, y0 + window_size)
        for y0 in ys
        for x0 in xs
    )
    compatible: list[int] = []
    for index, box in enumerate(boxes):
        bits = 0
        for other_index, other in enumerate(boxes):
            if index != other_index and _box_iou(box, other) <= iou_limit:
                bits |= 1 << other_index
        compatible.append(bits)
    return boxes, tuple(compatible)


def _first_ranked_feasible_windows(
    boxes: tuple[tuple[int, int, int, int], ...],
    compatible: tuple[int, ...],
    order: list[int],
    *,
    count: int,
) -> list[tuple[int, int, int, int]]:
    """Keep each ranked candidate only when a full set remains feasible."""

    if len(order) != len(boxes) or sorted(order) != list(range(len(boxes))):
        raise ValueError("order must be a permutation of all candidate boxes")
    suffix_bits = [0] * (len(order) + 1)
    for rank in range(len(order) - 1, -1, -1):
        suffix_bits[rank] = suffix_bits[rank + 1] | (1 << order[rank])

    @lru_cache(maxsize=None)
    def search(
        start_rank: int,
        available_bits: int,
        needed: int,
    ) -> tuple[int, ...] | None:
        if needed == 0:
            return ()
        if bin(available_bits).count("1") < needed:
            return None
        for rank in range(start_rank, len(order)):
            candidate = order[rank]
            candidate_bit = 1 << candidate
            if not available_bits & candidate_bit:
                continue
            remaining = (
                available_bits
                & compatible[candidate]
                & suffix_bits[rank + 1]
            )
            if bin(remaining).count("1") < needed - 1:
                continue
            completion = search(rank + 1, remaining, needed - 1)
            if completion is not None:
                return (candidate, *completion)
        return None

    selected = search(0, suffix_bits[0], count)
    if selected is None:
        raise RuntimeError("Proposal constraints cannot supply requested windows")
    return [boxes[index] for index in selected]


def greedy_saliency_windows(
    saliency: np.ndarray,
    *,
    window_size: int,
    count: int,
    stride: int,
    iou_limit: float,
) -> list[tuple[int, int, int, int]]:
    """Select high-mass, spatially diverse square proposal windows."""

    values = np.asarray(saliency, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("saliency must be a finite 2-D array")
    height, width = values.shape
    if not 0 < window_size <= min(height, width):
        raise ValueError("window_size must fit inside saliency")
    if count <= 0 or stride <= 0 or not 0 <= iou_limit < 1:
        raise ValueError("Invalid proposal selection parameters")
    integral = np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    boxes, compatible = _window_geometry(
        height,
        width,
        window_size,
        stride,
        iou_limit,
    )
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for box in boxes:
        x0, y0, x1, y1 = box
        mass = (
            integral[y1, x1]
            - integral[y0, x1]
            - integral[y1, x0]
            + integral[y0, x0]
        )
        candidates.append((float(mass), box))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    index_by_box = {box: index for index, box in enumerate(boxes)}
    order = [index_by_box[box] for _mass, box in candidates]
    return _first_ranked_feasible_windows(
        boxes,
        compatible,
        order,
        count=count,
    )


def random_diverse_windows(
    *,
    output_shape: tuple[int, int],
    window_size: int,
    count: int,
    stride: int,
    iou_limit: float,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    """Draw deterministic diverse windows for image-label-negative bags."""

    height, width = output_shape
    if not 0 < window_size <= min(height, width):
        raise ValueError("window_size must fit inside output geometry")
    if count <= 0 or stride <= 0 or not 0 <= iou_limit < 1:
        raise ValueError("Invalid random proposal parameters")
    boxes, compatible = _window_geometry(
        height,
        width,
        window_size,
        stride,
        iou_limit,
    )
    order = list(range(len(boxes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return _first_ranked_feasible_windows(
        boxes,
        compatible,
        order,
        count=count,
    )


def stitch_local_maps(
    patch_maps: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize proposal maps into content geometry and average overlaps."""

    values = np.asarray(patch_maps, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(boxes):
        raise ValueError("patch_maps must have shape [K,H,W] matching boxes")
    height, width = output_shape
    total = torch.zeros((1, 1, height, width), dtype=torch.float32)
    weight = torch.zeros_like(total)
    for patch, (x0, y0, x1, y1) in zip(values, boxes):
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Proposal box lies outside output geometry")
        resized = F.interpolate(
            torch.from_numpy(patch)[None, None],
            size=(y1 - y0, x1 - x0),
            mode="bilinear",
            align_corners=False,
        )
        total[:, :, y0:y1, x0:x1] += resized
        weight[:, :, y0:y1, x0:x1] += 1.0
    covered = weight > 0
    stitched = torch.where(covered, total / weight.clamp_min(1.0), total)
    return stitched[0, 0].numpy(), covered[0, 0].numpy()


def confidence_gated_rank_fusion(
    global_map: np.ndarray,
    local_map: np.ndarray,
    local_coverage: np.ndarray,
    *,
    local_confidence: float,
    normal_confidence_p99: float,
    keep_fraction: float,
    residual_weight: float,
    temperature: float,
) -> tuple[np.ndarray, float]:
    """Promote only confident local peaks while retaining the global map."""

    global_values = np.asarray(global_map, dtype=np.float32)
    local_values = np.asarray(local_map, dtype=np.float32)
    coverage = np.asarray(local_coverage, dtype=bool)
    if (
        global_values.shape != local_values.shape
        or coverage.shape != global_values.shape
        or global_values.ndim != 2
    ):
        raise ValueError("Fusion inputs must share one 2-D shape")
    if not np.isfinite(global_values).all() or not np.isfinite(local_values).all():
        raise ValueError("Fusion maps must be finite")
    if not 0 < keep_fraction <= 1 or residual_weight < 0 or temperature <= 0:
        raise ValueError("Invalid fusion hyperparameters")
    gate = float(
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    (local_confidence - normal_confidence_p99) / temperature,
                    -60.0,
                    60.0,
                )
            )
        )
    )
    selected_values = local_values[coverage]
    if selected_values.size == 0:
        return global_values.copy(), gate
    threshold = float(np.percentile(selected_values, 100.0 * (1.0 - keep_fraction)))
    peaks = coverage & (local_values >= threshold)
    residual = np.zeros_like(local_values)
    residual[peaks] = local_values[peaks]
    fused = global_values + residual_weight * gate * residual * (
        1.0 - global_values
    )
    return np.clip(fused, 0.0, 1.0), gate


__all__ = [
    "GlobalLocalMILConfig",
    "RadDinoGlobalLocalMILDecoder",
    "confidence_gated_rank_fusion",
    "greedy_saliency_windows",
    "local_mil_loss",
    "random_diverse_windows",
    "stitch_local_maps",
    "top_fraction_pool",
]
