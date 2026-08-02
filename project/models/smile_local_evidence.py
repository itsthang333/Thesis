from __future__ import annotations

"""SMILE local evidence and immutable-rich-gallery candidate readout.

SMILE is a binary plus subtype-conditioned, image-label-only local model.  It
does not consume proposal masks, source IDs, coordinates or spatial labels
during representation training.  Proposal masks are used only after the dense
maps have been frozen, to read candidate identity and extent compatibility.
"""

import math
from pathlib import Path
import re
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SMILE_METHOD = "subtype_matched_normal_intra_class_local_evidence"
SMILE_SCHEMA_VERSION = 1
SMILE_CLASS_COUNT = 10  # normal plus nine BTXRD tumor types
SMILE_POOL_FRACTIONS = (0.0005, 0.001, 0.0025)
SMILE_LOSS_WEIGHTS = {
    "binary_bag": 1.0,
    "subtype_bag": 0.50,
    "normal_binary_dense": 0.20,
    "normal_subtype_dense": 0.10,
    "foreground_subtype": 0.15,
    "background_normal": 0.05,
    "binary_subtype_alignment": 0.05,
}
SMILE_RESIDUAL_WEIGHTS = {
    "identity_only": (0.25, 0.0),
    "identity_extent": (0.15, 0.10),
}


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise AssertionError("unreachable")


class _Refine(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels),
            nn.ReLU(inplace=True),
        )


def normalize_feature_map(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[1] <= 0:
        raise ValueError("feature must be BCHW")
    _require_finite("feature", feature)
    return F.normalize(feature, dim=1, eps=1e-6)


def pool_reference_tokens(
    references: torch.Tensor,
    reference_valid: torch.Tensor,
    *,
    factor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validity-weighted BRCHW reference pooling."""

    if references.ndim != 5 or reference_valid.ndim != 4:
        raise ValueError("references/valid must be BRCHW/BRHW")
    if (
        references.shape[:2] != reference_valid.shape[:2]
        or references.shape[-2:] != reference_valid.shape[-2:]
        or factor <= 0
    ):
        raise ValueError("reference shapes or pooling factor are invalid")
    batch, count, channels, height, width = references.shape
    if height % factor or width % factor:
        raise ValueError("reference grid must be divisible by factor")
    flat = references.reshape(batch * count, channels, height, width)
    valid = reference_valid.reshape(batch * count, 1, height, width).to(flat)
    numerator = F.avg_pool2d(flat * valid, factor, factor)
    denominator = F.avg_pool2d(valid, factor, factor)
    pooled = numerator / denominator.clamp_min(1e-6)
    return (
        pooled.reshape(batch, count, channels, height // factor, width // factor),
        (denominator[:, 0] > 0).reshape(
            batch, count, height // factor, width // factor
        ),
    )


def matched_normal_counterparts(
    query: torch.Tensor,
    references: torch.Tensor,
    query_valid: torch.Tensor,
    reference_valid: torch.Tensor,
    *,
    temperature: float = 0.07,
    query_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-match each query cell to the unordered normal-reference tokens."""

    if query.ndim != 4 or references.ndim != 5:
        raise ValueError("query/references must be BCHW/BRCHW")
    batch, channels, height, width = query.shape
    if (
        references.shape[0] != batch
        or references.shape[2] != channels
        or query_valid.shape != (batch, height, width)
        or reference_valid.shape[:2] != references.shape[:2]
        or reference_valid.shape[-2:] != references.shape[-2:]
    ):
        raise ValueError("query/reference shapes are incompatible")
    if temperature <= 0 or query_chunk_size <= 0:
        raise ValueError("temperature/chunk size must be positive")
    if not reference_valid.reshape(batch, -1).any(dim=1).all():
        raise ValueError("every query requires a valid reference token")

    query_tokens = normalize_feature_map(query).flatten(2).transpose(1, 2)
    reference_tokens = F.normalize(references, dim=2, eps=1e-6).permute(
        0, 1, 3, 4, 2
    ).reshape(batch, -1, channels)
    reference_mask = reference_valid.reshape(batch, -1)
    query_mask = query_valid.reshape(batch, -1)
    matched_chunks: list[torch.Tensor] = []
    cosine_chunks: list[torch.Tensor] = []
    for start in range(0, query_tokens.shape[1], query_chunk_size):
        current = query_tokens[:, start : start + query_chunk_size]
        cosine = torch.bmm(current, reference_tokens.transpose(1, 2))
        attention = torch.softmax(
            (cosine / temperature).masked_fill(~reference_mask[:, None], -torch.inf),
            dim=-1,
        )
        counterpart = F.normalize(
            torch.bmm(attention, reference_tokens), dim=-1, eps=1e-6
        )
        matched_chunks.append(counterpart)
        cosine_chunks.append((attention * cosine).sum(dim=-1, keepdim=True))
    matched = torch.cat(matched_chunks, dim=1) * query_mask[..., None]
    expected_cosine = torch.cat(cosine_chunks, dim=1) * query_mask[..., None]
    matched_map = matched.transpose(1, 2).reshape(batch, channels, height, width)
    cosine_map = expected_cosine.transpose(1, 2).reshape(batch, 1, height, width)
    _require_finite("matched counterpart", matched_map)
    _require_finite("matched cosine", cosine_map)
    return matched_map, cosine_map


def local_evidence_representation(
    query: torch.Tensor,
    counterpart: torch.Tensor,
    cosine: torch.Tensor,
) -> torch.Tensor:
    if counterpart.shape != query.shape or cosine.shape != (
        query.shape[0], 1, *query.shape[-2:]
    ):
        raise ValueError("local representation inputs differ")
    query_norm = normalize_feature_map(query)
    counterpart_norm = normalize_feature_map(counterpart)
    difference = query_norm - counterpart_norm
    return torch.cat(
        (query_norm, counterpart_norm, difference, difference.abs(), cosine), dim=1
    )


def query_only_control_representation(query: torch.Tensor) -> torch.Tensor:
    query_norm = normalize_feature_map(query)
    zeros = torch.zeros_like(query_norm)
    return torch.cat(
        (
            query_norm,
            zeros,
            zeros,
            zeros,
            torch.zeros(
                query.shape[0], 1, *query.shape[-2:], device=query.device, dtype=query.dtype
            ),
        ),
        dim=1,
    )


def masked_sparse_pool(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    fractions: tuple[float, ...] = SMILE_POOL_FRACTIONS,
) -> torch.Tensor:
    """Pool BCHW logits independently for each class over fixed sparse scales."""

    if logits.ndim != 4 or valid.shape != (
        logits.shape[0], *logits.shape[-2:]
    ):
        raise ValueError("logits/valid shapes are incompatible")
    if not fractions or tuple(sorted(set(fractions))) != fractions:
        raise ValueError("fractions must be unique and increasing")
    outputs: list[torch.Tensor] = []
    for sample, mask in zip(logits, valid, strict=True):
        selected = sample[:, mask]
        if selected.shape[1] == 0:
            raise ValueError("sample has no valid cells")
        scale_values = []
        for fraction in fractions:
            count = max(1, math.ceil(fraction * selected.shape[1]))
            scale_values.append(torch.topk(selected, count, dim=1).values.mean(dim=1))
        outputs.append(torch.stack(scale_values).mean(dim=0))
    return torch.stack(outputs)


def target_subtype_margin(
    subtype_logits: torch.Tensor,
    subtype: torch.Tensor,
) -> torch.Tensor:
    """Target subtype logit minus normal logit, without feeding the label in."""

    if subtype_logits.ndim != 4 or subtype_logits.shape[1] != SMILE_CLASS_COUNT:
        raise ValueError("subtype logits must be B10HW")
    labels = subtype.long().reshape(-1)
    if labels.shape[0] != subtype_logits.shape[0] or bool(
        ((labels < 0) | (labels >= SMILE_CLASS_COUNT)).any()
    ):
        raise ValueError("invalid subtype labels")
    selected = subtype_logits.gather(
        1, labels[:, None, None, None].expand(-1, 1, *subtype_logits.shape[-2:])
    )
    return selected - subtype_logits[:, :1]


def soft_intra_class_discrimination(
    subtype_logits: torch.Tensor,
    valid: torch.Tensor,
    tumor: torch.Tensor,
    subtype: torch.Tensor,
    *,
    temperature: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softly separate a tumor subtype foreground from normal background.

    No detached winner is formed.  Spatially smoothed subtype margins induce
    differentiable foreground/background posteriors across the entire bag.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    margin = target_subtype_margin(subtype_logits, subtype)[:, 0]
    smooth = F.avg_pool2d(margin[:, None], 5, stride=1, padding=2)[:, 0]
    foreground_losses: list[torch.Tensor] = []
    background_losses: list[torch.Tensor] = []
    for local, smoothed, mask, is_tumor, label in zip(
        subtype_logits,
        smooth,
        valid,
        tumor.reshape(-1),
        subtype.reshape(-1),
        strict=True,
    ):
        if float(is_tumor.detach()) < 0.5:
            continue
        if int(label.detach()) <= 0:
            raise ValueError("tumor sample must carry a non-normal subtype")
        flat_scores = smoothed[mask]
        flat_logits = local[:, mask].transpose(0, 1)
        if flat_scores.numel() == 0:
            raise ValueError("tumor bag has no valid cells")
        foreground_weight = torch.softmax(flat_scores / temperature, dim=0)
        background_weight = torch.softmax(-flat_scores / temperature, dim=0)
        foreground_target = torch.full(
            (flat_scores.numel(),), int(label.detach()), device=local.device, dtype=torch.long
        )
        background_target = torch.zeros_like(foreground_target)
        foreground_losses.append(
            (F.cross_entropy(flat_logits, foreground_target, reduction="none") * foreground_weight).sum()
        )
        background_losses.append(
            (F.cross_entropy(flat_logits, background_target, reduction="none") * background_weight).sum()
        )
    zero = subtype_logits.sum() * 0.0
    return (
        torch.stack(foreground_losses).mean() if foreground_losses else zero,
        torch.stack(background_losses).mean() if background_losses else zero,
    )


def smile_image_label_objective(
    output: dict[str, torch.Tensor],
    tumor: torch.Tensor,
    subtype: torch.Tensor,
    *,
    loss_weights: dict[str, float] = SMILE_LOSS_WEIGHTS,
) -> dict[str, torch.Tensor]:
    """Image-label-only binary/subtype/intra-class objective."""

    required = {
        "binary_image_logits",
        "subtype_image_logits",
        "binary_evidence_logits",
        "subtype_local_logits",
        "evidence_valid",
    }
    if not required.issubset(output):
        raise ValueError("SMILE output is incomplete")
    if set(loss_weights) != set(SMILE_LOSS_WEIGHTS):
        raise ValueError("loss weights differ from the frozen SMILE objective")
    labels = tumor.float().reshape(-1)
    subtypes = subtype.long().reshape(-1)
    if labels.shape != subtypes.shape or bool(((subtypes > 0) != (labels > 0.5)).any()):
        raise ValueError("binary and subtype labels are inconsistent")
    binary_image = output["binary_image_logits"].reshape_as(labels)
    subtype_image = output["subtype_image_logits"]
    binary_local = output["binary_evidence_logits"]
    subtype_local = output["subtype_local_logits"]
    valid = output["evidence_valid"][:, 0] > 0.5

    components: dict[str, torch.Tensor] = {
        "binary_bag": F.binary_cross_entropy_with_logits(binary_image.float(), labels),
        "subtype_bag": F.cross_entropy(subtype_image.float(), subtypes),
    }
    normal = labels < 0.5
    zero = binary_local.sum() * 0.0
    if bool(normal.any()):
        normal_valid = valid[normal]
        normal_binary = binary_local[normal, 0][normal_valid]
        components["normal_binary_dense"] = F.binary_cross_entropy_with_logits(
            normal_binary, torch.zeros_like(normal_binary)
        )
        normal_subtype = subtype_local[normal].permute(0, 2, 3, 1)[normal_valid]
        components["normal_subtype_dense"] = F.cross_entropy(
            normal_subtype, torch.zeros(len(normal_subtype), device=normal_subtype.device, dtype=torch.long)
        )
    else:
        components["normal_binary_dense"] = zero
        components["normal_subtype_dense"] = zero
    foreground, background = soft_intra_class_discrimination(
        subtype_local, valid, labels, subtypes
    )
    components["foreground_subtype"] = foreground
    components["background_normal"] = background

    tumor_mask = labels > 0.5
    if bool(tumor_mask.any()):
        subtype_margin = target_subtype_margin(subtype_local, subtypes)
        jointly_valid = valid[tumor_mask]
        binary_probability = torch.sigmoid(binary_local[tumor_mask, 0][jointly_valid])
        subtype_probability = torch.sigmoid(subtype_margin[tumor_mask, 0][jointly_valid])
        components["binary_subtype_alignment"] = F.mse_loss(
            binary_probability, subtype_probability
        )
    else:
        components["binary_subtype_alignment"] = zero
    total = sum(loss_weights[name] * value for name, value in components.items())
    return {"total": total, **components}


def _normalize_legacy_densenet_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
        r"(?:weight|bias|running_mean|running_var))$"
    )
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = pattern.match(key)
        normalized_key = match.group(1) + match.group(2) if match else key
        if normalized_key in normalized:
            raise ValueError(f"duplicate DenseNet key: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


class SMILELocalEvidence(nn.Module):
    """DenseNet121-FPN stride-4 matched-normal local evidence model."""

    output_stride = 4
    reference_stride = 16

    def __init__(
        self,
        *,
        arm: Literal["control", "full"],
        fpn_channels: int = 96,
        dropout: float = 0.10,
        match_temperature: float = 0.07,
        query_chunk_size: int = 1024,
        pretrained_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if arm not in {"control", "full"}:
            raise ValueError("arm must be control or full")
        from torchvision.models import densenet121

        backbone = densenet121(weights=None)
        if pretrained_checkpoint is not None:
            state = torch.load(Path(pretrained_checkpoint), map_location="cpu", weights_only=True)
            backbone.load_state_dict(_normalize_legacy_densenet_state_dict(state), strict=True)
        self.features = backbone.features
        self.lateral1 = nn.Conv2d(256, fpn_channels, 1)
        self.lateral2 = nn.Conv2d(512, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(1024, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(1024, fpn_channels, 1)
        self.refine3 = _Refine(fpn_channels)
        self.refine2 = _Refine(fpn_channels)
        self.refine1 = _Refine(fpn_channels)
        local_channels = 4 * fpn_channels + 1
        self.shared_local = nn.Sequential(
            nn.Conv2d(local_channels, fpn_channels, 1, bias=False),
            _group_norm(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.binary_head = nn.Conv2d(fpn_channels, 1, 1)
        self.subtype_head = nn.Conv2d(fpn_channels, SMILE_CLASS_COUNT, 1)
        self.arm = arm
        self.fpn_channels = int(fpn_channels)
        self.dropout_probability = float(dropout)
        self.match_temperature = float(match_temperature)
        self.query_chunk_size = int(query_chunk_size)

    def train(self, mode: bool = True) -> "SMILELocalEvidence":
        super().train(mode)
        if mode:
            for module in self.features.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @staticmethod
    def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def encode_fpn(self, image: torch.Tensor) -> torch.Tensor:
        f = self.features
        stem = f.pool0(f.relu0(f.norm0(f.conv0(image))))
        block1 = f.denseblock1(stem)
        block2 = f.denseblock2(f.transition1(block1))
        block3 = f.denseblock3(f.transition2(block2))
        block4 = torch.relu(f.norm5(f.denseblock4(f.transition3(block3))))
        p4 = self.lateral4(block4)
        p3 = self.refine3(self.lateral3(block3) + self._upsample_like(p4, block3))
        p2 = self.refine2(self.lateral2(block2) + self._upsample_like(p3, block2))
        return self.refine1(self.lateral1(block1) + self._upsample_like(p2, block1))

    @staticmethod
    def _grid_valid(valid: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError("valid must be BHW/B1HW")
        return F.interpolate(valid.float(), size=size, mode="nearest")[:, 0] > 0.5

    def forward(
        self,
        query: torch.Tensor,
        query_valid: torch.Tensor,
        references: torch.Tensor | None = None,
        reference_valid: torch.Tensor | None = None,
        *,
        conditioning_subtype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if query.ndim != 4 or query.shape[1] != 3:
            raise ValueError("query must be B3HW")
        query_feature = self.encode_fpn(query)
        valid = self._grid_valid(query_valid, query_feature.shape[-2:])
        if self.arm == "full":
            if references is None or reference_valid is None or references.ndim != 5:
                raise ValueError("full arm requires normal references")
            batch, count, channels, height, width = references.shape
            if batch != len(query) or channels != 3:
                raise ValueError("reference batch is incompatible")
            with torch.no_grad():
                flat_feature = self.encode_fpn(references.reshape(batch * count, channels, height, width))
            reference_feature = flat_feature.reshape(
                batch, count, self.fpn_channels, *flat_feature.shape[-2:]
            )
            flat_valid = reference_valid.reshape(batch * count, *reference_valid.shape[-2:])
            reference_grid_valid = self._grid_valid(
                flat_valid, flat_feature.shape[-2:]
            ).reshape(batch, count, *flat_feature.shape[-2:])
            pooled, pooled_valid = pool_reference_tokens(
                reference_feature, reference_grid_valid, factor=4
            )
            counterpart, cosine = matched_normal_counterparts(
                query_feature,
                pooled,
                valid,
                pooled_valid,
                temperature=self.match_temperature,
                query_chunk_size=self.query_chunk_size,
            )
            representation = local_evidence_representation(query_feature, counterpart, cosine)
        else:
            representation = query_only_control_representation(query_feature)
        shared = self.shared_local(representation)
        binary_local = self.binary_head(shared)
        subtype_local = self.subtype_head(shared)
        result = {
            "binary_image_logits": masked_sparse_pool(binary_local, valid)[:, 0],
            "subtype_image_logits": masked_sparse_pool(subtype_local, valid),
            "binary_evidence_logits": binary_local,
            "subtype_local_logits": subtype_local,
            "evidence_valid": valid[:, None],
        }
        if conditioning_subtype is not None:
            result["conditioned_evidence_logits"] = (
                binary_local + target_subtype_margin(subtype_local, conditioning_subtype)
            )
        return result

    def checkpoint_model_config(self) -> dict[str, object]:
        return {
            "method": SMILE_METHOD,
            "schema_version": SMILE_SCHEMA_VERSION,
            "arm": self.arm,
            "architecture": "densenet121_fpn_os4_smile",
            "fpn_channels": self.fpn_channels,
            "output_stride": self.output_stride,
            "reference_stride": self.reference_stride,
            "class_count": SMILE_CLASS_COUNT,
            "dropout": self.dropout_probability,
            "match_temperature": self.match_temperature,
            "query_chunk_size": self.query_chunk_size,
            "pool_fractions": list(SMILE_POOL_FRACTIONS),
            "loss_weights": dict(SMILE_LOSS_WEIGHTS),
            "residual_weights": dict(SMILE_RESIDUAL_WEIGHTS),
            "absolute_position_channels": False,
            "global_classifier_bypass": False,
            "proposal_masks_used_in_training": False,
            "source_ids_used_in_training": False,
        }


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("rank values must be finite and non-empty")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / max(1, len(values) - 1)


def _binary_dilation(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive")
    return F.max_pool2d(mask.float()[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0.5


def score_gallery_candidates_from_evidence(
    evidence_logits: np.ndarray,
    candidates: np.ndarray,
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
    *,
    top_cells: int = 17,
    ring_radius: int = 5,
) -> dict[str, np.ndarray]:
    """Read frozen identity and extent scores without source or GT inputs."""

    evidence = np.asarray(evidence_logits, dtype=np.float32)
    masks = np.asarray(candidates, dtype=bool)
    g1 = np.asarray(g1_logits, dtype=np.float64).reshape(-1)
    upstream = np.asarray(upstream_scores, dtype=np.float64).reshape(-1)
    if evidence.ndim != 2 or masks.ndim != 3 or masks.shape[1:] != evidence.shape:
        raise ValueError("evidence/candidate shapes differ")
    if len(masks) != len(g1) or len(masks) != len(upstream) or len(masks) == 0:
        raise ValueError("candidate score counts differ")
    if top_cells <= 0 or ring_radius <= 0 or not np.isfinite(evidence).all():
        raise ValueError("invalid evidence/readout parameters")

    tensor = torch.from_numpy(evidence)
    valid_values = tensor.flatten()
    median = valid_values.median()
    mad = (valid_values - median).abs().median().clamp_min(1e-6)
    robust = (tensor - median) / (1.4826 * mad)
    support = torch.sigmoid((robust - 2.0) / 0.5)
    support_sum = support.sum().clamp_min(1e-6)
    identity: list[float] = []
    extent: list[float] = []
    for candidate in torch.from_numpy(masks):
        inside = tensor[candidate]
        if inside.numel() == 0:
            identity.append(float("-inf"))
            extent.append(0.0)
            continue
        count = min(top_cells, inside.numel())
        inside_top = torch.topk(inside, count).values.mean()
        ring = _binary_dilation(candidate, ring_radius) & ~candidate
        ring_value = tensor[ring].median() if bool(ring.any()) else median
        identity.append(float((inside_top - ring_value).item()))
        overlap = support[candidate].sum()
        soft_dice = 2.0 * overlap / (support_sum + candidate.sum().float()).clamp_min(1e-6)
        extent.append(float(soft_dice.item()))
    identity_array = np.asarray(identity, dtype=np.float64)
    finite_floor = float(np.min(identity_array[np.isfinite(identity_array)])) - 1.0
    identity_array[~np.isfinite(identity_array)] = finite_floor
    extent_array = np.asarray(extent, dtype=np.float64)
    baseline = 0.5 * (average_percentile_rank(g1) + average_percentile_rank(upstream))
    centered_identity = average_percentile_rank(identity_array) - 0.5
    centered_extent = average_percentile_rank(extent_array) - 0.5
    return {
        "baseline": baseline,
        "identity": identity_array,
        "extent": extent_array,
        "identity_only": baseline + 0.25 * centered_identity,
        "identity_extent": baseline + 0.15 * centered_identity + 0.10 * centered_extent,
    }

