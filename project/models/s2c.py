from __future__ import annotations

"""Resource-aware S2C-style localizer for image-label-only BTXRD WSSS.

This is a clean adaptation of S2C's two ideas: SAM-segment contrasting (SSC)
and CAM/SAM consistency (CPM).  It uses cached automatic SAM segments instead
of keeping SAM ViT-H online during every training iteration.
"""

from pathlib import Path
from typing import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.classifier import DenseNet121AnatomyClassifier


class SpatialClassPool2d(nn.Module):
    """Sparse MIL pooling used by the S2C spatial tumor head.

    The image logit is coupled directly to the strongest spatial responses;
    there is no independent global-classification bypass.  Keeping the small
    implementation here avoids importing the retired high-resolution model
    family into the thesis-final X4 source bundle.
    """

    def __init__(self, *, top_fraction: float) -> None:
        super().__init__()
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        self.mode = "top_percent"
        self.top_fraction = float(top_fraction)

    def forward(self, class_maps: torch.Tensor) -> torch.Tensor:
        if class_maps.ndim != 4:
            raise ValueError("class_maps must be [B,C,H,W]")
        flat = class_maps.flatten(start_dim=2)
        top_k = max(1, int(math.ceil(self.top_fraction * flat.shape[-1])))
        return flat.topk(top_k, dim=-1).values.mean(dim=-1)


class LegacyStride32DenseNet121S2C(nn.Module):
    """Compatibility encoder for reproducibility of retired global-local runs.

    This is deliberately separate from :class:`DenseNet121S2C`.  It preserves
    the old subtype/GAP checkpoint contract so existing audited artifacts can
    still be loaded, but it must not be used by the active binary stride-4
    teacher.
    """

    def __init__(
        self,
        *,
        num_tumor_types: int = 10,
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
        radimagenet_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        backbone = DenseNet121AnatomyClassifier(
            num_classes=num_tumor_types,
            pretrained=pretrained,
            dropout=dropout,
            radimagenet_checkpoint=radimagenet_checkpoint,
        )
        self.features = backbone.features
        channels = backbone.classifier_input_features
        self.embedding = nn.Sequential(
            nn.Conv2d(channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.tumor_cam_head = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.tumor_type_head = nn.Linear(channels, num_tumor_types)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.relu(self.features(image))
        embedding = self.embedding(features)
        tumor_cam_logits = self.tumor_cam_head(features)
        tumor_logit = self.avgpool(tumor_cam_logits).flatten(1).squeeze(1)
        pooled = self.avgpool(features).flatten(1)
        return {
            "features": features,
            "embedding": embedding,
            "tumor_cam_logits": tumor_cam_logits,
            "tumor_logit": tumor_logit,
            "tumor_type_logits": self.tumor_type_head(self.dropout(pooled)),
        }


class DenseNet121S2C(nn.Module):
    """Binary stride-4 FPN localizer regularized by cached SAM segments.

    The first implementation used the final stride-32 DenseNet tensor and
    global-average pooling.  That is structurally unsuitable for BTXRD: every
    validation lesion below 1% area is smaller than one final cell at 320 px,
    and GAP rewards spreading a weak positive response over the image.  This
    version exposes stride-4 features and uses a sparse top-k MIL pool tied
    directly to the one-channel tumor CAM.
    """

    output_stride = 4

    def __init__(
        self,
        *,
        fpn_channels: int = 64,
        embedding_dim: int = 64,
        pretrained: bool = True,
        dropout: float = 0.1,
        top_fraction: float = 0.0025,
        radimagenet_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if fpn_channels <= 0 or embedding_dim <= 0:
            raise ValueError("fpn_channels and embedding_dim must be positive")
        backbone = DenseNet121AnatomyClassifier(
            num_classes=2,
            pretrained=pretrained,
            dropout=dropout,
            radimagenet_checkpoint=radimagenet_checkpoint,
        )
        self.features = backbone.features
        self.fpn_channels = int(fpn_channels)
        self.embedding_dim = int(embedding_dim)
        self.dropout_probability = float(dropout)
        self.top_fraction = float(top_fraction)

        self.lateral1 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(512, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.refine3 = self._refine(fpn_channels)
        self.refine2 = self._refine(fpn_channels)
        self.refine1 = self._refine(fpn_channels)
        self.embedding = nn.Sequential(
            nn.Conv2d(fpn_channels, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(self._group_count(embedding_dim), embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.tumor_cam_head = nn.Conv2d(fpn_channels, 1, kernel_size=1, bias=False)
        self.classification_pool = SpatialClassPool2d(
            top_fraction=top_fraction,
        )

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        raise AssertionError("unreachable")

    @classmethod
    def _refine(cls, channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(cls._group_count(channels), channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward_pyramid(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f = self.features
        stem = f.pool0(f.relu0(f.norm0(f.conv0(image))))
        block1 = f.denseblock1(stem)
        block2 = f.denseblock2(f.transition1(block1))
        block3 = f.denseblock3(f.transition2(block2))
        block4 = torch.relu(f.norm5(f.denseblock4(f.transition3(block3))))
        return block1, block2, block3, block4

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        block1, block2, block3, block4 = self.forward_pyramid(image)
        pyramid4 = self.lateral4(block4)
        pyramid3 = self.refine3(
            self.lateral3(block3) + self._upsample_like(pyramid4, block3)
        )
        pyramid2 = self.refine2(
            self.lateral2(block2) + self._upsample_like(pyramid3, block2)
        )
        pyramid1 = self.refine1(
            self.lateral1(block1) + self._upsample_like(pyramid2, block1)
        )
        embedding = self.embedding(pyramid1)
        tumor_cam_logits = self.tumor_cam_head(self.dropout(pyramid1))
        tumor_logit = self.classification_pool(tumor_cam_logits).squeeze(1)
        return {
            "features": pyramid1,
            "embedding": embedding,
            "tumor_cam_logits": tumor_cam_logits,
            "tumor_logit": tumor_logit,
        }


def load_s2c_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[DenseNet121S2C, dict[str, object]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    supported_methods = {
        "cached_sam_s2c_style_wsss",
        "x4_cached_sam_s2c_style_wsss",
    }
    if not isinstance(payload, dict) or payload.get("method") not in supported_methods:
        raise ValueError(f"Not a cached-SAM S2C WSSS checkpoint: {checkpoint_path}")
    if payload.get("ground_truth_spatial_supervision") is not False:
        raise ValueError("S2C checkpoint does not assert annotation-free spatial supervision")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("S2C checkpoint is missing model_config")
    expected_config = {
        "architecture",
        "output_stride",
        "fpn_channels",
        "embedding_dim",
        "dropout",
        "pool_mode",
        "top_fraction",
        "classes",
    }
    if set(config) != expected_config:
        raise ValueError("S2C checkpoint model_config does not match binary stride-4 schema")
    if (
        config.get("architecture") != "densenet121_binary_fpn_s2c"
        or config.get("output_stride") != 4
        or config.get("pool_mode") != "top_percent"
        or config.get("classes") != ["normal", "tumor"]
    ):
        raise ValueError("Unsupported S2C architecture or class contract")
    model = DenseNet121S2C(
        fpn_channels=int(config["fpn_channels"]),
        embedding_dim=int(config.get("embedding_dim", 256)),
        pretrained=False,
        dropout=float(config.get("dropout", 0.2)),
        top_fraction=float(config["top_fraction"]),
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("S2C checkpoint is missing state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, payload


def normalize_positive_cam(
    tumor_cam_logits: torch.Tensor,
    *,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """S2C-style positive CAM with a fail-closed all-zero baseline.

    Sigmoid is deliberately avoided: sigmoid(0)=0.5 would make an
    uninformative/random classifier pass the CPM proposal threshold merely
    because SAM quality is high.
    """

    if tumor_cam_logits.ndim != 4 or tumor_cam_logits.shape[1] != 1:
        raise ValueError("tumor_cam_logits must be [B,1,H,W]")
    logits = tumor_cam_logits
    if size is not None and tuple(logits.shape[-2:]) != tuple(size):
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    positive = torch.relu(logits[:, 0])
    maxima = positive.flatten(1).amax(dim=1).view(-1, 1, 1)
    return torch.where(maxima > 1e-6, positive / maxima.clamp_min(1e-6), torch.zeros_like(positive))


def segment_contrastive_loss(
    embedding: torch.Tensor,
    segment_maps: torch.Tensor,
    *,
    temperature: float = 0.07,
    min_segment_pixels: int = 2,
    max_pixels_per_segment: int = 64,
) -> torch.Tensor:
    """Contrast each pixel with detached prototypes of its SAM segment.

    Segment id 0 is ignored.  The implementation deliberately uses stock
    PyTorch rather than torch-scatter so the preflight is portable.
    """

    if embedding.ndim != 4 or segment_maps.ndim != 3:
        raise ValueError("Expected embedding [B,D,H,W] and segment_maps [B,H,W]")
    if embedding.shape[0] != segment_maps.shape[0]:
        raise ValueError("Embedding/segment batch sizes differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if max_pixels_per_segment < 1:
        raise ValueError("max_pixels_per_segment must be positive")

    # Stratified high-resolution sampling preserves every sufficiently large
    # SAM region without materializing BxDxHxW embeddings or a full
    # pixel-by-prototype matrix at 320/512 px.  Sampling coordinates from the
    # original SAM grid avoids the small-lesion erasure caused by reducing the
    # segment map to DenseNet's output stride.
    resized = segment_maps.to(torch.long)
    losses: list[torch.Tensor] = []
    weights: list[int] = []
    map_height, map_width = resized.shape[-2:]
    for batch_index in range(embedding.shape[0]):
        labels = resized[batch_index].reshape(-1)
        valid_ids, counts = torch.unique(labels[labels > 0], sorted=True, return_counts=True)
        valid_ids = valid_ids[counts >= min_segment_pixels]
        if valid_ids.numel() < 2:
            continue

        sampled_indices: list[torch.Tensor] = []
        sampled_labels: list[torch.Tensor] = []
        for segment_id in valid_ids:
            indices = torch.nonzero(labels == segment_id, as_tuple=False)[:, 0]
            if indices.numel() > max_pixels_per_segment:
                order = torch.randperm(indices.numel(), device=indices.device)[:max_pixels_per_segment]
                indices = indices[order]
            sampled_indices.append(indices)
            sampled_labels.append(segment_id.expand(indices.numel()))
        flat_indices = torch.cat(sampled_indices)
        pixel_labels = torch.cat(sampled_labels)
        y = torch.div(flat_indices, map_width, rounding_mode="floor")
        x = flat_indices.remainder(map_width)
        grid_x = 2.0 * (x.to(embedding.dtype) + 0.5) / map_width - 1.0
        grid_y = 2.0 * (y.to(embedding.dtype) + 0.5) / map_height - 1.0
        grid = torch.stack((grid_x, grid_y), dim=1).view(1, -1, 1, 2)
        pixel_features = F.grid_sample(
            embedding[batch_index : batch_index + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].t()
        pixel_features = F.normalize(pixel_features, dim=1)
        target = torch.searchsorted(valid_ids, pixel_labels)

        prototypes: list[torch.Tensor] = []
        detached = pixel_features.detach()
        for target_index in range(valid_ids.numel()):
            prototypes.append(detached[target == target_index].mean(dim=0))
        prototype_tensor = F.normalize(torch.stack(prototypes), dim=1)
        logits = pixel_features @ prototype_tensor.t() / temperature
        losses.append(F.cross_entropy(logits, target))
        weights.append(int(pixel_features.shape[0]))

    if not losses:
        return embedding.sum() * 0.0
    weight_tensor = embedding.new_tensor(weights, dtype=torch.float32)
    stacked = torch.stack(losses)
    return (stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def _resize_segments(segment_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        segment_map[None, None].to(torch.float32),
        size=size,
        mode="nearest",
    )[0, 0].to(torch.long)


def score_cam_segments(
    cam_probability: torch.Tensor,
    segment_map: torch.Tensor,
    quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return segment ids, CAM means and quality-weighted means."""

    if cam_probability.ndim != 2 or segment_map.ndim != 2 or quality.ndim != 1:
        raise ValueError("Expected 2-D CAM/segments and 1-D quality")
    segments = _resize_segments(segment_map, tuple(cam_probability.shape))
    ids = torch.unique(segments)
    ids = ids[ids > 0]
    if ids.numel() == 0:
        empty = cam_probability.new_empty((0,))
        return ids, empty, empty
    if int(ids.max()) >= int(quality.numel()):
        raise ValueError("SAM quality vector does not cover every segment id")
    means = torch.stack([cam_probability[segments == segment_id].mean() for segment_id in ids])
    joint = means * quality.to(cam_probability.device)[ids]
    return ids, means, joint


def _resize_proposals(proposal_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if proposal_masks.ndim != 3:
        raise ValueError("proposal_masks must be [N,H,W]")
    if proposal_masks.shape[0] == 0:
        return torch.zeros((0, *size), dtype=torch.bool, device=proposal_masks.device)
    masks = proposal_masks.to(torch.bool)
    if tuple(masks.shape[-2:]) == tuple(size):
        return masks
    source_height, source_width = masks.shape[-2:]
    target_height, target_width = size
    y = torch.div(
        torch.arange(target_height, device=masks.device) * source_height,
        target_height,
        rounding_mode="floor",
    ).clamp_max(source_height - 1)
    x = torch.div(
        torch.arange(target_width, device=masks.device) * source_width,
        target_width,
        rounding_mode="floor",
    ).clamp_max(source_width - 1)
    return masks.index_select(1, y).index_select(2, x)


def score_cam_proposals(
    cam_probability: torch.Tensor,
    proposal_masks: torch.Tensor,
    proposal_quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean CAM and SAM-quality-weighted score per overlapping mask."""

    if cam_probability.ndim != 2 or proposal_quality.ndim != 1:
        raise ValueError("Expected a 2-D CAM and 1-D proposal quality")
    masks = _resize_proposals(proposal_masks, tuple(cam_probability.shape))
    if masks.shape[0] != proposal_quality.numel():
        raise ValueError("Proposal mask/quality counts differ")
    if masks.shape[0] == 0:
        empty = cam_probability.new_empty((0,))
        return empty, empty
    flat_cam = cam_probability.reshape(-1)
    means_chunks: list[torch.Tensor] = []
    for start in range(0, masks.shape[0], 16):
        flat_masks = masks[start : start + 16].reshape(-1, flat_cam.numel()).to(cam_probability.dtype)
        means_chunks.append((flat_masks @ flat_cam) / flat_masks.sum(dim=1).clamp_min(1.0))
    means = torch.cat(means_chunks)
    joint = means * proposal_quality.to(cam_probability.device)
    return means, joint


def select_cam_guided_proposals(
    cam_probability: torch.Tensor,
    proposal_masks: torch.Tensor,
    proposal_quality: torch.Tensor,
    *,
    image_is_tumor: bool,
    positive_threshold: float = 0.35,
    min_positive_score: float = 0.20,
    min_sam_quality: float = 0.70,
    top_k: int = 3,
    nms_iou_threshold: float = 0.80,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select non-redundant overlapping SAM masks using CAM and image label."""

    masks = _resize_proposals(proposal_masks, tuple(cam_probability.shape))
    output = torch.zeros_like(cam_probability, dtype=torch.bool)
    if not image_is_tumor:
        return output, {"selected_ids": [], "selected_scores": [], "reason": "known_image_label_normal"}
    if masks.shape[0] == 0:
        return output, {"selected_ids": [], "selected_scores": [], "reason": "no_proposals"}
    _means, joint = score_cam_proposals(cam_probability, proposal_masks, proposal_quality)
    quality = proposal_quality.to(joint.device)
    eligible = torch.nonzero(quality >= min_sam_quality, as_tuple=False)[:, 0]
    if eligible.numel() == 0:
        return output, {"selected_ids": [], "selected_scores": [], "reason": "no_quality_proposals"}
    order = eligible[torch.argsort(joint[eligible], descending=True)]
    thresholded = order[joint[order] >= positive_threshold]
    if thresholded.numel() == 0 and float(joint[order[0]].item()) >= min_positive_score:
        thresholded = order[:1]

    selected: list[int] = []
    for candidate_tensor in thresholded:
        candidate = int(candidate_tensor.item())
        candidate_mask = masks[candidate]
        redundant = False
        for previous in selected:
            intersection = (candidate_mask & masks[previous]).sum().to(torch.float32)
            union = (candidate_mask | masks[previous]).sum().to(torch.float32).clamp_min(1.0)
            if float((intersection / union).item()) >= nms_iou_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    if selected:
        selected_tensor = torch.tensor(selected, device=masks.device, dtype=torch.long)
        output = masks[selected_tensor].any(dim=0)
        selected_scores = joint[selected_tensor]
    else:
        selected_scores = joint.new_empty((0,))
    return output, {
        "selected_ids": selected,
        "selected_scores": [float(value) for value in selected_scores.detach().cpu().tolist()],
        "reason": "ok" if selected else "below_confidence",
    }


def select_cam_guided_segments(
    cam_probability: torch.Tensor,
    segment_map: torch.Tensor,
    quality: torch.Tensor,
    *,
    image_is_tumor: bool,
    positive_threshold: float = 0.35,
    min_positive_score: float = 0.20,
    min_sam_quality: float = 0.70,
    top_k: int = 3,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select a pseudo lesion using only CAM, SAM cache and image label."""

    segments = _resize_segments(segment_map, tuple(cam_probability.shape))
    output = torch.zeros_like(segments, dtype=torch.bool)
    if not image_is_tumor:
        return output, {"selected_ids": [], "selected_scores": [], "reason": "known_image_label_normal"}
    ids, _means, joint = score_cam_segments(cam_probability, segment_map, quality)
    if ids.numel() == 0:
        return output, {"selected_ids": [], "selected_scores": [], "reason": "no_segments"}
    valid_quality = quality.to(joint.device)[ids] >= min_sam_quality
    eligible_ids = ids[valid_quality]
    eligible_scores = joint[valid_quality]
    if eligible_ids.numel() == 0:
        return output, {"selected_ids": [], "selected_scores": [], "reason": "no_quality_segments"}

    order = torch.argsort(eligible_scores, descending=True)
    keep = order[eligible_scores[order] >= positive_threshold][:top_k]
    if keep.numel() == 0 and float(eligible_scores[order[0]].item()) >= min_positive_score:
        keep = order[:1]
    selected_ids = eligible_ids[keep]
    selected_scores = eligible_scores[keep]
    for segment_id in selected_ids:
        output |= segments == segment_id
    reason = "ok" if selected_ids.numel() else "below_confidence"
    return output, {
        "selected_ids": [int(value) for value in selected_ids.detach().cpu().tolist()],
        "selected_scores": [float(value) for value in selected_scores.detach().cpu().tolist()],
        "reason": reason,
    }


def build_cached_cpm_targets(
    cam_probability: torch.Tensor,
    segment_maps: torch.Tensor,
    qualities: Sequence[torch.Tensor],
    tumor_labels: torch.Tensor,
    *,
    proposal_masks: Sequence[torch.Tensor] | None = None,
    proposal_qualities: Sequence[torch.Tensor] | None = None,
    positive_threshold: float = 0.35,
    min_positive_score: float = 0.20,
    negative_cam_threshold: float = 0.10,
    min_sam_quality: float = 0.70,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Create detached CPM targets from cached SAM regions and current CAM."""

    if cam_probability.ndim != 3 or segment_maps.ndim != 3:
        raise ValueError("Expected CAM and segment maps shaped [B,H,W]")
    if len(qualities) != cam_probability.shape[0]:
        raise ValueError("qualities length differs from batch size")
    if (proposal_masks is None) != (proposal_qualities is None):
        raise ValueError("proposal_masks and proposal_qualities must be supplied together")
    if proposal_masks is not None and len(proposal_masks) != cam_probability.shape[0]:
        raise ValueError("proposal bank length differs from batch size")
    targets = torch.full_like(cam_probability, 255, dtype=torch.long)
    weights = torch.zeros_like(cam_probability)
    selected_images = 0
    selected_segments = 0
    with torch.no_grad():
        for batch_index in range(cam_probability.shape[0]):
            if not bool(tumor_labels[batch_index].item() > 0.5):
                targets[batch_index].fill_(0)
                weights[batch_index].fill_(1.0)
                continue
            segments = _resize_segments(segment_maps[batch_index], tuple(cam_probability.shape[-2:]))
            quality = qualities[batch_index].to(cam_probability.device)
            ids, means, _joint = score_cam_segments(
                cam_probability[batch_index],
                segment_maps[batch_index],
                quality,
            )
            if ids.numel() == 0:
                continue
            reliable = quality[ids] >= min_sam_quality
            # Confidently inactive SAM regions teach background; ambiguous
            # regions stay ignored rather than becoming false negatives.
            for segment_id, mean, is_reliable in zip(ids, means, reliable):
                if bool(is_reliable) and float(mean.item()) <= negative_cam_threshold:
                    region = segments == segment_id
                    targets[batch_index][region] = 0
                    weights[batch_index][region] = quality[segment_id]
            use_proposals = proposal_masks is not None and proposal_masks[batch_index].shape[0] > 0
            if use_proposals:
                selected, info = select_cam_guided_proposals(
                    cam_probability[batch_index],
                    proposal_masks[batch_index].to(cam_probability.device),
                    proposal_qualities[batch_index].to(cam_probability.device),  # type: ignore[index]
                    image_is_tumor=True,
                    positive_threshold=positive_threshold,
                    min_positive_score=min_positive_score,
                    min_sam_quality=min_sam_quality,
                    top_k=top_k,
                )
            else:
                selected, info = select_cam_guided_segments(
                    cam_probability[batch_index],
                    segment_maps[batch_index],
                    quality,
                    image_is_tumor=True,
                    positive_threshold=positive_threshold,
                    min_positive_score=min_positive_score,
                    min_sam_quality=min_sam_quality,
                    top_k=top_k,
                )
            if selected.any():
                targets[batch_index][selected] = 1
                selected_ids = info["selected_ids"]
                if use_proposals:
                    resized_proposals = _resize_proposals(
                        proposal_masks[batch_index].to(cam_probability.device),  # type: ignore[index]
                        tuple(cam_probability.shape[-2:]),
                    )
                    selected_quality = proposal_qualities[batch_index].to(cam_probability.device)  # type: ignore[index]
                    for proposal_id in selected_ids:
                        region = resized_proposals[int(proposal_id)]
                        weights[batch_index][region] = torch.maximum(
                            weights[batch_index][region],
                            selected_quality[int(proposal_id)].expand_as(weights[batch_index][region]),
                        )
                else:
                    for segment_id in selected_ids:
                        region = segments == int(segment_id)
                        weights[batch_index][region] = quality[int(segment_id)]
                selected_images += 1
                selected_segments += len(selected_ids)
    return targets, weights, {
        "selected_images": selected_images,
        "selected_segments": selected_segments,
    }


def cpm_consistency_loss(
    tumor_cam_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    positive_weight_max: float = 20.0,
) -> torch.Tensor:
    if tumor_cam_logits.ndim != 4 or tumor_cam_logits.shape[1] != 1:
        raise ValueError("tumor_cam_logits must be [B,1,H,W]")
    if positive_weight_max < 1.0:
        raise ValueError("positive_weight_max must be at least 1")
    spatial_logits = F.interpolate(
        tumor_cam_logits,
        size=targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    binary_targets = targets.clamp(0, 1).to(spatial_logits.dtype)
    per_pixel = F.binary_cross_entropy_with_logits(spatial_logits, binary_targets, reduction="none")
    valid_weight = weights * (targets != 255).to(weights.dtype)
    per_image_losses: list[torch.Tensor] = []
    for batch_index in range(spatial_logits.shape[0]):
        sample_weight = valid_weight[batch_index]
        if not bool((sample_weight > 0).any()):
            continue
        sample_target = binary_targets[batch_index]
        positive_mass = (sample_weight * sample_target).sum()
        negative_mass = (sample_weight * (1.0 - sample_target)).sum()
        if bool(positive_mass > 0) and bool(negative_mass > 0):
            positive_multiplier = torch.clamp(
                negative_mass / positive_mass.clamp_min(1e-6),
                min=1.0,
                max=float(positive_weight_max),
            )
            sample_weight = sample_weight * torch.where(
                sample_target > 0.5,
                positive_multiplier,
                torch.ones_like(sample_target),
            )
        per_image_losses.append(
            (per_pixel[batch_index] * sample_weight).sum()
            / sample_weight.sum().clamp_min(1.0)
        )
    if not per_image_losses:
        return tumor_cam_logits.sum() * 0.0
    return torch.stack(per_image_losses).mean()
