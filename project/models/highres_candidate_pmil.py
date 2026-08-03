"""Dataset-agnostic primitives for S10 high-resolution proposal MIL.

The module contains no BTXRD loader, segmentation target, evaluator or test
access. Inputs are feature maps, class-agnostic candidate supports and image
labels only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchvision.models import resnet50
except (ImportError, RuntimeError):  # pragma: no cover - environment guard
    resnet50 = None

from models.bas_candidate_localizer import within_bag_percentile_ranks


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_candidate_tensors(
    feature_map: torch.Tensor,
    candidate_weights: torch.Tensor,
    ring_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if feature_map.ndim != 4 or candidate_weights.ndim != 4:
        raise ValueError("feature_map/candidate_weights must be BCHW/BNHW")
    if ring_weights.shape != candidate_weights.shape:
        raise ValueError("candidate and ring supports differ")
    batch, channels, height, width = feature_map.shape
    if candidate_weights.shape[0] != batch or candidate_weights.shape[-2:] != (
        height,
        width,
    ):
        raise ValueError("candidate supports do not align with feature map")
    candidates = candidate_weights.shape[1]
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("candidate_valid must be BN")
    for name, value in (
        ("feature_map", feature_map),
        ("candidate_weights", candidate_weights),
        ("ring_weights", ring_weights),
    ):
        _finite(name, value)
    if bool((candidate_weights < 0).any()) or bool((ring_weights < 0).any()):
        raise ValueError("candidate supports must be non-negative")
    inside_mass = candidate_weights.sum(dim=(-2, -1))
    ring_mass = ring_weights.sum(dim=(-2, -1))
    if bool(((inside_mass <= 0) & candidate_valid).any()):
        raise ValueError("valid candidate has zero inside mass")
    if bool(((ring_mass <= 0) & candidate_valid).any()):
        raise ValueError("valid candidate has zero ring mass")
    if not candidate_valid.any(dim=1).all():
        raise ValueError("every image requires a valid candidate")
    return batch, channels, height, width, candidates


def masked_candidate_zone_descriptors(
    feature_map: torch.Tensor,
    candidate_weights: torch.Tensor,
    ring_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Pool inside/ring/global and inside-minus-ring without raw geometry."""

    batch, channels, height, width, candidates = _validate_candidate_tensors(
        feature_map, candidate_weights, ring_weights, candidate_valid
    )
    spatial = feature_map.float().reshape(batch, channels, height * width)
    inside_weights = candidate_weights.float().reshape(batch, candidates, -1)
    outside_weights = ring_weights.float().reshape(batch, candidates, -1)
    inside = torch.einsum("bnp,bcp->bnc", inside_weights, spatial)
    inside = inside / inside_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    ring = torch.einsum("bnp,bcp->bnc", outside_weights, spatial)
    ring = ring / outside_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    global_context = spatial.mean(dim=-1)[:, None].expand(-1, candidates, -1)
    result = torch.cat((inside, ring, inside - ring, global_context), dim=-1)
    result = result * candidate_valid[..., None].to(result.dtype)
    if result.shape != (batch, candidates, 4 * channels):
        raise RuntimeError("candidate zone descriptor shape changed")
    _finite("candidate zone descriptors", result)
    return result


class CandidateSetTransformer(nn.Module):
    """Permutation-equivariant contextualization of proposal descriptors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or hidden_dim % heads or layers <= 0:
            raise ValueError("candidate transformer dimensions are invalid")
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classification = nn.Linear(hidden_dim, 1)
        self.detection = nn.Linear(hidden_dim, 1)

    def forward(
        self, descriptors: torch.Tensor, candidate_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.input_dim:
            raise ValueError("descriptors must be BND")
        if candidate_valid.shape != descriptors.shape[:2]:
            raise ValueError("candidate_valid does not align with descriptors")
        if not candidate_valid.any(dim=1).all():
            raise ValueError("every image requires a valid candidate")
        _finite("descriptors", descriptors)
        hidden = self.projection(descriptors.float())
        hidden = self.encoder(hidden, src_key_padding_mask=~candidate_valid)
        classification = self.classification(hidden)[..., 0]
        detection = self.detection(hidden)[..., 0]
        classification = classification.masked_fill(~candidate_valid, -torch.inf)
        detection = detection.masked_fill(~candidate_valid, -torch.inf)
        return classification, detection


@dataclass(frozen=True)
class HighResProposalMILOutput:
    classification_logits: torch.Tensor
    detection_logits: torch.Tensor
    dense_logits: torch.Tensor
    candidate_weights: torch.Tensor
    ring_weights: torch.Tensor
    candidate_area: torch.Tensor
    candidate_valid: torch.Tensor


class ResNet50FeaturePyramid(nn.Module):
    """Trainable ResNet-50 C2-C5 pyramid fused at output stride four."""

    def __init__(
        self,
        *,
        channels: int = 128,
        backbone_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("FPN channels must be positive")
        if resnet50 is None:
            raise RuntimeError("torchvision ResNet-50 is unavailable")
        backbone = resnet50(weights=None)
        if backbone_state_dict is not None:
            backbone.load_state_dict(backbone_state_dict, strict=True)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lateral = nn.ModuleList(
            nn.Conv2d(source, channels, kernel_size=1)
            for source in (256, 512, 1024, 2048)
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(4)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be Bx3xHxW")
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError("image dimensions must be divisible by 32")
        _finite("images", images)
        stem = self.stem(images)
        c2 = self.layer1(stem)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        sources = (c2, c3, c4, c5)
        pyramid: list[torch.Tensor] = [torch.empty(0)] * 4
        top: torch.Tensor | None = None
        for index in range(3, -1, -1):
            lateral = self.lateral[index](sources[index])
            if top is not None:
                lateral = lateral + F.interpolate(
                    top, size=lateral.shape[-2:], mode="nearest"
                )
            top = lateral
            pyramid[index] = self.smooth[index](lateral)
        target = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                value
                if value.shape[-2:] == target
                else F.interpolate(value, size=target, mode="bilinear", align_corners=False)
                for value in pyramid
            ],
            dim=1,
        )
        output = self.fuse(fused)
        if output.shape[-2:] != (images.shape[-2] // 4, images.shape[-1] // 4):
            raise RuntimeError("FPN output stride changed")
        return output


def project_candidate_supports(
    square_candidate_masks: torch.Tensor,
    square_content_masks: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    output_size: tuple[int, int],
    ring_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Area-project candidates and construct a bounded exterior ring."""

    if square_candidate_masks.ndim != 4 or square_content_masks.ndim != 3:
        raise ValueError("candidate/content supports must be BNHW/BHW")
    if square_candidate_masks.shape[0] != square_content_masks.shape[0]:
        raise ValueError("candidate/content batch differs")
    if square_candidate_masks.shape[-2:] != square_content_masks.shape[-2:]:
        raise ValueError("candidate/content spatial shapes differ")
    if candidate_valid.shape != square_candidate_masks.shape[:2]:
        raise ValueError("candidate_valid must be BN")
    if output_size[0] <= 0 or output_size[1] <= 0 or ring_radius <= 0:
        raise ValueError("support projection controls are invalid")
    for name, value in (
        ("square candidate masks", square_candidate_masks),
        ("square content masks", square_content_masks),
    ):
        _finite(name, value)
        if bool((value < 0).any()):
            raise ValueError(f"{name} must be non-negative")
    batch, candidates = square_candidate_masks.shape[:2]
    masks = F.interpolate(
        square_candidate_masks.float().reshape(batch * candidates, 1, *square_candidate_masks.shape[-2:]),
        size=output_size,
        mode="area",
    ).reshape(batch, candidates, *output_size).clamp(0.0, 1.0)
    content = F.interpolate(
        square_content_masks.float()[:, None], size=output_size, mode="area"
    )[:, 0].clamp(0.0, 1.0)
    masks = masks * content[:, None]
    kernel = 2 * ring_radius + 1
    dilated = F.max_pool2d(
        masks.reshape(batch * candidates, 1, *output_size),
        kernel_size=kernel,
        stride=1,
        padding=ring_radius,
    ).reshape(batch, candidates, *output_size)
    rings = (dilated - masks).clamp_min(0.0) * content[:, None]
    area = masks.sum(dim=(-2, -1))
    ring_area = rings.sum(dim=(-2, -1))
    if bool(((area <= 0) & candidate_valid).any()):
        raise ValueError("valid candidate vanished during FPN projection")
    if bool(((ring_area <= 0) & candidate_valid).any()):
        raise ValueError("valid candidate has no projected exterior ring")
    masks = masks * candidate_valid[..., None, None].to(masks.dtype)
    rings = rings * candidate_valid[..., None, None].to(rings.dtype)
    return masks, rings, area


class HighResProposalMIL(nn.Module):
    """End-to-end high-resolution proposal-conditioned binary MIL model."""

    def __init__(
        self,
        *,
        fpn_channels: int = 128,
        set_hidden_dim: int = 256,
        set_heads: int = 4,
        set_layers: int = 2,
        set_dropout: float = 0.1,
        ring_radius: int = 3,
        backbone_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if ring_radius <= 0:
            raise ValueError("ring radius must be positive")
        self.ring_radius = ring_radius
        self.fpn = ResNet50FeaturePyramid(
            channels=fpn_channels, backbone_state_dict=backbone_state_dict
        )
        self.dense_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fpn_channels // 2, 1, kernel_size=1),
        )
        self.proposal_head = CandidateSetTransformer(
            4 * fpn_channels,
            hidden_dim=set_hidden_dim,
            heads=set_heads,
            layers=set_layers,
            dropout=set_dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        square_candidate_masks: torch.Tensor,
        square_content_masks: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> HighResProposalMILOutput:
        features = self.fpn(images)
        candidate_weights, ring_weights, candidate_area = project_candidate_supports(
            square_candidate_masks,
            square_content_masks,
            candidate_valid,
            output_size=features.shape[-2:],
            ring_radius=self.ring_radius,
        )
        descriptors = masked_candidate_zone_descriptors(
            features, candidate_weights, ring_weights, candidate_valid
        )
        classification, detection = self.proposal_head(descriptors, candidate_valid)
        dense_logits = self.dense_head(features)[:, 0]
        return HighResProposalMILOutput(
            classification_logits=classification,
            detection_logits=detection,
            dense_logits=dense_logits,
            candidate_weights=candidate_weights,
            ring_weights=ring_weights,
            candidate_area=candidate_area,
            candidate_valid=candidate_valid,
        )


def dual_stream_bag_probability(
    classification_logits: torch.Tensor,
    detection_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """WSDDN-style proposal probability with within-bag detection competition."""

    if (
        classification_logits.ndim != 2
        or detection_logits.shape != classification_logits.shape
        or candidate_valid.shape != classification_logits.shape
    ):
        raise ValueError("proposal logits/valid mask must be BN")
    if not candidate_valid.any(dim=1).all():
        raise ValueError("every bag requires a valid candidate")
    if not torch.isfinite(classification_logits[candidate_valid]).all():
        raise ValueError("classification logits are non-finite")
    if not torch.isfinite(detection_logits[candidate_valid]).all():
        raise ValueError("detection logits are non-finite")
    classification = torch.sigmoid(classification_logits).masked_fill(
        ~candidate_valid, 0.0
    )
    attention = torch.softmax(
        detection_logits.masked_fill(~candidate_valid, -torch.inf), dim=1
    ).masked_fill(~candidate_valid, 0.0)
    bag_probability = (classification * attention).sum(dim=1)
    if bool((bag_probability <= 0).any()) or bool((bag_probability >= 1).any()):
        bag_probability = bag_probability.clamp(1.0e-7, 1.0 - 1.0e-7)
    return {
        "bag_probability": bag_probability,
        "classification_probability": classification,
        "detection_attention": attention,
    }


def area_orthogonality_penalty(
    logits: torch.Tensor,
    candidate_area: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Squared nuisance projection onto log area, zero for degenerate area."""

    if logits.shape != candidate_area.shape or logits.shape != candidate_valid.shape:
        raise ValueError("logits/area/valid shapes differ")
    if bool((candidate_area[candidate_valid] <= 0).any()):
        raise ValueError("valid candidate area must be positive")
    _finite("candidate area", candidate_area)
    penalties: list[torch.Tensor] = []
    for row_logits, row_area, row_valid in zip(logits, candidate_area, candidate_valid):
        values = row_logits[row_valid]
        areas = row_area[row_valid].float().log()
        if len(values) < 2:
            penalties.append(values.sum() * 0.0)
            continue
        values = values - values.mean()
        areas = areas - areas.mean()
        denominator = areas.square().sum()
        if float(denominator.detach()) <= 1.0e-12:
            penalties.append(values.sum() * 0.0)
        else:
            penalties.append((values * areas).sum().square() / denominator)
    return torch.stack(penalties).mean()


def top_instance_dropout_mask(
    detection_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Deterministically hide the most important training instances per bag."""

    if detection_logits.ndim != 2 or candidate_valid.shape != detection_logits.shape:
        raise ValueError("detection logits/validity must be BN")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("dropout fraction must lie in [0,1)")
    if not candidate_valid.any(dim=1).all():
        raise ValueError("every bag requires a valid candidate")
    if not torch.isfinite(detection_logits[candidate_valid]).all():
        raise ValueError("valid detection logits are non-finite")
    retained = candidate_valid.clone()
    for row in range(detection_logits.shape[0]):
        indices = torch.nonzero(candidate_valid[row], as_tuple=False).reshape(-1)
        drop_count = min(int(len(indices) * fraction), len(indices) - 1)
        if drop_count <= 0:
            continue
        # Stable tie break by local candidate order; inference never calls this.
        ranked = sorted(
            indices.tolist(),
            key=lambda index: (-float(detection_logits[row, index].detach()), index),
        )
        retained[row, ranked[:drop_count]] = False
    return retained


def aligned_view_consistency(
    original_candidate_logits: torch.Tensor,
    aligned_flip_candidate_logits: torch.Tensor,
    original_dense_logits: torch.Tensor,
    aligned_flip_dense_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Symmetric proposal and dense-map consistency after physical alignment."""

    if (
        original_candidate_logits.shape != aligned_flip_candidate_logits.shape
        or original_candidate_logits.shape != candidate_valid.shape
    ):
        raise ValueError("aligned candidate views must share BN shape")
    if (
        original_dense_logits.ndim != 3
        or aligned_flip_dense_logits.shape != original_dense_logits.shape
        or original_dense_logits.shape[0] != candidate_valid.shape[0]
    ):
        raise ValueError("aligned dense views must share BHW shape")
    for name, value in (
        ("original candidate logits", original_candidate_logits[candidate_valid]),
        ("aligned flip candidate logits", aligned_flip_candidate_logits[candidate_valid]),
        ("original dense logits", original_dense_logits),
        ("aligned flip dense logits", aligned_flip_dense_logits),
    ):
        _finite(name, value)
    proposal = F.smooth_l1_loss(
        original_candidate_logits[candidate_valid],
        aligned_flip_candidate_logits[candidate_valid],
    )
    dense = F.smooth_l1_loss(original_dense_logits, aligned_flip_dense_logits)
    return 0.5 * (proposal + dense)


def image_label_proposal_loss(
    classification_logits: torch.Tensor,
    detection_logits: torch.Tensor,
    dense_logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Bag BCE plus explicit dense proposal/pixel negatives for normal images."""

    labels = labels.float().reshape(-1)
    if dense_logits.ndim != 3 or dense_logits.shape[0] != labels.shape[0]:
        raise ValueError("dense logits must be BHW")
    if classification_logits.shape[0] != labels.shape[0]:
        raise ValueError("proposal logits and labels differ")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("labels must be binary")
    output = dual_stream_bag_probability(
        classification_logits, detection_logits, candidate_valid
    )
    bag = F.binary_cross_entropy(output["bag_probability"], labels)
    normal = labels == 0
    zero = classification_logits[candidate_valid].sum() * 0.0
    candidate_negative = (
        F.softplus(classification_logits[normal][candidate_valid[normal]]).mean()
        if bool(normal.any())
        else zero
    )
    pixel_negative = (
        F.softplus(dense_logits[normal]).mean() if bool(normal.any()) else zero
    )
    return {
        "total": bag + candidate_negative + pixel_negative,
        "bag": bag,
        "normal_candidate": candidate_negative,
        "normal_pixel": pixel_negative,
        **output,
    }


def attention_union_consistency(
    dense_logits: torch.Tensor,
    candidate_weights: torch.Tensor,
    detection_attention: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Soft-Dice alignment of dense evidence and attention-weighted proposals."""

    if dense_logits.ndim != 3 or candidate_weights.ndim != 4:
        raise ValueError("dense logits/candidates must be BHW/BNHW")
    if candidate_weights.shape[0] != dense_logits.shape[0] or candidate_weights.shape[-2:] != dense_logits.shape[-2:]:
        raise ValueError("candidate supports do not align with dense logits")
    if detection_attention.shape != candidate_weights.shape[:2]:
        raise ValueError("detection attention must be BN")
    if candidate_valid.shape != detection_attention.shape:
        raise ValueError("candidate_valid must be BN")
    _finite("dense logits", dense_logits)
    _finite("candidate weights", candidate_weights)
    _finite("detection attention", detection_attention)
    union = (
        detection_attention.masked_fill(~candidate_valid, 0.0)[..., None, None]
        * candidate_weights.float()
    ).sum(dim=1).clamp(0.0, 1.0)
    evidence = torch.sigmoid(dense_logits.float())
    intersection = (union * evidence).sum(dim=(-2, -1))
    denominator = union.sum(dim=(-2, -1)) + evidence.sum(dim=(-2, -1))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def candidate_capture_purity(
    dense_logits: torch.Tensor,
    candidate_weights: torch.Tensor,
    ring_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
    content_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous evidence capture and local purity for immutable candidates."""

    _validate_candidate_tensors(
        dense_logits[:, None], candidate_weights, ring_weights, candidate_valid
    )
    if content_weights.shape != dense_logits.shape:
        raise ValueError("content_weights must be BHW")
    _finite("content_weights", content_weights)
    if bool((content_weights < 0).any()):
        raise ValueError("content weights must be non-negative")
    evidence = torch.sigmoid(dense_logits.float()) * content_weights.float()
    inside_mass = candidate_weights.sum(dim=(-2, -1)).clamp_min(1.0e-12)
    ring_mass = ring_weights.sum(dim=(-2, -1)).clamp_min(1.0e-12)
    inside_evidence = (candidate_weights * evidence[:, None]).sum(dim=(-2, -1))
    ring_evidence = (ring_weights * evidence[:, None]).sum(dim=(-2, -1))
    total_evidence = evidence.sum(dim=(-2, -1), keepdim=False).clamp_min(1.0e-12)
    capture = inside_evidence / total_evidence[:, None]
    purity = inside_evidence / inside_mass - ring_evidence / ring_mass
    capture = capture.masked_fill(~candidate_valid, -torch.inf)
    purity = purity.masked_fill(~candidate_valid, -torch.inf)
    return capture, purity


@dataclass(frozen=True)
class ParetoSelection:
    selected_index: int
    switched: bool
    dominator_count: int


def pareto_guarded_selection(
    identity_scores: np.ndarray,
    capture_scores: np.ndarray,
    purity_scores: np.ndarray,
    candidate_indices: np.ndarray,
    control_local_index: int,
) -> ParetoSelection:
    """Select only a component-wise rank dominator of the control winner."""

    identity = np.asarray(identity_scores, dtype=np.float32)
    capture = np.asarray(capture_scores, dtype=np.float32)
    purity = np.asarray(purity_scores, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if (
        identity.ndim != 1
        or capture.shape != identity.shape
        or purity.shape != identity.shape
        or indices.shape != identity.shape
        or len(identity) == 0
        or not np.isfinite(identity).all()
        or not np.isfinite(capture).all()
        or not np.isfinite(purity).all()
        or len(np.unique(indices)) != len(indices)
        or not 0 <= control_local_index < len(identity)
    ):
        raise ValueError("Pareto selection inputs are invalid")
    valid = torch.ones((1, len(identity)), dtype=torch.bool)

    def rank(values: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(values[None])
        return within_bag_percentile_ranks(tensor, valid)[0].numpy()

    identity_rank = rank(identity)
    capture_rank = rank(capture)
    purity_rank = rank(purity)
    baseline = np.asarray(
        (
            identity_rank[control_local_index],
            capture_rank[control_local_index],
            purity_rank[control_local_index],
        ),
        dtype=np.float32,
    )
    components = np.stack((identity_rank, capture_rank, purity_rank), axis=1)
    weak = np.all(components >= baseline[None], axis=1)
    strict = np.any(components > baseline[None], axis=1)
    eligible = np.flatnonzero(weak & strict)
    if len(eligible) == 0:
        return ParetoSelection(
            selected_index=int(indices[control_local_index]),
            switched=False,
            dominator_count=0,
        )
    # Deterministic lexicographic maximum: balanced evidence, identity, then
    # smallest immutable candidate index (encoded as negative for max).
    best = max(
        eligible.tolist(),
        key=lambda row: (
            float(components[row].min()),
            float(identity_rank[row]),
            -int(indices[row]),
        ),
    )
    return ParetoSelection(
        selected_index=int(indices[best]),
        switched=best != control_local_index,
        dominator_count=int(len(eligible)),
    )


__all__ = [
    "CandidateSetTransformer",
    "HighResProposalMIL",
    "HighResProposalMILOutput",
    "ParetoSelection",
    "ResNet50FeaturePyramid",
    "aligned_view_consistency",
    "area_orthogonality_penalty",
    "attention_union_consistency",
    "candidate_capture_purity",
    "dual_stream_bag_probability",
    "image_label_proposal_loss",
    "masked_candidate_zone_descriptors",
    "pareto_guarded_selection",
    "project_candidate_supports",
    "top_instance_dropout_mask",
]
