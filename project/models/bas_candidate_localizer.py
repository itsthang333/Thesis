from __future__ import annotations

"""Image-label-only BAS localization and immutable-candidate scoring primitives.

The module deliberately has no dataset, validation-mask, subgroup, or evaluator
API.  It adapts the activation-suppression mechanism of Wu et al. (CVPR 2022)
to a two-class radiograph classifier, then converts the resulting tumor
activation map into coverage/purity evidence for an already frozen candidate
gallery.
"""

import copy
from dataclasses import dataclass
from typing import Literal, Mapping, NamedTuple, Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
except Exception:  # pragma: no cover - optional in lightweight audit envs
    ResNet50_Weights = None
    resnet50 = None


@dataclass(frozen=True)
class BASLossConfig:
    area_weight: float = 1.2
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.area_weight < 0:
            raise ValueError("area_weight must be nonnegative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class ForegroundControlLossConfig:
    """Weights for the continuous foreground-control replacement probe.

    The published chest-X-ray objective uses the target-class activation kept
    by the foreground map instead of a hard-gated background ratio.  Detaching
    the full-image reference preserves the matched BAS probe semantics: the
    localization branch must explain a fixed classifier signal rather than
    moving its denominator.
    """

    foreground_control_weight: float = 1.5
    area_weight: float = 1.2
    reference_ratio: float = 0.5
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.foreground_control_weight <= 0:
            raise ValueError("foreground_control_weight must be positive")
        if self.area_weight < 0:
            raise ValueError("area_weight must be nonnegative")
        if not 0.0 <= self.reference_ratio <= 1.0:
            raise ValueError("reference_ratio must lie in [0,1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


class BASForwardOutput(NamedTuple):
    class_logits: torch.Tensor
    foreground_logits: torch.Tensor
    class_activation_maps: torch.Tensor
    localization_maps: torch.Tensor
    background_logits: torch.Tensor


ClassifierOutputActivation = Literal["relu", "softplus"]


def classifier_output_activation(name: ClassifierOutputActivation) -> nn.Module:
    """Return a nonnegative class-map activation with explicit semantics.

    ``relu`` preserves the official multi-class BAS implementation. ``softplus``
    is the single bounded binary-transfer correction: it keeps class maps
    nonnegative for the activation-ratio objective but cannot enter a dead
    negative-preactivation state with exactly zero gradient.
    """

    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "softplus":
        return nn.Softplus(beta=1.0, threshold=20.0)
    raise ValueError(f"unsupported BAS classifier output activation: {name}")


def _gather_class_map(maps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if maps.ndim != 4:
        raise ValueError("maps must have shape [B,C,H,W]")
    labels = labels.reshape(-1).long()
    if labels.shape[0] != maps.shape[0]:
        raise ValueError("labels must align with the map batch")
    if torch.any(labels < 0) or torch.any(labels >= maps.shape[1]):
        raise ValueError("labels lie outside the map class dimension")
    rows = torch.arange(maps.shape[0], device=maps.device)
    return maps[rows, labels][:, None]


def bas_activation_suppression_loss(
    output: BASForwardOutput,
    labels: torch.Tensor,
    *,
    config: BASLossConfig = BASLossConfig(),
) -> torch.Tensor:
    """Return the official BAS background-ratio plus area constraint.

    The classifier head is ReLU-bounded, matching the primary implementation.
    When the erased/background activation is not below the full activation, the
    ratio term is set to zero exactly as in the official ResNet code; the area
    constraint still supplies a localization gradient.
    """

    labels = labels.reshape(-1).long()
    for name, logits in (
        ("class_logits", output.class_logits),
        ("background_logits", output.background_logits),
    ):
        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise ValueError(f"{name} must have shape [B,C]")
    if output.class_logits.shape != output.background_logits.shape:
        raise ValueError("full/background logits must share shape")
    if output.localization_maps.ndim != 4 or output.localization_maps.shape[:2] != (
        labels.shape[0],
        1,
    ):
        raise ValueError("localization_maps must have shape [B,1,H,W]")

    rows = torch.arange(labels.shape[0], device=labels.device)
    # The official epsilon is below float16's normal range.  Keep the ratio and
    # area arithmetic in float32 even when the surrounding runner uses AMP.
    full = output.class_logits[rows, labels].float()
    background = output.background_logits[rows, labels].float()
    ratio = background / (full.detach() + config.epsilon)
    ratio = torch.where(background < full.detach(), ratio, torch.zeros_like(ratio))
    area = output.localization_maps.float().flatten(start_dim=1).mean(dim=1)
    loss = ratio + config.area_weight * area
    if not torch.isfinite(loss).all():
        raise RuntimeError("BAS loss is non-finite")
    return loss.mean()


def foreground_control_area_loss(
    output: BASForwardOutput,
    labels: torch.Tensor,
    *,
    config: ForegroundControlLossConfig = ForegroundControlLossConfig(),
) -> torch.Tensor:
    """Continuous foreground-control ratio plus area constraint.

    This is the bounded B2.2 scientific delta.  Unlike the transferred BAS
    background ratio, it has no ``background >= full -> 0`` branch.  At an
    empty map its derivative is spatially selective: cells whose target-class
    evidence exceeds the area/control balance are pushed upward, while weak
    cells are pushed downward.
    """

    labels = labels.reshape(-1).long()
    for name, logits in (
        ("class_logits", output.class_logits),
        ("foreground_logits", output.foreground_logits),
    ):
        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise ValueError(f"{name} must have shape [B,C]")
    if output.class_logits.shape != output.foreground_logits.shape:
        raise ValueError("full/foreground logits must share shape")
    if output.localization_maps.ndim != 4 or output.localization_maps.shape[:2] != (
        labels.shape[0],
        1,
    ):
        raise ValueError("localization_maps must have shape [B,1,H,W]")

    rows = torch.arange(labels.shape[0], device=labels.device)
    full = output.class_logits[rows, labels].float().detach()
    foreground = output.foreground_logits[rows, labels].float()
    ratio = foreground / (full + config.epsilon)
    control = config.reference_ratio - ratio
    area = output.localization_maps.float().flatten(start_dim=1).mean(dim=1)
    loss = (
        config.foreground_control_weight * control
        + config.area_weight * area
    )
    if not torch.isfinite(loss).all():
        raise RuntimeError("foreground-control loss is non-finite")
    return loss.mean()


class BASResNet50Localizer(nn.Module):
    """Two-class ResNet-50 BAS model with an exact-resolution erasing path.

    The official ResNet BAS implementation keeps stage 3 at output stride 8,
    inserts a 2x max-pool, and keeps stage 4 at output stride 16.  The shadow
    erasing branch is refreshed from the live weights on every forward pass;
    its parameters are frozen while gradients still reach the localization
    map.  This reproduces the semantics of the official per-forward deep copy
    without allocating a new block for every batch.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        backbone_state_dict: Mapping[str, torch.Tensor] | None = None,
        num_classes: int = 2,
        classifier_activation: ClassifierOutputActivation = "relu",
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("BTXRD BAS localizer is fixed to normal/tumor classes")
        if classifier_activation not in ("relu", "softplus"):
            raise ValueError("classifier_activation must be relu or softplus")
        if resnet50 is None:
            raise RuntimeError("torchvision ResNet-50 is unavailable")
        if pretrained and backbone_state_dict is not None:
            raise ValueError("choose torchvision weights or an explicit state dict")
        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained and ResNet50_Weights is not None
            else None
        )
        backbone = resnet50(weights=weights)
        if backbone_state_dict is not None:
            backbone.load_state_dict(backbone_state_dict, strict=True)
        self._configure_large_feature_map(backbone)
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
        self.classifier_activation = classifier_activation
        self.classifier_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, num_classes, kernel_size=1),
            classifier_output_activation(classifier_activation),
        )
        self.localization_head = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.background_layer4 = copy.deepcopy(self.layer4)
        self.background_classifier_head = copy.deepcopy(self.classifier_head)
        self._initialize_new_heads()
        self.sync_background_branch()

    @staticmethod
    def _configure_large_feature_map(backbone: nn.Module) -> None:
        """Match the official BAS ResNet output strides using pretrained blocks."""

        for stage in (backbone.layer3, backbone.layer4):
            first = stage[0]
            first.conv2.stride = (1, 1)
            first.downsample[0].stride = (1, 1)

    def _initialize_new_heads(self) -> None:
        for module in (self.classifier_head, self.localization_head):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def sync_background_branch(self) -> None:
        self.background_layer4.load_state_dict(self.layer4.state_dict())
        self.background_classifier_head.load_state_dict(
            self.classifier_head.state_dict()
        )
        self.background_layer4.requires_grad_(False).eval()
        self.background_classifier_head.requires_grad_(False).eval()

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> BASForwardOutput:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        labels = labels.reshape(-1).long()
        if labels.shape[0] != images.shape[0]:
            raise ValueError("labels must align with images")
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        stage3 = self.layer3(x)

        pooled_stage3 = F.max_pool2d(stage3, kernel_size=2)
        class_maps = self.classifier_head(self.layer4(pooled_stage3))
        class_logits = F.adaptive_avg_pool2d(class_maps, 1).flatten(1)
        all_localization = self.localization_head(stage3)
        localization = _gather_class_map(all_localization, labels)

        # Refresh before every call: unlike an epoch-stale teacher, this mirrors
        # the official deep-copied branch at the current optimizer step.
        self.sync_background_branch()
        self.background_layer4.train(self.training)
        self.background_classifier_head.train(self.training)
        erased = F.max_pool2d(stage3.detach() * (1.0 - localization), kernel_size=2)
        background_maps = self.background_classifier_head(
            self.background_layer4(erased)
        )
        background_logits = F.adaptive_avg_pool2d(background_maps, 1).flatten(1)

        resized_localization = F.interpolate(
            localization,
            size=class_maps.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        foreground_logits = F.adaptive_avg_pool2d(
            class_maps * resized_localization,
            1,
        ).flatten(1)
        return BASForwardOutput(
            class_logits=class_logits,
            foreground_logits=foreground_logits,
            class_activation_maps=class_maps,
            localization_maps=localization,
            background_logits=background_logits,
        )

    @torch.no_grad()
    def tumor_activation(self, images: torch.Tensor) -> torch.Tensor:
        """Return the class-1 localization map without requiring spatial labels."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        maps = self.localization_head(x)[:, 1:2]
        if not torch.isfinite(maps).all():
            raise RuntimeError("tumor activation is non-finite")
        return maps

    @torch.no_grad()
    def classify_and_tumor_activation(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and class-1 map without the training-only branch."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        stage3 = self.layer3(x)
        maps = self.localization_head(stage3)[:, 1:2]
        class_maps = self.classifier_head(
            self.layer4(F.max_pool2d(stage3, kernel_size=2))
        )
        logits = F.adaptive_avg_pool2d(class_maps, 1).flatten(1)
        if not torch.isfinite(logits).all() or not torch.isfinite(maps).all():
            raise RuntimeError("BAS inference output is non-finite")
        return logits, maps


def minmax_normalize_activation(activation: torch.Tensor) -> torch.Tensor:
    """Per-image BAS normalization used by the official localization path."""

    if activation.ndim != 4 or activation.shape[1] != 1:
        raise ValueError("activation must have shape [B,1,H,W]")
    if not torch.isfinite(activation).all():
        raise ValueError("activation must be finite")
    flat = activation.flatten(start_dim=2)
    lower = flat.amin(dim=2, keepdim=True).unsqueeze(-1)
    upper = flat.amax(dim=2, keepdim=True).unsqueeze(-1)
    return (activation - lower) / (upper - lower).clamp_min(1.0e-10)


def candidate_activation_evidence(
    activation: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return activation coverage, purity, and their harmonic mean per candidate."""

    if activation.ndim != 4 or activation.shape[1] != 1:
        raise ValueError("activation must have shape [B,1,H,W]")
    if candidate_masks.ndim != 4:
        raise ValueError("candidate_masks must have shape [B,N,H,W]")
    if candidate_masks.shape[0] != activation.shape[0]:
        raise ValueError("activation and candidates must share a batch")
    if candidate_valid.shape != candidate_masks.shape[:2]:
        raise ValueError("candidate_valid must align with candidates")
    if not torch.isfinite(activation).all() or not torch.isfinite(candidate_masks).all():
        raise ValueError("activation/candidate masks must be finite")

    activation = minmax_normalize_activation(activation)
    batch, candidates = candidate_masks.shape[:2]
    masks = F.interpolate(
        candidate_masks.float().reshape(
            batch * candidates,
            1,
            candidate_masks.shape[-2],
            candidate_masks.shape[-1],
        ),
        size=activation.shape[-2:],
        mode="area",
    ).reshape(batch, candidates, *activation.shape[-2:]).clamp(0.0, 1.0)
    valid = candidate_valid.bool()
    overlap = (masks * activation).sum(dim=(-2, -1))
    activation_mass = activation.sum(dim=(-2, -1)).clamp_min(1.0e-8)
    mask_mass = masks.sum(dim=(-2, -1)).clamp_min(1.0e-8)
    coverage = overlap / activation_mass
    purity = overlap / mask_mass
    harmonic = 2.0 * coverage * purity / (coverage + purity).clamp_min(1.0e-8)
    zeros = torch.zeros_like(coverage)
    coverage = torch.where(valid, coverage, zeros)
    purity = torch.where(valid, purity, zeros)
    harmonic = torch.where(valid, harmonic, zeros)
    return coverage, purity, harmonic


def within_bag_percentile_ranks(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Tie-aware [0,1] within-bag ranks with deterministic invalid zeros."""

    if scores.ndim != 2 or candidate_valid.shape != scores.shape:
        raise ValueError("scores/validity must share shape [B,N]")
    if not torch.isfinite(scores[candidate_valid.bool()]).all():
        raise ValueError("valid scores must be finite")
    valid = candidate_valid.bool()
    result = torch.zeros_like(scores)
    for row in range(scores.shape[0]):
        indices = torch.nonzero(valid[row], as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            raise ValueError("every bag must contain a valid candidate")
        values = scores[row, indices]
        if indices.numel() == 1:
            result[row, indices] = 1.0
            continue
        less = (values[:, None] > values[None, :]).sum(dim=1).to(values.dtype)
        equal = (values[:, None] == values[None, :]).sum(dim=1).to(values.dtype)
        ranks = (less + 0.5 * (equal - 1.0)) / float(indices.numel() - 1)
        result[row, indices] = ranks
    return result


def equal_rank_fusion(
    baseline_scores: torch.Tensor,
    activation_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Fixed 1:1 rank fusion; no scale, area, subgroup, or GT-dependent weight."""

    return equal_rank_aggregate(
        (baseline_scores, activation_scores),
        candidate_valid,
    )


def equal_rank_aggregate(
    score_vectors: Sequence[torch.Tensor],
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Unweighted Borda mean of two or more within-bag percentile ranks."""

    if len(score_vectors) < 2:
        raise ValueError("rank aggregation requires at least two score vectors")
    expected = candidate_valid.shape
    if candidate_valid.ndim != 2 or any(value.shape != expected for value in score_vectors):
        raise ValueError("all rank-aggregation scores must share shape with validity")
    ranks = [
        within_bag_percentile_ranks(value, candidate_valid)
        for value in score_vectors
    ]
    return torch.stack(ranks, dim=0).mean(dim=0)


__all__ = [
    "BASForwardOutput",
    "BASLossConfig",
    "BASResNet50Localizer",
    "ClassifierOutputActivation",
    "ForegroundControlLossConfig",
    "bas_activation_suppression_loss",
    "candidate_activation_evidence",
    "classifier_output_activation",
    "equal_rank_fusion",
    "equal_rank_aggregate",
    "foreground_control_area_loss",
    "minmax_normalize_activation",
    "within_bag_percentile_ranks",
]
