from __future__ import annotations

"""HR-CBPMIL-IE+ model primitives.

This module intentionally accepts only radiographs, image labels, candidate
masks, validity flags and duplicate-cluster IDs.  Proposal source, prompt type,
SAM scores, coordinates and every spatial annotation are absent from the API.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import densenet121


@dataclass(frozen=True)
class HRCBPMILConfig:
    image_size: int = 640
    mask_size: int = 320
    grid_size: int = 160
    fpn_channels: int = 128
    descriptor_dim: int = 384
    hidden_dim: int = 256
    dropout: float = 0.10
    probability_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.image_size != 640 or self.mask_size != 320 or self.grid_size != 160:
            raise ValueError("HR-CBPMIL-IE+ fixes image/mask/grid sizes at 640/320/160")
        if self.fpn_channels != 128 or self.descriptor_dim != 384:
            raise ValueError("HR-CBPMIL-IE+ fixes FPN/descriptor widths at 128/384")


class _FPNRefine(nn.Sequential):
    def __init__(self, channels: int = 128) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )


class DenseNet121FPN(nn.Module):
    """Ten-class DenseNet121 with the exact stride-4 FPN required by the design."""

    def __init__(self, checkpoint_path: str | Path, *, channels: int = 128) -> None:
        super().__init__()
        backbone = densenet121(weights=None)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        if "classifier.weight" not in state or tuple(state["classifier.weight"].shape) != (10, 1024):
            raise RuntimeError("Checkpoint is not the frozen ten-class DenseNet121")
        backbone.classifier = nn.Linear(backbone.classifier.in_features, 10)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Ten-class checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self.features = backbone.features
        self.classifier10 = backbone.classifier
        self.lateral2 = nn.Conv2d(256, channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(512, channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(1024, channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(1024, channels, kernel_size=1)
        self.refine5 = _FPNRefine(channels)
        self.refine4 = _FPNRefine(channels)
        self.refine3 = _FPNRefine(channels)
        self.refine2 = _FPNRefine(channels)

    def train(self, mode: bool = True) -> "DenseNet121FPN":
        super().train(mode)
        if mode:
            # Freeze running statistics only. BN affine parameters remain trainable.
            for module in self.features.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.classifier10.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features.conv0(image)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        c2 = self.features.denseblock1(x)
        x = self.features.transition1(c2)
        c3 = self.features.denseblock2(x)
        x = self.features.transition2(c3)
        c4 = self.features.denseblock3(x)
        x = self.features.transition3(c4)
        c5 = self.features.denseblock4(x)

        pooled = F.adaptive_avg_pool2d(F.relu(self.features.norm5(c5)), 1).flatten(1)
        logits10 = self.classifier10(pooled)

        p5 = self.refine5(self.lateral5(c5))
        p4 = self.refine4(self.lateral4(c4) + F.interpolate(p5, c4.shape[-2:], mode="nearest"))
        p3 = self.refine3(self.lateral3(c3) + F.interpolate(p4, c3.shape[-2:], mode="nearest"))
        p2 = self.refine2(self.lateral2(c2) + F.interpolate(p3, c2.shape[-2:], mode="nearest"))
        return p2, logits10


def project_candidate_masks(candidate_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 320 masks to the 160 FPN grid, preserving fractional area and survival."""

    if candidate_masks.ndim != 4 or candidate_masks.shape[-2:] != (320, 320):
        raise ValueError("candidate_masks must have shape [B,N,320,320]")
    flat = candidate_masks.float().flatten(0, 1)[:, None]
    fractional = F.avg_pool2d(flat, kernel_size=2, stride=2)[:, 0]
    survival = F.max_pool2d(flat, kernel_size=2, stride=2)[:, 0] > 0
    shape = (*candidate_masks.shape[:2], 160, 160)
    return fractional.reshape(shape), survival.reshape(shape)


def adaptive_candidate_rings(
    survival_masks: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Adaptive exterior rings with the specified complement/global fallbacks."""

    if survival_masks.ndim != 4 or survival_masks.shape[-2:] != (160, 160):
        raise ValueError("survival_masks must have shape [B,N,160,160]")
    if candidate_valid.shape != survival_masks.shape[:2]:
        raise ValueError("candidate_valid does not align with masks")
    flat = survival_masks.flatten(0, 1).bool()
    valid = candidate_valid.flatten().bool()
    areas = flat.sum(dim=(-2, -1)).float()
    radii = torch.sqrt(areas / torch.pi)
    widths = torch.round(0.15 * radii).clamp(2, 8).to(torch.int64)
    rings = torch.zeros_like(flat)
    for width in range(2, 9):
        selected = valid & (widths == width)
        if selected.any():
            dilated = F.max_pool2d(
                flat[selected, None].float(),
                kernel_size=2 * width + 1,
                stride=1,
                padding=width,
            )[:, 0] > 0
            rings[selected] = dilated & ~flat[selected]
    missing = valid & (rings.sum(dim=(-2, -1)) < 1)
    if missing.any():
        complement = ~flat[missing]
        empty_complement = complement.sum(dim=(-2, -1)) < 1
        if empty_complement.any():
            complement[empty_complement] = True
        rings[missing] = complement
    return rings.reshape_as(survival_masks)


def _weighted_spatial_mean(
    feature: torch.Tensor,
    weights: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Compute the exact FP32 weighted mean and reject empty valid regions."""

    if weights.shape[:2] != candidate_valid.shape:
        raise ValueError(f"{name} weights do not align with candidate_valid")
    with torch.autocast(device_type=feature.device.type, enabled=False):
        feature32 = feature.float()
        weights32 = weights.float()
        mass = weights32.sum(dim=(-2, -1), keepdim=False)
        invalid_mass = candidate_valid & (mass <= 0)
        if invalid_mass.any():
            locations = torch.nonzero(invalid_mass, as_tuple=False)[:16].cpu().tolist()
            raise FloatingPointError(
                json.dumps(
                    {"name": f"{name}_mass", "nonpositive_valid_locations": locations},
                    sort_keys=True,
                )
            )
        numerator = torch.einsum("bchw,bnhw->bnc", feature32, weights32)
        output = numerator / mass.clamp_min(1.0e-6)[..., None]
        check_finite(name, output)
        return output


def check_finite(name: str, value: torch.Tensor) -> None:
    """Fail at the first non-finite intermediate with compact tensor statistics."""

    finite_mask = torch.isfinite(value)
    if bool(finite_mask.all()):
        return
    finite = value[finite_mask]
    stats: dict[str, object] = {
        "name": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "nan": int(torch.isnan(value).sum().item()),
        "posinf": int(torch.isposinf(value).sum().item()),
        "neginf": int(torch.isneginf(value).sum().item()),
    }
    if finite.numel():
        stats.update(
            {
                "finite_min": float(finite.min().item()),
                "finite_max": float(finite.max().item()),
                "finite_mean": float(finite.mean().item()),
            }
        )
    raise FloatingPointError(json.dumps(stats, sort_keys=True))


def cluster_balanced_detection(
    detection_logits: torch.Tensor,
    cluster_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cluster mass, within-cluster mass and candidate detection mass."""

    if detection_logits.shape != cluster_ids.shape or cluster_ids.shape != candidate_valid.shape:
        raise ValueError("Detection, cluster and validity tensors must align")
    batch, candidates = detection_logits.shape
    # Always compute the probability simplex in FP32. Under AMP the incoming
    # logits are FP16, while softmax/logsumexp may promote to FP32.
    with torch.autocast(device_type=detection_logits.device.type, enabled=False):
        logits32 = detection_logits.float()
        check_finite("detection_logits", logits32)
        cluster_mass = torch.zeros_like(logits32)
        within = torch.zeros_like(logits32)
        detection = torch.zeros_like(logits32)
        for batch_index in range(batch):
            valid_indices = torch.nonzero(candidate_valid[batch_index], as_tuple=False).flatten()
            if not len(valid_indices):
                raise ValueError("Every image must contain at least one valid candidate")
            ids = cluster_ids[batch_index, valid_indices]
            if (ids < 0).any():
                raise ValueError("Valid candidates require non-negative cluster IDs")
            unique_ids = torch.unique(ids, sorted=True)
            balanced_logits: list[torch.Tensor] = []
            member_sets: list[torch.Tensor] = []
            for cluster_id in unique_ids:
                members = valid_indices[ids == cluster_id]
                logits = logits32[batch_index, members]
                balanced_logits.append(
                    torch.logsumexp(logits, dim=0)
                    - torch.log(logits.new_tensor(float(len(members))))
                )
                member_sets.append(members)
            masses = torch.softmax(torch.stack(balanced_logits), dim=0)
            for mass, members in zip(masses, member_sets, strict=True):
                local = torch.softmax(logits32[batch_index, members], dim=0)
                cluster_mass[batch_index, members] = mass
                within[batch_index, members] = local
                detection[batch_index, members] = mass * local
        check_finite("cluster_mass", cluster_mass)
        check_finite("within_cluster_mass", within)
        check_finite("detection_mass", detection)
        simplex = detection.sum(dim=1)
        if not torch.allclose(simplex, torch.ones_like(simplex), atol=1.0e-5, rtol=0.0):
            raise FloatingPointError(
                json.dumps({"name": "detection_simplex", "sums": simplex.cpu().tolist()})
            )
        return cluster_mass, within, detection


class HRCBPMILIEPlus(nn.Module):
    def __init__(self, checkpoint_path: str | Path, config: HRCBPMILConfig | None = None) -> None:
        super().__init__()
        self.config = config or HRCBPMILConfig()
        self.backbone = DenseNet121FPN(checkpoint_path, channels=self.config.fpn_channels)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(self.config.descriptor_dim),
            nn.Linear(self.config.descriptor_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.classification_head = nn.Linear(self.config.hidden_dim, 1)
        self.detection_head = nn.Linear(self.config.hidden_dim, 1)
        self.dense_head = nn.Sequential(
            nn.Conv2d(self.config.fpn_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_valid: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p2, logits10 = self.backbone(image)
        if p2.shape[-2:] != (160, 160):
            raise RuntimeError(f"Unexpected P2 geometry: {tuple(p2.shape)}")
        with torch.autocast(device_type=image.device.type, enabled=False):
            fractional, survival = project_candidate_masks(candidate_masks)
            rings = adaptive_candidate_rings(survival, candidate_valid)
            fractional = fractional.float() * candidate_valid[..., None, None]
            rings = rings.float() * candidate_valid[..., None, None]
            inside = _weighted_spatial_mean(
                p2, fractional, candidate_valid, name="candidate_inside_descriptor"
            )
            ring = _weighted_spatial_mean(
                p2, rings, candidate_valid, name="candidate_ring_descriptor"
            )
            descriptor = torch.cat((inside, ring, inside - ring), dim=-1)
            check_finite("candidate_descriptor", descriptor)
        encoded = self.candidate_encoder(descriptor)
        classification_logits = self.classification_head(encoded).squeeze(-1)
        detection_logits = self.detection_head(encoded).squeeze(-1)
        classification_logits = classification_logits.masked_fill(~candidate_valid, 0.0)
        detection_logits = detection_logits.masked_fill(~candidate_valid, 0.0)
        with torch.autocast(device_type=image.device.type, enabled=False):
            cluster_mass, within_mass, detection_mass = cluster_balanced_detection(
                detection_logits, cluster_ids, candidate_valid
            )
            classification_logits32 = classification_logits.float()
            check_finite("classification_logits", classification_logits32)
            instance_probability = torch.sigmoid(classification_logits32)
            image_probability = (instance_probability * detection_mass).sum(dim=1)
            check_finite("instance_probability", instance_probability)
            check_finite("image_probability", image_probability)

        # Convolutions remain AMP-eligible.  All following dense reductions and
        # normalizations cross the explicit FP32 boundary before use.
        dense_logits = self.dense_head(p2)[:, 0]
        with torch.autocast(device_type=image.device.type, enabled=False):
            dense_logits = dense_logits.float()
            check_finite("dense_logits_after_head", dense_logits)
            dense_inside = _weighted_spatial_mean(
                dense_logits[:, None], fractional, candidate_valid, name="dense_inside"
            )[..., 0]
            dense_ring = _weighted_spatial_mean(
                dense_logits[:, None], rings, candidate_valid, name="dense_ring"
            )[..., 0]
        return {
            "logits10": logits10,
            "classification_logits": classification_logits,
            "detection_logits": detection_logits,
            "detection_mass": detection_mass,
            "instance_probability": instance_probability,
            "image_probability": image_probability,
            "dense_logits": dense_logits,
            "dense_inside": dense_inside,
            "dense_ring": dense_ring,
        }


def intra_loss_weight(epoch_number: int) -> float:
    if epoch_number <= 3:
        return 0.0
    if epoch_number == 4:
        return 0.0625
    if epoch_number == 5:
        return 0.125
    if epoch_number == 6:
        return 0.1875
    return 0.25


def hr_cbpmil_loss(
    output: dict[str, torch.Tensor],
    binary_labels: torch.Tensor,
    class10_labels: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    epoch_number: int,
    epsilon: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    device_type = output["image_probability"].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        labels = binary_labels.float().reshape(-1)
        tumor = labels > 0.5
        normal = ~tumor
        probability = output["image_probability"].float()
        check_finite("pmil_probability", probability)
        if bool(((probability < 0.0) | (probability > 1.0 + 1.0e-6)).any()):
            raise FloatingPointError("PMIL probability is outside [0,1]")
        probability_safe = probability.clamp(epsilon, 1.0 - epsilon)
        pmil = -(
            labels * probability_safe.log()
            + (1.0 - labels) * torch.log1p(-probability_safe)
        ).mean()

        dense_logits = output["dense_logits"].float().flatten(1)
        check_finite("dense_logits_loss_input", dense_logits)
        dense_per_image = dense_logits.new_zeros((len(labels),))
        if tumor.any():
            tumor_logits = dense_logits[tumor]
            top4 = torch.topk(tumor_logits, k=4, dim=1).values.mean(dim=1)
            top16 = torch.topk(tumor_logits, k=16, dim=1).values.mean(dim=1)
            check_finite("dense_top4", top4)
            check_finite("dense_top16", top16)
            dense_per_image[tumor] = 0.5 * F.softplus(-top4) + 0.5 * F.softplus(-top16)
        if normal.any():
            dense_per_image[normal] = F.softplus(dense_logits[normal]).mean(dim=1)
        dense_loss = dense_per_image.mean()

        candidate_negative = dense_logits.new_zeros(())
        if normal.any():
            mask = candidate_valid[normal]
            negative_logits = output["classification_logits"][normal].float()
            check_finite("candidate_negative_logits", negative_logits)
            candidate_negative = F.softplus(negative_logits)[mask].mean()

        intra = dense_logits.new_zeros(())
        if tumor.any() and intra_loss_weight(epoch_number) > 0:
            valid = candidate_valid[tumor]
            classification_logits = output["classification_logits"][tumor].float()
            detection_mass = output["detection_mass"][tumor].float()
            check_finite("intra_classification_logits", classification_logits)
            check_finite("intra_detection_mass", detection_mass)
            log_responsibility = F.logsigmoid(classification_logits) + torch.log(
                detection_mass.clamp_min(1.0e-12)
            )
            log_responsibility = log_responsibility.masked_fill(~valid, -torch.inf)
            weights = torch.softmax(log_responsibility, dim=1).detach()
            check_finite("intra_responsibility", weights)
            responsibility_sums = weights.sum(dim=1)
            if not torch.allclose(
                responsibility_sums,
                torch.ones_like(responsibility_sums),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise FloatingPointError("Intra responsibilities do not sum to one")
            dense_inside = output["dense_inside"][tumor].float()
            dense_ring = output["dense_ring"][tumor].float()
            check_finite("intra_dense_inside", dense_inside)
            check_finite("intra_dense_ring", dense_ring)
            delta = dense_inside - dense_ring
            check_finite("intra_delta", delta)
            terms = F.softplus(-delta)
            check_finite("intra_per_candidate", terms)
            intra = (weights * terms * valid).sum(dim=1).mean()

        aux10 = dense_logits.new_zeros(())
        if epoch_number >= 3:
            logits10 = output["logits10"].float()
            check_finite("aux10_logits", logits10)
            aux10 = F.cross_entropy(logits10, class10_labels.long())
        total = (
            pmil
            + 0.5 * dense_loss
            + 0.25 * candidate_negative
            + intra_loss_weight(epoch_number) * intra
            + (0.1 * aux10 if epoch_number >= 3 else 0.0)
        )
        losses = {
            "total": total,
            "pmil": pmil,
            "dense": dense_loss,
            "candidate_negative": candidate_negative,
            "intra": intra,
            "aux10": aux10,
        }
        for name, value in losses.items():
            check_finite(f"loss_{name}", value)
        return losses
