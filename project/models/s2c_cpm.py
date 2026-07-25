from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .classifier import (
    DenseNet121_Weights,
    densenet121,
    load_radimagenet_densenet121_state_dict,
)
from .layercam import LayerCAMOutput


class DenseNet121S2CCPMClassifier(nn.Module):
    """DenseNet classifier with a deep stride-8 CAM head.

    DenseNet's stride-8 ``denseblock2`` tensor preserves small spatial
    structures, while the final tensor supplies deeper semantics. Both are
    projected to a shared feature space and fused at stride 8. Classification
    is global average pooling over the resulting tumor CAM, matching the
    classifier/CAM coupling required by S2C's CPM objective.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        feature_channels: int = 256,
        radimagenet_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")

        if radimagenet_checkpoint is not None:
            backbone = densenet121(weights=None)
            state_dict = load_radimagenet_densenet121_state_dict(
                radimagenet_checkpoint
            )
            missing, unexpected = backbone.features.load_state_dict(
                state_dict,
                strict=False,
            )
            if missing:
                raise RuntimeError(
                    f"RadImageNet checkpoint missing expected keys: {missing[:5]}"
                )
            print(
                f"Loaded RadImageNet backbone from {radimagenet_checkpoint} "
                f"({len(state_dict)} tensors, "
                f"{len(unexpected)} unexpected keys ignored)"
            )
        elif pretrained and DenseNet121_Weights is not None:
            backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            backbone = densenet121(weights=None)

        self.features = backbone.features
        self.low_projection = nn.Conv2d(
            512,
            feature_channels,
            kernel_size=1,
            bias=False,
        )
        self.high_projection = nn.Conv2d(
            backbone.classifier.in_features,
            feature_channels,
            kernel_size=1,
            bias=False,
        )
        self.cam_head = nn.Conv2d(
            feature_channels,
            1,
            kernel_size=1,
            bias=False,
        )
        nn.init.xavier_uniform_(self.low_projection.weight)
        nn.init.xavier_uniform_(self.high_projection.weight)
        nn.init.xavier_uniform_(self.cam_head.weight)

        self.feature_channels = int(feature_channels)
        self.feature_stride = 8

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_spatial: bool = False,
    ):
        low = None
        with torch.cuda.amp.autocast(enabled=False):
            features = x.float()
            for name, module in self.features.named_children():
                features = module(features)
                if name == "denseblock2":
                    low = torch.relu(features)
            high = torch.relu(features)
            if low is None:
                raise RuntimeError("DenseNet denseblock2 feature was not found")

            fused = self.low_projection(low)
            high_semantics = F.interpolate(
                self.high_projection(high),
                size=low.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = torch.relu(fused + high_semantics)
            cam_logits = self.cam_head(fused)
            logits = F.adaptive_avg_pool2d(cam_logits, output_size=1).flatten(1)

        if return_spatial:
            return logits, fused, cam_logits
        return logits


def normalized_foreground_cam(
    cam_logits: torch.Tensor,
    *,
    output_size: tuple[int, int] | None = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Return S2C-style per-image max-normalized positive CAMs."""

    if cam_logits.ndim != 4 or cam_logits.shape[1] != 1:
        raise ValueError("cam_logits must have shape [B,1,H,W]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    cam = torch.relu(cam_logits.float())
    maxima = F.adaptive_max_pool2d(cam, output_size=1)
    cam = cam / (maxima + epsilon)
    if output_size is not None:
        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
    return cam


def cpm_cross_entropy_loss(
    cam_logits: torch.Tensor,
    foreground_masks: torch.Tensor,
    image_labels: torch.Tensor,
) -> torch.Tensor:
    """Binary CPM loss with class order ``[tumor, background]``.

    ``foreground_masks`` must be generated without polygon supervision. Known
    normal images are forced to all-background even if a caller accidentally
    supplies non-empty masks.
    """

    if foreground_masks.ndim == 3:
        foreground_masks = foreground_masks.unsqueeze(1)
    if foreground_masks.ndim != 4 or foreground_masks.shape[1] != 1:
        raise ValueError("foreground_masks must have shape [B,1,H,W]")
    labels = image_labels.reshape(-1)
    if labels.shape[0] != cam_logits.shape[0]:
        raise ValueError("image_labels batch does not match cam_logits")
    if foreground_masks.shape[0] != cam_logits.shape[0]:
        raise ValueError("foreground_masks batch does not match cam_logits")

    cam = normalized_foreground_cam(
        cam_logits,
        output_size=tuple(int(value) for value in foreground_masks.shape[-2:]),
    )
    class_scores = torch.cat((cam, 1.0 - cam), dim=1)
    positive_images = labels.to(device=cam.device) > 0.5
    masks = foreground_masks.to(device=cam.device) > 0.5
    masks = masks & positive_images[:, None, None, None]
    targets = torch.ones(
        masks.shape[0],
        masks.shape[2],
        masks.shape[3],
        dtype=torch.long,
        device=cam.device,
    )
    targets[masks[:, 0]] = 0
    return F.cross_entropy(class_scores, targets)


class S2CCPMDirectCAM:
    """Adapter exposing the trained CPM CAM through the LayerCAM call surface."""

    def __init__(
        self,
        model: DenseNet121S2CCPMClassifier,
        *,
        scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    ) -> None:
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")
        self.model = model
        self.scales = tuple(float(scale) for scale in scales)

    def close(self) -> None:
        return None

    @torch.inference_mode()
    def cam_for_class(
        self,
        input_tensor: torch.Tensor,
        class_index: int | torch.Tensor,
    ) -> LayerCAMOutput:
        if isinstance(class_index, torch.Tensor):
            if bool((class_index != 0).any()):
                raise ValueError("The binary CPM classifier has only class index 0")
        elif int(class_index) != 0:
            raise ValueError("The binary CPM classifier has only class index 0")
        target_size = tuple(int(value) for value in input_tensor.shape[-2:])
        accumulated = None
        reference_logits = None
        for scale in self.scales:
            scaled = (
                input_tensor
                if scale == 1.0
                else F.interpolate(
                    input_tensor,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            logits, _features, cam_logits = self.model(
                scaled,
                return_spatial=True,
            )
            if scale == 1.0:
                reference_logits = logits
            cam = normalized_foreground_cam(
                cam_logits,
                output_size=target_size,
            )
            accumulated = cam if accumulated is None else accumulated + cam
        assert accumulated is not None
        maxima = F.adaptive_max_pool2d(accumulated, output_size=1)
        cam = accumulated / (maxima + 1e-5)
        if reference_logits is None:
            reference_logits = self.model(input_tensor)
        return LayerCAMOutput(
            logits=reference_logits,
            cam=cam[:, 0],
        )

    def cams_for_active_classes(
        self,
        input_tensor: torch.Tensor,
        class_weights,
        confidence_threshold: float = 0.5,
    ):
        weights = [float(value) for value in class_weights]
        if len(weights) != 1:
            raise ValueError("The direct CPM CAM backend requires one tumor class")
        output = self.cam_for_class(input_tensor, 0)
        return (
            output.logits,
            [output.cam[0].detach().cpu().numpy()],
            [weights[0]],
            [0],
        )
