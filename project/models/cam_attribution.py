from __future__ import annotations

"""Published CAM-family attribution rules on a common DenseNet target layer."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .layercam import LayerCAMOutput


@dataclass
class _State:
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None


def normalize_and_resize(cam: torch.Tensor, input_size: tuple[int, int]) -> torch.Tensor:
    cam = F.relu(cam)
    batch = cam.shape[0]
    flat = cam.reshape(batch, -1)
    low = flat.min(dim=1).values.reshape(batch, 1, 1, 1)
    high = flat.max(dim=1).values.reshape(batch, 1, 1, 1)
    cam = (cam - low) / (high - low + 1e-8)
    return F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)


def gradcam_map(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Selvaraju et al.: global-average gradient weights, then ReLU."""
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    return F.relu((weights * activations).sum(dim=1, keepdim=True))


def gradcam_plus_plus_map(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Chattopadhyay et al. Grad-CAM++ spatial alpha weighting."""
    gradients2 = gradients.pow(2)
    gradients3 = gradients2 * gradients
    denominator = 2.0 * gradients2 + (activations * gradients3).sum(dim=(2, 3), keepdim=True)
    alpha = gradients2 / (denominator + 1e-8)
    positive_gradients = F.relu(gradients)
    weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
    return F.relu((weights * activations).sum(dim=1, keepdim=True))


def linear_cam_map(activations: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """Zhou et al. CAM: class-head weights linearly combine final feature maps."""
    if class_weights.ndim != 2 or class_weights.shape[0] != activations.shape[0]:
        raise ValueError("class_weights must have shape [B,C]")
    return F.relu((activations * class_weights[:, :, None, None]).sum(dim=1, keepdim=True))


class FinalLayerCAMFamily:
    """CAM, Grad-CAM, or Grad-CAM++ with one shared DenseNet final layer.

    LayerCAM remains implemented separately because its published rule is
    spatially weighted and this thesis fuses three DenseNet stages.
    """

    def __init__(self, model: torch.nn.Module, method: str, device: torch.device | None = None) -> None:
        if method not in {"cam", "gradcam", "gradcam_plus_plus"}:
            raise ValueError("method must be cam, gradcam, or gradcam_plus_plus")
        self.model = model
        self.method = method
        self.device = device or next(model.parameters()).device
        self.state = _State()
        target = model.features.norm5
        self.handles = [
            target.register_forward_hook(self._forward_hook),
            target.register_full_backward_hook(self._backward_hook),
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _forward_hook(self, module, inputs, output) -> None:
        self.state.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.state.gradients = grad_output[0].detach()

    def _reset(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.state.activations = None
        self.state.gradients = None

    def _score(self, logits: torch.Tensor, class_index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(class_index, torch.Tensor):
            indices = torch.arange(logits.shape[0], device=logits.device)
            return logits[indices, class_index].sum()
        return logits[:, class_index].sum()

    def cam_for_class(self, input_tensor: torch.Tensor, class_index: int | torch.Tensor) -> LayerCAMOutput:
        self._reset()
        logits = self.model(input_tensor)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if self.state.activations is None:
            raise RuntimeError("target-layer hook did not capture activations")
        if self.method == "cam":
            activations = F.relu(self.state.activations)
            if isinstance(class_index, torch.Tensor):
                weights = self.model.classifier.weight[class_index]
            else:
                weights = self.model.classifier.weight[class_index].unsqueeze(0).expand(logits.shape[0], -1)
            raw = linear_cam_map(activations, weights)
        else:
            self._score(logits, class_index).backward()
            if self.state.gradients is None:
                raise RuntimeError("target-layer hook did not capture gradients")
            if self.method == "gradcam":
                raw = gradcam_map(self.state.activations, self.state.gradients)
            else:
                raw = gradcam_plus_plus_map(self.state.activations, self.state.gradients)
        cam = normalize_and_resize(raw, input_tensor.shape[-2:]).squeeze(1)
        return LayerCAMOutput(logits=logits, cam=cam)

    def cams_for_active_classes(
        self,
        input_tensor: torch.Tensor,
        class_weights: Sequence[float],
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, list[np.ndarray], list[float], list[int]]:
        active = [index for index, weight in enumerate(class_weights) if weight >= confidence_threshold]
        if not active:
            active = [int(np.argmax(class_weights))]
        logits = None
        cams: list[np.ndarray] = []
        weights: list[float] = []
        for index in active:
            output = self.cam_for_class(input_tensor, index)
            logits = output.logits
            cams.append(output.cam[0].detach().cpu().numpy())
            weights.append(float(class_weights[index]))
        assert logits is not None
        return logits, cams, weights, active

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None) -> LayerCAMOutput:
        if class_index is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                class_index = int(logits.argmax(dim=1).item())
        return self.cam_for_class(input_tensor, class_index)
