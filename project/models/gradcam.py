from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GradCAMOutput:
    logits: torch.Tensor
    cam: torch.Tensor


class GradCAM:
    """Manual Grad-CAM implementation for a single target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device: torch.device | None = None) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _resolve_logits(self, outputs):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def _compute_cam(self, input_tensor: torch.Tensor, class_index: int) -> GradCAMOutput:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_tensor)
        logits = self._resolve_logits(outputs)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if logits.shape[1] == 1:
            selected_scores = logits[:, 0]
        else:
            selected_scores = logits[:, class_index]

        selected_scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        return GradCAMOutput(logits=logits, cam=cam.squeeze(1))

    def cam_for_class(self, input_tensor: torch.Tensor, class_index: int) -> GradCAMOutput:
        return self._compute_cam(input_tensor, class_index=class_index)

    def cams_for_classes(self, input_tensor: torch.Tensor, class_indices: Sequence[int]) -> tuple[torch.Tensor, list[GradCAMOutput]]:
        outputs: list[GradCAMOutput] = []
        logits: torch.Tensor | None = None
        for class_index in class_indices:
            result = self._compute_cam(input_tensor, class_index=class_index)
            logits = result.logits
            outputs.append(result)
        if logits is None:
            raise RuntimeError("No class indices were provided to GradCAM.cams_for_classes.")
        return logits, outputs

    def __call__(self, input_tensor: torch.Tensor, class_index: int | torch.Tensor | None = None) -> GradCAMOutput:
        if class_index is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = self._resolve_logits(outputs)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                resolved_index = int(logits.argmax(dim=1).item()) if logits.shape[1] > 1 else 0
            return self._compute_cam(input_tensor, class_index=resolved_index)
        if isinstance(class_index, torch.Tensor):
            class_index = int(class_index.item())
        return self._compute_cam(input_tensor, class_index=class_index)


def aggregate_cams(cams: Sequence[np.ndarray], weights: Sequence[float] | None = None, mode: str = "weighted_mean") -> np.ndarray:
    if not cams:
        raise ValueError("aggregate_cams requires at least one CAM.")

    stacked = np.stack([np.asarray(cam, dtype=np.float32) for cam in cams], axis=0)
    stacked = stacked - stacked.min(axis=(1, 2), keepdims=True)
    stacked = stacked / (stacked.max(axis=(1, 2), keepdims=True) + 1e-8)

    if mode == "max":
        return stacked.max(axis=0)

    if weights is None:
        weights_array = np.ones(stacked.shape[0], dtype=np.float32)
    else:
        weights_array = np.asarray(weights, dtype=np.float32)
        if weights_array.shape[0] != stacked.shape[0]:
            raise ValueError("weights must match the number of CAMs.")

    weights_array = np.clip(weights_array, 0.0, None)
    if float(weights_array.sum()) == 0.0:
        weights_array = np.ones_like(weights_array)
    weights_array = weights_array / weights_array.sum()
    aggregated = np.tensordot(weights_array, stacked, axes=(0, 0))
    aggregated = aggregated - aggregated.min()
    aggregated = aggregated / (aggregated.max() + 1e-8)
    return aggregated
