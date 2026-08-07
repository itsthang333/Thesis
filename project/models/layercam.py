from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LayerCAMOutput:
    logits: torch.Tensor
    cam: torch.Tensor  # [B, H, W], upsampled to input size, values in [0,1]


@dataclass
class _HookState:
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None


def collapsed_tumor_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Exact normal-vs-any-tumor log-odds for mutually exclusive logits."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [B,K] with K>=2")
    return torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]


class LayerCAM:
    LAYER_WEIGHTS = (0.2, 0.3, 0.5)

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | None = None,
        layer_weights: Sequence[float] | None = None,
        gradient_mode: str = "positive",
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        if gradient_mode not in {"positive", "absolute"}:
            raise ValueError("gradient_mode must be 'positive' or 'absolute'")
        self.gradient_mode = gradient_mode
        if layer_weights is None:
            self.layer_weights = tuple(float(value) for value in self.LAYER_WEIGHTS)
        else:
            values = tuple(float(value) for value in layer_weights)
            if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
                raise ValueError("layer_weights must contain three non-negative values with positive sum")
            total = sum(values)
            self.layer_weights = tuple(value / total for value in values)

        target_layers = [
            model.features.denseblock2,
            model.features.denseblock3,
            model.features.denseblock4,
        ]
        self._states: list[_HookState] = [_HookState() for _ in target_layers]
        self._handles: list = []

        for state, layer in zip(self._states, target_layers):
            self._handles.append(
                layer.register_forward_hook(self._make_forward_hook(state))
            )
            self._handles.append(
                layer.register_full_backward_hook(self._make_backward_hook(state))
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _make_forward_hook(state: _HookState):
        def hook(module, inputs, output):
            state.activations = output.detach()
        return hook

    @staticmethod
    def _make_backward_hook(state: _HookState):
        def hook(module, grad_input, grad_output):
            state.gradients = grad_output[0].detach()
        return hook

    # ------------------------------------------------------------------
    # core LayerCAM computation
    # ------------------------------------------------------------------

    def _compute_layer_cam(self, state: _HookState, input_size: tuple[int, int]) -> torch.Tensor:
        assert state.activations is not None and state.gradients is not None
        A = state.activations          # [B, C, H, W]
        G = state.gradients            # [B, C, H, W]

        gradient_evidence = F.relu(G) if self.gradient_mode == "positive" else G.abs()
        cam = (A * gradient_evidence).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        mn = cam_flat.min(dim=1).values.view(B, 1, 1, 1)
        mx = cam_flat.max(dim=1).values.view(B, 1, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)

        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
        return cam  # [B, 1, H_in, W_in]

    def _finish_cam(self, logits: torch.Tensor, input_size: tuple[int, int]) -> LayerCAMOutput:
        fused = None
        for state, w in zip(self._states, self.layer_weights):
            if state.activations is None or state.gradients is None:
                continue
            layer_cam = self._compute_layer_cam(state, input_size)
            fused = layer_cam * w if fused is None else fused + layer_cam * w
        if fused is None:
            raise RuntimeError("LayerCAM hooks did not capture any activations.")
        B = fused.shape[0]
        fused_flat = fused.view(B, -1)
        mn = fused_flat.min(dim=1).values.view(B, 1, 1, 1)
        mx = fused_flat.max(dim=1).values.view(B, 1, 1, 1)
        fused = (fused - mn) / (mx - mn + 1e-8)
        return LayerCAMOutput(logits=logits, cam=fused.squeeze(1))

    def _compute_cam(self, input_tensor: torch.Tensor, class_index: int | torch.Tensor) -> LayerCAMOutput:
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None

        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if isinstance(class_index, torch.Tensor):
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            score = logits[batch_indices, class_index].sum()
        else:
            score = logits[:, class_index].sum()
        score.backward()

        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cam_for_tumor_union(self, input_tensor: torch.Tensor) -> LayerCAMOutput:
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None
        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[1] < 2:
            raise ValueError("cam_for_tumor_union requires a multi-class head with a normal class at index 0")
        logits[:, 1:].sum().backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cam_for_tumor_union_contrast(self, input_tensor: torch.Tensor) -> LayerCAMOutput:
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None
        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[1] < 2:
            raise ValueError(
                "cam_for_tumor_union_contrast requires a multi-class head with a normal class at index 0"
            )
        (logits[:, 1:].sum(dim=1) - logits[:, 0]).sum().backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cam_for_tumor_log_odds(self, input_tensor: torch.Tensor) -> LayerCAMOutput:
        """LayerCAM for exact collapsed tumor-vs-normal softmax log-odds.

        For mutually exclusive classes with class 0=normal and 1..K=tumor,
        ``log(P(tumor)/P(normal))`` simplifies exactly to
        ``logsumexp(z_1..z_K) - z_0``.  This gives the ten-class E1 arm the
        same binary semantic target as the one-logit arm without requiring a
        tumor subtype at inference.
        """
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None
        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[1] < 2:
            raise ValueError(
                "cam_for_tumor_log_odds requires a multi-class head with "
                "a normal class at index 0"
            )
        tumor_log_odds = collapsed_tumor_log_odds(logits)
        tumor_log_odds.sum().backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def cam_for_class(self, input_tensor: torch.Tensor, class_index: int | torch.Tensor) -> LayerCAMOutput:
        return self._compute_cam(input_tensor, class_index=class_index)

    def cam_for_class_contrast(
        self,
        input_tensor: torch.Tensor,
        class_index: int | torch.Tensor,
        reference_index: int = 0,
    ) -> LayerCAMOutput:
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None

        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if isinstance(class_index, torch.Tensor):
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            score = (logits[batch_indices, class_index] - logits[batch_indices, reference_index]).sum()
        else:
            score = (logits[:, class_index] - logits[:, reference_index]).sum()
        score.backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cams_for_active_classes(
        self,
        input_tensor: torch.Tensor,
        class_weights: Sequence[float],
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, list[np.ndarray], list[float], list[int]]:
        active_indices = [i for i, w in enumerate(class_weights) if w >= confidence_threshold]

        if not active_indices:
            active_indices = [int(np.argmax(class_weights))]

        active_cams: list[np.ndarray] = []
        logits_out: torch.Tensor | None = None
        active_weights: list[float] = []

        for idx in active_indices:
            out = self._compute_cam(input_tensor, class_index=idx)
            logits_out = out.logits
            active_cams.append(out.cam[0].detach().cpu().numpy())
            active_weights.append(float(class_weights[idx]))

        assert logits_out is not None
        return logits_out, active_cams, active_weights, active_indices

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None) -> LayerCAMOutput:
        if class_index is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                class_index = int(logits.argmax(dim=1).item())
        return self._compute_cam(input_tensor, class_index=class_index)
