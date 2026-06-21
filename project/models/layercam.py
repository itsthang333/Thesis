from __future__ import annotations

from dataclasses import dataclass, field
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


class LayerCAM:
    """Multi-layer LayerCAM for DenseNet121.

    Registers hooks on denseblock2, denseblock3, denseblock4 and fuses their
    per-layer CAMs with fixed weights [0.2, 0.3, 0.5] before upsampling.

    Reference: Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation
    Maps for Localization", IEEE TIP 2021.
    """

    LAYER_WEIGHTS = (0.2, 0.3, 0.5)

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device

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
        """Compute LayerCAM for one layer.

        LayerCAM differs from Grad-CAM: instead of global-average-pooling the
        gradients and then multiplying, it does element-wise multiply
        (activations * relu(gradients)) per channel, then sums over channels.

        Returns: [B, 1, H_layer, W_layer] (not yet upsampled)
        """
        assert state.activations is not None and state.gradients is not None
        A = state.activations          # [B, C, H, W]
        G = state.gradients            # [B, C, H, W]

        # LayerCAM: element-wise product with relu(gradients)
        cam = (A * F.relu(G)).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        # per-sample min-max normalise at this layer
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        mn = cam_flat.min(dim=1).values.view(B, 1, 1, 1)
        mx = cam_flat.max(dim=1).values.view(B, 1, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)

        # upsample to input image size
        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
        return cam  # [B, 1, H_in, W_in]

    def _compute_cam(self, input_tensor: torch.Tensor, class_index: int) -> LayerCAMOutput:
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None

        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        score = logits[:, class_index].sum()
        score.backward()

        input_size = input_tensor.shape[-2:]
        fused = None
        for state, w in zip(self._states, self.LAYER_WEIGHTS):
            if state.activations is None or state.gradients is None:
                continue
            layer_cam = self._compute_layer_cam(state, input_size)  # [B,1,H,W]
            fused = layer_cam * w if fused is None else fused + layer_cam * w

        if fused is None:
            raise RuntimeError("LayerCAM hooks did not capture any activations.")

        # final normalise across fused map
        B = fused.shape[0]
        fused_flat = fused.view(B, -1)
        mn = fused_flat.min(dim=1).values.view(B, 1, 1, 1)
        mx = fused_flat.max(dim=1).values.view(B, 1, 1, 1)
        fused = (fused - mn) / (mx - mn + 1e-8)

        return LayerCAMOutput(logits=logits, cam=fused.squeeze(1))  # [B, H, W]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def cam_for_class(self, input_tensor: torch.Tensor, class_index: int) -> LayerCAMOutput:
        return self._compute_cam(input_tensor, class_index=class_index)

    def cams_for_active_classes(
        self,
        input_tensor: torch.Tensor,
        class_weights: Sequence[float],
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, list[np.ndarray], list[float], list[int]]:
        """Run CAM only for classes whose sigmoid score >= confidence_threshold.

        Returns:
            logits:         [1, C]
            active_cams:    list of [H, W] numpy arrays (one per active class)
            active_weights: list of float weights corresponding to each cam
            active_indices: list of class indices that were used
        """
        active_indices = [i for i, w in enumerate(class_weights) if w >= confidence_threshold]

        # fallback: if no class is confident, use the top-scoring class
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
