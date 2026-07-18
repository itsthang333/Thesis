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
        """Compute LayerCAM for one layer.

        LayerCAM differs from Grad-CAM: instead of global-average-pooling the
        gradients and then multiplying, it does element-wise multiply
        (activations * relu(gradients)) per channel, then sums over channels.

        Returns: [B, 1, H_layer, W_layer] (not yet upsampled)
        """
        assert state.activations is not None and state.gradients is not None
        A = state.activations          # [B, C, H, W]
        G = state.gradients            # [B, C, H, W]

        # Standard LayerCAM keeps positive class evidence only.  The optional
        # absolute mode is a controlled ablation for low-confidence target
        # classes whose useful spatial signal can appear in negative gradients.
        gradient_evidence = F.relu(G) if self.gradient_mode == "positive" else G.abs()
        cam = (A * gradient_evidence).sum(dim=1, keepdim=True)  # [B, 1, H, W]
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
            # Per-sample target class (one class per batch item, e.g. each
            # sample's own ground-truth tumor_type) -- gradient for each
            # sample's activations is isolated to that sample's own class,
            # since summing independent per-sample scalar terms and calling
            # backward() once produces exactly the same per-sample gradients
            # as backward()-ing each one separately (verified: the sum trick
            # gives each row a clean one-hot gradient at its own class index,
            # with zero cross-contamination between batch items). This lets
            # a full batch with differing target classes share ONE forward+
            # backward pass instead of one per sample.
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            score = logits[batch_indices, class_index].sum()
        else:
            score = logits[:, class_index].sum()
        score.backward()

        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cam_for_tumor_union(self, input_tensor: torch.Tensor) -> LayerCAMOutput:
        """LayerCAM for aggregate non-normal evidence in a 10-class BTXRD head.

        This is an opt-in localization ablation: it uses one backward pass for
        the sum of tumor logits (classes 1..C-1), while leaving image-level
        normal/tumor detection and all downstream prompts unchanged.
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
            raise ValueError("cam_for_tumor_union requires a multi-class head with a normal class at index 0")
        logits[:, 1:].sum().backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cam_for_tumor_union_contrast(self, input_tensor: torch.Tensor) -> LayerCAMOutput:
        """LayerCAM for non-normal evidence contrasted against the normal logit.

        The aggregate tumor score is ``sum(logits[:, 1:]) - logits[:, 0]``.
        This keeps the class-agnostic localization of :meth:`cam_for_tumor_union`
        while suppressing features that also explain the normal class.  It is
        an image-level-only ablation and never consumes a segmentation label.
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
                "cam_for_tumor_union_contrast requires a multi-class head with a normal class at index 0"
            )
        (logits[:, 1:].sum(dim=1) - logits[:, 0]).sum().backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def cam_for_class(self, input_tensor: torch.Tensor, class_index: int | torch.Tensor) -> LayerCAMOutput:
        """class_index: a single int (same class for every sample in the
        batch) or a [B] long tensor (one class per sample)."""
        return self._compute_cam(input_tensor, class_index=class_index)

    def cam_for_class_contrast(
        self,
        input_tensor: torch.Tensor,
        class_index: int | torch.Tensor,
        reference_index: int = 0,
    ) -> LayerCAMOutput:
        """LayerCAM for ``logit[class_index] - logit[reference_index]``.

        BTXRD's class 0 is normal.  This opt-in diagnostic suppresses spatial
        evidence shared by the normal class while preserving the image-level
        target class; no polygon or segmentation label is involved.
        """
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

    def cam_for_region_conditioned_class_contrast(
        self,
        input_tensor: torch.Tensor,
        class_index: int | torch.Tensor,
        region_index: int | torch.Tensor,
        region_weight: float = 0.5,
        reference_index: int = 0,
    ) -> LayerCAMOutput:
        """Class-vs-normal CAM plus tumor-vs-normal evidence in the same region."""
        self.model.zero_grad(set_to_none=True)
        for state in self._states:
            state.activations = None
            state.gradients = None

        logits, _anatomy_logits, region_logits, _features = self.model(
            input_tensor, return_anatomy=True
        )
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        if not isinstance(class_index, torch.Tensor):
            class_index = torch.full_like(batch_indices, int(class_index))
        if not isinstance(region_index, torch.Tensor):
            region_index = torch.full_like(batch_indices, int(region_index))
        global_contrast = logits[batch_indices, class_index] - logits[batch_indices, reference_index]
        matched_region_contrast = (
            region_logits[batch_indices, region_index, 1]
            - region_logits[batch_indices, region_index, 0]
        )
        (global_contrast + float(region_weight) * matched_region_contrast).sum().backward()
        return self._finish_cam(logits, input_tensor.shape[-2:])

    def cams_for_active_classes(
        self,
        input_tensor: torch.Tensor,
        class_weights: Sequence[float],
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, list[np.ndarray], list[float], list[int]]:
        """Run CAM only for classes whose classifier score >= confidence_threshold.

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
