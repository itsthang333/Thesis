from __future__ import annotations

"""Generate per-class and fused LayerCAM heatmaps for a single image tensor.

Pure computation helper — no I/O. Called from generate_pseudo_masks.py and inference.py.
"""

import numpy as np
import torch

from models.layercam import LayerCAM


def generate_fused_cam(
    layercam: LayerCAM,
    image_tensor: torch.Tensor,
    class_weights: np.ndarray,
    confidence_threshold: float = 0.5,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Generate confidence-filtered, weighted-average fused LayerCAM.

    Args:
        layercam:             Initialised LayerCAM instance.
        image_tensor:         [1, 3, H, W] on the correct device.
        class_weights:        classifier scores, shape [C], numpy float32.
        confidence_threshold: Classes below this are excluded from fusion.

    Returns:
        fused_cam:      [H, W] float32 in [0, 1].
        per_class_cams: list of [H, W] numpy arrays for each active class.
        active_indices: list of class indices that were used.
    """
    # active_indices comes directly from cams_for_active_classes — no re-derivation
    _, per_class_cams, active_weights, active_indices = layercam.cams_for_active_classes(
        image_tensor,
        class_weights=class_weights,
        confidence_threshold=confidence_threshold,
    )

    # weighted average fusion
    w_arr = np.array(active_weights, dtype=np.float32)
    w_arr = np.clip(w_arr, 0.0, None)
    w_arr = w_arr / (w_arr.sum() + 1e-8)

    # Use np.zeros as start so sum() on an empty iterable returns a zero array,
    # not Python int 0 (which would crash on .astype()).
    zeros = np.zeros_like(per_class_cams[0]) if per_class_cams else None
    if zeros is None:
        raise RuntimeError("generate_fused_cam: no active CAMs to fuse.")
    fused = sum((w * cam for w, cam in zip(w_arr, per_class_cams)), zeros)
    fused = fused.astype(np.float32)

    mn, mx = float(fused.min()), float(fused.max())
    fused = (fused - mn) / (mx - mn + 1e-8)

    return fused, per_class_cams, active_indices
