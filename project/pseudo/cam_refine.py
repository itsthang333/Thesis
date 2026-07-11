from __future__ import annotations

"""Feature-guided CAM refinement (Stage 2.5, optional via --cam-refine).

Motivation: across this project's own diagnostics (prompt_quality.csv's
foreground_precision/recall, the oracle_best_single_dice_clipped vs
selected_dice decomposition), swapping SAM backends (SAM1/SAM2/MedSAM2) and
retuning mask_selection.py's bone_hybrid scoring produced only marginal Dice
changes, while foreground_precision stayed low (~0.04-0.08) throughout. That
localizes the bottleneck upstream, at LayerCAM/support generation itself,
not at SAM candidate quality or mask selection.

This module proposes one CAM refinement step in that spirit (loosely
inspired by AffinityNet/IRNet/S2C's seed-propagation idea, not a faithful
reimplementation of any one paper): treat high-confidence LayerCAM pixels as
seeds, then propagate their activation to other pixels whose DenseNet121
feature vectors (already computed for LayerCAM, no extra model needed) are
similar -- pixels that look like the seed region but were missed by the raw
gradient-based CAM get pulled up, while dissimilar high-activation pixels
(the systemic small-mask/noise bias seen in this project's own candidate-
flow debugging) are not reinforced by feature similarity and stay low.

This is intentionally an additive, optional step (--cam-refine, default
off) so the existing CAM path is unchanged unless explicitly enabled --
letting the fused CAM be compared with and without refinement via the same
prompt_quality.csv / oracle diagnostics already used elsewhere in this
project, rather than replacing the baseline outright.
"""

import numpy as np
import torch
import torch.nn.functional as F


def extract_feature_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    layer_name: str = "denseblock3",
) -> torch.Tensor:
    """Run one forward pass and return one DenseNet121 denseblock's activation map.

    Reuses the same feature space LayerCAM already hooks into (denseblock2/3/4),
    so no extra model or training is needed. denseblock3 is the default: finer
    spatial resolution than denseblock4 (roughly H/16 x W/16 vs H/32 x W/32 for
    typical DenseNet121 input sizes) while still semantically deep enough for
    similarity to reflect tissue/lesion identity rather than raw pixel intensity.

    Args:
        model:       DenseNet121AnatomyClassifier (or compatible: must expose
                     model.features.<layer_name>).
        image_tensor: [1, 3, H, W] on the correct device.
        layer_name:  One of "denseblock2", "denseblock3", "denseblock4".

    Returns:
        feature_map: [C, h, w] float32 tensor (detached, on the same device as
                     image_tensor), NOT upsampled to input resolution.
    """
    target_layer = getattr(model.features, layer_name)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["activations"] = output.detach()

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(image_tensor)
    finally:
        handle.remove()

    if "activations" not in captured:
        raise RuntimeError(f"Forward hook on model.features.{layer_name} did not fire.")

    return captured["activations"][0]  # [C, h, w]


def _connected_components_grid(binary: np.ndarray) -> list[np.ndarray]:
    """8-connected components on a small (feature-grid-resolution) boolean array.

    A separate, self-contained implementation from pseudo/tumor_morphology.py's
    _connected_components (which operates at full input resolution and is
    module-private there) -- this one is sized for the much smaller feature
    grid (e.g. 12x12-24x24), where a plain BFS is fast enough and avoids an
    extra cv2 dependency path for what is a tiny array.
    """
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for row in range(h):
        for col in range(w):
            if not binary[row, col] or visited[row, col]:
                continue
            queue = [(row, col)]
            visited[row, col] = True
            coords = []
            head = 0
            while head < len(queue):
                r, c = queue[head]
                head += 1
                coords.append((r, c))
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            mask = np.zeros((h, w), dtype=bool)
            for r, c in coords:
                mask[r, c] = True
            components.append(mask)
    return components


def refine_cam_with_feature_affinity(
    cam: np.ndarray,
    feature_map: torch.Tensor,
    high_conf_percentile: float = 90.0,
    low_conf_percentile: float = 40.0,
    propagation_strength: float = 0.6,
    strong_similarity_percentile: float = 90.0,
    weak_similarity_percentile: float = 40.0,
) -> np.ndarray:
    """Propagate high-confidence CAM activation to feature-similar pixels.

    Two safeguards against propagating from a noisy seed (a small spurious
    high-activation blob unrelated to the lesion, the same failure mode seen
    in this project's own candidate-flow debugging where a tiny SAM candidate
    scored highest by chance):

    1. Seed selection uses connected components, not a raw percentile mask.
       Thresholding the downsampled CAM at high_conf_percentile can produce
       several disjoint blobs; only the component with the highest (area *
       mean CAM value) is kept as the seed region. Area is part of the
       ranking, not just mean value, because a single spurious pixel can
       have a higher mean than a large true lesion blob (the same small-
       mask-wins-on-mean bias found in this project's own SAM mask-selection
       debugging) -- weighting by area as well favors the larger, more
       spatially coherent blob as the seed.
    2. Propagation strength is adaptive per-pixel based on cosine similarity
       to the seed feature, not a single fixed blend weight or fixed absolute
       similarity thresholds: on real DenseNet121 features the observed
       cosine similarity to a seed rarely approaches 1.0 (empirically closer
       to a 0.0-0.75 range for a real BTXRD image, not the near-1.0 values a
       synthetic feature space can produce), so fixed thresholds like 0.95
       could end up never triggering meaningful propagation strength at all.
       strong_similarity_percentile/weak_similarity_percentile instead define
       the "top X%" and "bottom Y%" of THIS image's own similarity
       distribution as the saturation points, so propagation always has a
       meaningful dynamic range regardless of the absolute similarity scale
       a given image/feature space happens to produce.

    Args:
        cam:                  [H, W] float32 in [0, 1] (the raw fused LayerCAM,
                               at input resolution).
        feature_map:          [C, h, w] tensor from extract_feature_map,
                               at a lower (backbone) resolution.
        high_conf_percentile: Percentile threshold defining the seed candidate
                               mask, before connected-component filtering.
        low_conf_percentile:  Percentile threshold defining refinement targets.
        propagation_strength: Overall cap on how much of the similarity-
                               weighted seed value to blend in (0=no change,
                               1=fully replace at the strong-similarity point).
        strong_similarity_percentile: Percentile of this image's own
                               similarity-to-seed distribution at/above which
                               propagation strength saturates at
                               propagation_strength.
        weak_similarity_percentile: Percentile of this image's own
                               similarity-to-seed distribution at/below which
                               propagation strength is zero.

    Returns:
        refined_cam: [H, W] float32 in [0, 1].
    """
    h_in, w_in = cam.shape
    device = feature_map.device

    # Downsample the CAM to the feature map's resolution to select seeds in
    # the same grid the similarity computation runs on, then upsample the
    # refined result back -- avoids upsampling low-res features (which would
    # blur similarity boundaries) in favor of downsampling the CAM instead.
    cam_tensor = torch.from_numpy(cam).to(device=device, dtype=torch.float32)
    cam_low = F.interpolate(
        cam_tensor[None, None], size=feature_map.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]

    c, h, w = feature_map.shape
    features_flat = feature_map.reshape(c, h * w).T  # [h*w, C]
    features_norm = F.normalize(features_flat, dim=1, eps=1e-8)

    cam_flat = cam_low.reshape(-1)  # [h*w]
    high_thresh = float(torch.quantile(cam_flat, high_conf_percentile / 100.0))
    low_thresh = float(torch.quantile(cam_flat, low_conf_percentile / 100.0))

    seed_candidate = (cam_low >= high_thresh).cpu().numpy()
    components = _connected_components_grid(seed_candidate)
    if not components:
        return cam  # nothing confident enough to propagate from; leave CAM unchanged

    cam_low_np = cam_low.cpu().numpy()
    best_component = max(
        components, key=lambda comp: float(comp.sum()) * float(cam_low_np[comp].mean())
    )
    seed_mask = torch.from_numpy(best_component.reshape(-1)).to(device=device)

    seed_feature = features_norm[seed_mask].mean(dim=0, keepdim=True)  # [1, C]
    seed_feature = F.normalize(seed_feature, dim=1, eps=1e-8)
    seed_value = float(cam_flat[seed_mask].mean())

    similarity = (features_norm @ seed_feature.T).squeeze(1)  # [h*w], in [-1, 1]
    similarity = similarity.clamp(min=0.0)  # only positive similarity can pull the CAM up

    refine_target = cam_flat < low_thresh

    # Derive the strong/weak similarity saturation points from this image's
    # own distribution, restricted to refine_target pixels (excludes the seed
    # itself, which trivially has similarity ~1.0 with its own mean feature
    # and would otherwise skew the percentiles upward).
    if refine_target.any():
        similarity_targets = similarity[refine_target]
        strong_similarity = float(torch.quantile(similarity_targets, strong_similarity_percentile / 100.0))
        weak_similarity = float(torch.quantile(similarity_targets, weak_similarity_percentile / 100.0))
    else:
        strong_similarity, weak_similarity = 1.0, 0.0

    # Adaptive strength: ramp linearly from 0 at weak_similarity to
    # propagation_strength at strong_similarity, flat outside that range.
    denom = max(1e-6, strong_similarity - weak_similarity)
    ramp = ((similarity - weak_similarity) / denom).clamp(min=0.0, max=1.0)
    adaptive_strength = ramp * propagation_strength
    propagated_value = similarity * seed_value

    refined_flat = cam_flat.clone()
    blend = adaptive_strength[refine_target]
    refined_flat[refine_target] = (
        (1.0 - blend) * cam_flat[refine_target] + blend * propagated_value[refine_target]
    )
    refined_low = refined_flat.reshape(1, 1, h, w)

    refined = F.interpolate(refined_low, size=(h_in, w_in), mode="bilinear", align_corners=False)[0, 0]
    refined_np = refined.detach().cpu().numpy().astype(np.float32)

    mn, mx = float(refined_np.min()), float(refined_np.max())
    return (refined_np - mn) / (mx - mn + 1e-8)
