from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Grayscale dilation via max-pooling (stride=1, same-size output)."""
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def erode(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Grayscale erosion: erode(x) = -dilate(-x). Complementary to dilate()."""
    return -dilate(-x, kernel_size=kernel_size)


def opening(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Erosion then dilation -- removes small bright specks/noise without
    growing the remaining region back past its original size.

    NOTE: only meaningful when the foreground occupies enough of the grid
    that a kernel_size x kernel_size neighborhood can plausibly survive
    erosion intact. At DenseNet121's native 12x12 feature-map resolution,
    a percentile-96 threshold keeps only ~6 of 144 pixels, almost always
    scattered with no adjacent foreground pixels -- erosion (which requires
    an ENTIRE kernel-sized neighborhood to be foreground) then wipes out
    every single one, and dilation has nothing left to restore. Confirmed
    empirically on real DenseNet121 CAMs. Do not use this at resolutions
    that low; see cam_to_soft_attention_target, which no longer calls this.
    """
    return dilate(erode(x, kernel_size), kernel_size)


def closing(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Dilation then erosion -- fills small holes/gaps inside a bright region
    without shrinking its outer boundary. Same low-resolution caveat as
    opening() applies."""
    return erode(dilate(x, kernel_size), kernel_size)


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """Separable Gaussian blur, same-size output. x: [B, 1, H, W]."""
    device, dtype = x.device, x.dtype
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    pad = kernel_size // 2
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1)
    x = F.conv2d(x, kernel_h, padding=(0, pad))
    x = F.conv2d(x, kernel_v, padding=(pad, 0))
    return x


def cam_to_soft_attention_target(
    cam: torch.Tensor,
    percentile: float = 96.0,
    blur_kernel_size: int = 3,
    blur_sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a diffuse CAM into a sharper soft attention target for
    distillation: percentile-based soft gating (NOT a hard binary threshold)
    -> Gaussian blur -> renormalize to [0, 1].

    Design history: an earlier version hard-thresholded at the percentile,
    then ran morphological opening+closing (kernel_size=3) to denoise/fill
    gaps before blurring. That works fine on a full-resolution image (e.g.
    384x384, where a 3px kernel is a small, local operation) but is wrong
    at DenseNet121's native 12x12 feature-map resolution -- a percentile-96
    threshold there keeps only ~6 of 144 pixels, almost never forming a
    solid kernel_size x kernel_size neighborhood, so erosion (which opening
    starts with) wipes out every single pixel and dilation has nothing left
    to restore -- confirmed empirically to degenerate the target to
    all-zero on every tested batch, real DenseNet121 included. At this grid
    size a 3x3 kernel already covers 25% of the grid's width -- not the
    "small, local" denoising operation morphology is meant to be.

    Fix: skip the hard threshold + opening/closing pipeline entirely for a
    soft (not binary) target -- normalize the CAM to [0,1], softly upweight
    values above the percentile (rather than hard-zeroing everything below
    it) via a smooth sigmoid-like gate, then blur. This keeps the "focus on
    the CAM's most confident ~4% region" motivation (lesions average ~2.6%
    of image area vs the CAM's usual ~15% thresholded area) without ever
    creating a sparse binary mask that morphology could wipe out. Soft
    targets are also generally preferable for BCE-based distillation over
    hard 0/1 masks, since they don't force the student toward a brittle,
    already-thresholded decision boundary.

    cam: [B, 1, H, W] or [B, H, W], values in [0, 1] (already min-max
         normalized per-sample by the caller, e.g. LayerCAM's output).
    Returns:
        soft_target: [B, 1, H, W], values in [0, 1]. For a degenerate sample
            (see valid_mask below) this is all-zero, but MUST NOT be treated
            as "background everywhere" -- the caller should exclude it from
            any loss via valid_mask, not train against it as a real target.
        valid_mask: [B] bool. False for samples whose CAM was too flat to
            carry any real signal (e.g. a freshly-EMA'd teacher early in
            training, or the same all-identical-channel pathological input
            class found earlier in this project).
    """
    if cam.ndim == 3:
        cam = cam.unsqueeze(1)

    with torch.no_grad():
        # torch.quantile (used below) hard-requires float32/float64 input --
        # raises "quantile() input tensor must be either float or double
        # dtype" on fp16, a separate crash risk from autocast's own
        # BCE-specific block (torch.quantile isn't on any documented
        # autocast safelist/blocklist -- it just inherits whatever dtype it
        # receives). Force fp32 explicitly here rather than relying on the
        # caller's autocast(enabled=False) to have already produced an fp32
        # tensor by construction.
        cam = cam.float()
        batch_size = cam.shape[0]
        flat = cam.view(batch_size, -1)
        mn = flat.min(dim=1).values.view(batch_size, 1, 1, 1)
        mx = flat.max(dim=1).values.view(batch_size, 1, 1, 1)
        cam_range = (mx - mn).view(batch_size)
        valid_mask = cam_range > 1e-6

        cam_norm = torch.where(
            cam_range.view(batch_size, 1, 1, 1) > 1e-6,
            (cam - mn) / (mx - mn + 1e-8),
            torch.zeros_like(cam),
        )

        # Soft gate centered on the percentile threshold: values well above
        # it approach 1, values well below it approach 0, with a smooth
        # transition instead of a hard cutoff (avoids the sparse-binary-mask
        # problem morphology could no longer recover from).
        thresholds = torch.quantile(flat, percentile / 100.0, dim=1).view(batch_size, 1, 1, 1)
        sharpness = 10.0  # steepness of the soft gate around the threshold
        gated = torch.sigmoid(sharpness * (cam_norm - thresholds))

        soft_target = gaussian_blur(gated, blur_kernel_size, blur_sigma)

        flat_soft = soft_target.view(batch_size, -1)
        soft_mn = flat_soft.min(dim=1).values.view(batch_size, 1, 1, 1)
        soft_mx = flat_soft.max(dim=1).values.view(batch_size, 1, 1, 1)
        soft_target = torch.where(
            (soft_mx - soft_mn).view(batch_size, 1, 1, 1) > 1e-6,
            (soft_target - soft_mn) / (soft_mx - soft_mn + 1e-8),
            torch.zeros_like(soft_target),
        )

    return soft_target, valid_mask
