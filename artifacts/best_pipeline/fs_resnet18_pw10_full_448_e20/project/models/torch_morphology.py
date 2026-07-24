from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def erode(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return -dilate(-x, kernel_size=kernel_size)


def opening(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return dilate(erode(x, kernel_size), kernel_size)


def closing(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return erode(dilate(x, kernel_size), kernel_size)


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
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
    if cam.ndim == 3:
        cam = cam.unsqueeze(1)

    with torch.no_grad():
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
