from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import IMAGENET_MEAN, IMAGENET_STD


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def tensor_to_pil(tensor: torch.Tensor, mean: tuple[float, float, float] = IMAGENET_MEAN, std: tuple[float, float, float] = IMAGENET_STD) -> Image.Image:
    image = tensor.detach().cpu().float().clone()
    if image.ndim == 4:
        image = image[0]
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    image = image * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray(_to_uint8(image))


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(image: Image.Image | np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim == 3:
        cam = cam.squeeze()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_rgb = jet_colormap(cam)
    if cam_rgb.shape[:2] != image.shape[:2]:
        cam_rgb = np.array(Image.fromarray(_to_uint8(cam_rgb)).resize((image.shape[1], image.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    overlay = (1.0 - alpha) * image + alpha * cam_rgb
    return _to_uint8(overlay)


def save_overlay(image: Image.Image | np.ndarray, cam: np.ndarray, output_path: str | Path, alpha: float = 0.45) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap(image, cam, alpha=alpha)
    Image.fromarray(overlay).save(output_path)


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8, mode="L").save(output_path)
