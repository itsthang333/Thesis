from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
XRAY_PREPROCESSING_MODES = ("none", "clahe", "contrast", "gamma", "foreground_crop")


def as_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def build_image_index(image_roots: str | Path | Sequence[str | Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in as_paths(image_roots):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
                index.setdefault(path.stem, path)
    return index


def apply_clahe(image: Image.Image) -> Image.Image:
    if cv2 is None:
        return ImageOps.equalize(image)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    rgb = cv2.cvtColor(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _percentile_contrast(image: Image.Image, low: float = 1.0, high: float = 99.0) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    lo = float(np.percentile(gray, low))
    hi = float(np.percentile(gray, high))
    if hi <= lo + 1e-6:
        return image.convert("RGB")
    arr = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _gamma_correct(image: Image.Image, gamma: float = 0.80) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def _foreground_crop(image: Image.Image, padding_ratio: float = 0.06) -> Image.Image:
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    threshold = max(8, int(float(np.percentile(gray, 99.0)) * 0.08))
    rows, cols = np.where(gray > threshold)
    if rows.size == 0 or cols.size == 0:
        return rgb
    h, w = gray.shape
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    pad_y = max(4, int((y1 - y0 + 1) * padding_ratio))
    pad_x = max(4, int((x1 - x0 + 1) * padding_ratio))
    box = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x + 1),
        min(h, y1 + pad_y + 1),
    )
    return rgb.crop(box)


def preprocess_xray_image(image: Image.Image, mode: str = "none") -> Image.Image:
    mode = (mode or "none").lower()
    if mode not in XRAY_PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode '{mode}'. Choose from {XRAY_PREPROCESSING_MODES}.")
    image = image.convert("RGB")
    if mode == "none":
        return image
    if mode == "clahe":
        return apply_clahe(image)
    if mode == "contrast":
        return _percentile_contrast(image)
    if mode == "gamma":
        return _gamma_correct(image)
    if mode == "foreground_crop":
        return _foreground_crop(image)
    return image


class XRayPreprocessTransform:
    def __init__(self, mode: str = "none") -> None:
        self.mode = mode

    def __call__(self, image: Image.Image) -> Image.Image:
        return preprocess_xray_image(image, self.mode)


class RadImageNetNormalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        bgr = tensor[[2, 1, 0], :, :]
        return (bgr * 255.0 - 127.5) * 2 / 255.0


def make_classification_transform(
    image_size: int,
    augment: bool = False,
    preprocessing_mode: str = "none",
    normalization: str = "imagenet",
) -> transforms.Compose:
    transform_list: list[object] = []
    if preprocessing_mode and preprocessing_mode.lower() != "none":
        transform_list.append(XRayPreprocessTransform(preprocessing_mode))
    transform_list.append(transforms.Resize((image_size, image_size)))
    if augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.ToTensor())
    if normalization == "radimagenet":
        transform_list.append(RadImageNetNormalize())
    else:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(transform_list)


def make_segmentation_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_segmentation_mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
