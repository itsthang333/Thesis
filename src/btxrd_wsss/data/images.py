from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_native_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("L")
        values = np.asarray(image, dtype=np.float32) / 255.0
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid radiograph: {path}")
    return values


def percentile_normalize(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _clahe(image: np.ndarray) -> np.ndarray:
    try:
        import cv2

        u8 = np.round(percentile_normalize(image) * 255).astype(np.uint8)
        return (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255
        )
    except ImportError:
        pil = Image.fromarray(np.round(percentile_normalize(image) * 255).astype(np.uint8))
        return np.asarray(ImageOps.equalize(pil), dtype=np.float32) / 255


def _local_contrast(image: np.ndarray) -> np.ndarray:
    normalized = percentile_normalize(image)
    pil = Image.fromarray(np.round(normalized * 255).astype(np.uint8))
    blurred = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=3.0)), dtype=np.float32) / 255
    return np.clip(0.5 + normalized - blurred, 0.0, 1.0).astype(np.float32)


def make_hrnet_channels(image: np.ndarray) -> np.ndarray:
    raw = percentile_normalize(image)
    return np.stack((raw, _clahe(raw), _local_contrast(raw)), axis=0).astype(np.float32)


def make_rgb(image: np.ndarray) -> np.ndarray:
    raw = percentile_normalize(image)
    return np.repeat(raw[None], 3, axis=0).astype(np.float32)
