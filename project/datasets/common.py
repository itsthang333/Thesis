from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile, ImageOps
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def make_classification_transform(image_size: int, augment: bool = False) -> transforms.Compose:
    transform_list: list[object] = [transforms.Resize((image_size, image_size))]
    if augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
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
