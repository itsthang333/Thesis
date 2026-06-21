from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from config import DEFAULT_ANATOMY_COLUMNS, IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _as_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def build_image_index(image_roots: str | Path | Sequence[str | Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in _as_paths(image_roots):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
                index.setdefault(path.stem, path)
    return index


def build_train_val_indices(num_samples: int, val_fraction: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(num_samples * val_fraction)) if num_samples > 1 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices = val_indices
    return train_indices, val_indices


def _apply_clahe(image: Image.Image) -> Image.Image:
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


# --- REFACTOR: TÁCH BIỆT CÁC HÀM TRANSFORM ---

def _make_classification_transform(image_size: int, augment: bool = False) -> transforms.Compose:
    """Transform dành riêng cho Classification (chỉ áp dụng trên ảnh, tự do augment hình học)"""
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


def _make_segmentation_image_transform(image_size: int) -> transforms.Compose:
    """
    Transform ảnh cơ bản cho Segmentation.
    KHÔNG ĐƯỢC CHỨA hình học (Flip, Rotate...) vì chúng phải được áp dụng đồng bộ với mask.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _make_segmentation_mask_transform(image_size: int) -> transforms.Compose:
    """Transform cho Mask (Dùng nội suy NEAREST để giữ nguyên giá trị nhãn)"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


class FracAtlasClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_roots: str | Path | Sequence[str | Path],
        target_columns: Sequence[str] = DEFAULT_ANATOMY_COLUMNS,
        image_size: int = 512,
        augment: bool = False,
        use_clahe: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_index = build_image_index(image_roots)
        self.target_columns = list(target_columns)
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe
        # Sử dụng đúng hàm transform của classification
        self.image_transform = _make_classification_transform(image_size, augment=augment)

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            self.rows = list(reader)

        self.samples: list[dict[str, object]] = []
        for row in self.rows:
            image_name = row["image_id"]
            image_path = self._resolve_image_path(image_name)
            if image_path is None:
                continue
            self.samples.append({"image_path": image_path, "row": row})

        if not self.samples:
            raise FileNotFoundError("No images matched the CSV rows. Check image roots and file names.")

    def _resolve_image_path(self, image_name: str) -> Path | None:
        if image_name in self.image_index:
            return self.image_index[image_name]
        stem = Path(image_name).stem
        if stem in self.image_index:
            return self.image_index[stem]
        for extension in IMAGE_EXTENSIONS:
            candidate = f"{stem}{extension}"
            if candidate in self.image_index:
                return self.image_index[candidate]
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        row = sample["row"]

        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            with Image.open(image_path) as loaded_image:
                loaded_image.load()
                image = loaded_image.convert("RGB")
        if self.use_clahe:
            image = _apply_clahe(image)
        image_tensor = self.image_transform(image)

        target_values = [float(row.get(column, 0.0) or 0.0) for column in self.target_columns]
        target_tensor = torch.tensor(target_values, dtype=torch.float32)
        if target_tensor.numel() == 1:
            target_tensor = target_tensor.view(1)
        return image_tensor, target_tensor, image_path.name


class FracAtlasSegmentationDataset(Dataset):
    def __init__(
        self,
        image_roots: str | Path | Sequence[str | Path],
        mask_root: str | Path,
        image_size: int = 512,
        augment: bool = False,
        use_clahe: bool = False,
    ) -> None:
        self.image_index = build_image_index(image_roots)
        self.mask_index = build_image_index(mask_root)
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe
        
        # Sử dụng các hàm transform đã được thiết kế an toàn cho Segmentation
        self.image_transform = _make_segmentation_image_transform(image_size)
        self.mask_transform = _make_segmentation_mask_transform(image_size)

        self.samples: list[tuple[Path, Path]] = []
        unique_image_paths = sorted(set(self.image_index.values()))
        for image_path in unique_image_paths:
            image_name = image_path.name
            mask_path = self._resolve_mask_path(image_name)
            if mask_path is not None:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError("No image/mask pairs found. Check pseudo-mask directory naming.")

    def _resolve_mask_path(self, image_name: str) -> Path | None:
        stem = Path(image_name).stem
        if image_name in self.mask_index:
            return self.mask_index[image_name]
        if stem in self.mask_index:
            return self.mask_index[stem]
        for extension in IMAGE_EXTENSIONS:
            candidate = f"{stem}{extension}"
            if candidate in self.mask_index:
                return self.mask_index[candidate]
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.use_clahe:
            image = _apply_clahe(image)

        # Áp dụng Augmentation HÌNH HỌC thủ công đồng thời lên cả Image và Mask
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        return image_tensor, mask_tensor, image_path.name
