from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import DEFAULT_ANATOMY_COLUMNS
from datasets.common import (
    IMAGE_EXTENSIONS,
    apply_clahe,
    make_classification_transform,
    make_segmentation_image_transform,
    make_segmentation_mask_transform,
)

try:
    from pycocotools import mask as coco_mask  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    coco_mask = None


DEFAULT_NON_BONE_KEYWORDS = (
    "soft",
    "tissue",
    "implant",
    "intravenous",
    "cannula",
    "ring",
    "artifact",
    "marker",
    "label",
    "ruler",
)


def resolve_ramh1200_segmentation_root(root: str | Path) -> Path:
    """Return the directory that contains train/val/test split folders."""
    root = Path(root)
    candidates = [
        root,
        root / "Segmentation",
        root / "RAM-H1200-v1" / "Segmentation",
    ]
    for candidate in candidates:
        if (candidate / "train").exists() or (candidate / "val").exists() or (candidate / "validation").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find RAM-H1200 Segmentation split folders. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _resolve_split_dir(segmentation_root: Path, split: str) -> tuple[str, Path]:
    candidates = [split]
    if split == "validation":
        candidates.append("val")
    elif split == "val":
        candidates.append("validation")

    for candidate in candidates:
        split_dir = segmentation_root / candidate
        if split_dir.exists():
            return candidate, split_dir
    return split, segmentation_root / split


def _decode_coco_segmentation(segmentation: object, height: int, width: int) -> np.ndarray:
    if coco_mask is None:
        raise ImportError(
            "pycocotools is required to decode RAM-H1200 COCO RLE masks. "
            "Install it with: pip install pycocotools"
        )

    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(rles)
    elif isinstance(segmentation, dict):
        decoded = coco_mask.decode(segmentation)
    else:
        raise ValueError(f"Unsupported COCO segmentation type: {type(segmentation)!r}")

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


class RAMH1200SegmentationDataset(Dataset):
    """RAM-H1200 full-hand bone segmentation dataset.

    The dataset card describes COCO RLE annotation files inside:
    Segmentation/{train,val,test}/_annotations_bone_rle.coco.json.
    This loader turns all selected bone instances into one binary mask because
    the current thesis pipeline trains/evaluates binary visible-bone masks.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        augment: bool = False,
        use_clahe: bool = False,
        annotation_name: str = "_annotations_bone_rle.coco.json",
        include_categories: Sequence[str] | None = None,
        exclude_keywords: Sequence[str] = DEFAULT_NON_BONE_KEYWORDS,
    ) -> None:
        self.segmentation_root = resolve_ramh1200_segmentation_root(root)
        self.split, self.split_dir = _resolve_split_dir(self.segmentation_root, split)
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe
        self.annotation_path = self.split_dir / annotation_name

        if not self.split_dir.exists():
            raise FileNotFoundError(f"RAM-H1200 split directory not found: {self.split_dir}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"RAM-H1200 annotation file not found: {self.annotation_path}")

        with self.annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = {int(item["id"]): str(item.get("name", item["id"])) for item in coco.get("categories", [])}
        include_set = {name.lower() for name in include_categories} if include_categories else None
        exclude_tokens = tuple(keyword.lower() for keyword in exclude_keywords)

        self.category_names = categories
        self.selected_category_ids = {
            category_id
            for category_id, name in categories.items()
            if (include_set is None or name.lower() in include_set)
            and not any(token in name.lower() for token in exclude_tokens)
        }

        self.annotations_by_image_id: dict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            category_id = int(annotation.get("category_id", -1))
            if category_id in self.selected_category_ids:
                self.annotations_by_image_id[int(annotation["image_id"])].append(annotation)

        self.samples: list[dict[str, object]] = []
        for image_info in coco.get("images", []):
            image_path = self._resolve_image_path(str(image_info["file_name"]))
            if image_path is None:
                continue
            image_id = int(image_info["id"])
            if not self.annotations_by_image_id.get(image_id):
                continue
            self.samples.append(
                {
                    "id": image_id,
                    "file_name": str(image_info["file_name"]),
                    "image_path": image_path,
                    "height": int(image_info.get("height", 0)),
                    "width": int(image_info.get("width", 0)),
                }
            )

        if not self.samples:
            raise FileNotFoundError(
                "No RAM-H1200 image/mask pairs found. Check split name, annotations, "
                "and include/exclude category filters."
            )

        self.image_transform = make_segmentation_image_transform(image_size)
        self.mask_transform = make_segmentation_mask_transform(image_size)

    def _resolve_image_path(self, file_name: str) -> Path | None:
        candidates = [self.split_dir / file_name, self.split_dir / Path(file_name).name]
        stem = Path(file_name).stem
        candidates.extend(self.split_dir / f"{stem}{extension}" for extension in IMAGE_EXTENSIONS)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def _build_mask(self, sample: dict[str, object], fallback_size: tuple[int, int]) -> Image.Image:
        width = int(sample["width"]) or fallback_size[0]
        height = int(sample["height"]) or fallback_size[1]
        mask = np.zeros((height, width), dtype=bool)
        for annotation in self.annotations_by_image_id[int(sample["id"])]:
            mask |= _decode_coco_segmentation(annotation["segmentation"], height, width)
        return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        image = Image.open(image_path).convert("RGB")
        mask = self._build_mask(sample, image.size)

        if self.use_clahe:
            image = apply_clahe(image)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        return image_tensor, mask_tensor, image_path.name


class RAMH1200ClassificationDataset(Dataset):
    """Image-only RAM-H1200 dataset for Stage 2 pseudo-mask generation.

    RAM-H1200 segmentation images are full hand radiographs. The returned target
    is a constant anatomy vector with hand=1 and other configured anatomy labels
    set to 0, which keeps the existing Stage 2 loader interface intact.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        target_columns: Sequence[str] = DEFAULT_ANATOMY_COLUMNS,
        image_size: int = 512,
        use_clahe: bool = False,
    ) -> None:
        self.segmentation_root = resolve_ramh1200_segmentation_root(root)
        self.split, self.split_dir = _resolve_split_dir(self.segmentation_root, split)
        self.target_columns = list(target_columns)
        self.use_clahe = use_clahe
        if not self.split_dir.exists():
            raise FileNotFoundError(f"RAM-H1200 split directory not found: {self.split_dir}")

        self.image_paths = sorted(
            path
            for path in self.split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No RAM-H1200 images found in: {self.split_dir}")

        self.image_transform = make_classification_transform(image_size, augment=False)
        self.target = torch.tensor(
            [1.0 if column.lower() == "hand" else 0.0 for column in self.target_columns],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.use_clahe:
            image = apply_clahe(image)
        return self.image_transform(image), self.target.clone(), image_path.name
