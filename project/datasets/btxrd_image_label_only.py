from __future__ import annotations

"""BTXRD classification input that never opens segmentation annotations."""

from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.common import apply_clahe, make_classification_transform
from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
)


FROZEN_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)


def _resolve_image_root(root: str | Path) -> Path:
    supplied = Path(root)
    candidates = (supplied, supplied / "BTXRD")
    matches = [candidate.resolve() for candidate in candidates if (candidate / "images").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one BTXRD image root, found {matches}")
    return matches[0]


class BTXRDImageLabelOnlyDataset(Dataset):
    """Load verified radiographs and binary image labels, never annotation bytes."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        split_manifest: str | Path,
        expected_split_sha256: str,
        image_size: int,
        augment: bool,
        expected_cohort: int | None = None,
        use_clahe: bool = False,
        preprocessing_mode: str = "none",
        normalization: str = "imagenet",
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("Image-label-only BTXRD input permits train/val only")
        self.root = _resolve_image_root(root)
        self.rows = load_split_rows_without_annotations(
            Path(split_manifest),
            expected_sha256=expected_split_sha256,
            split=split,
        )
        if expected_cohort is not None and len(self.rows) != expected_cohort:
            raise ValueError(f"Image-label-only {split} cohort mismatch")
        self.paths = [locate_verified_image(self.root, row) for row in self.rows]
        self.use_clahe = bool(use_clahe)
        self.preprocessing_mode = (
            "clahe" if use_clahe and preprocessing_mode == "none" else preprocessing_mode
        )
        self.transform = make_classification_transform(
            image_size,
            augment=augment,
            preprocessing_mode=self.preprocessing_mode,
            normalization=normalization,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.paths[index]).convert("RGB")
        if self.use_clahe and self.preprocessing_mode != "clahe":
            image = apply_clahe(image)
        target = torch.tensor([float(row["tumor"])], dtype=torch.float32)
        return self.transform(image), target, str(row["image_id"])


def build_image_label_only_classification_dataset(
    root: str | Path,
    split: str,
    target_columns: Sequence[str],
    image_size: int,
    use_clahe: bool = False,
    augment: bool = False,
    preprocessing_mode: str = "none",
    normalization: str = "imagenet",
    split_manifest: str | Path | None = None,
) -> BTXRDImageLabelOnlyDataset:
    if list(target_columns) != ["tumor"]:
        raise ValueError("B4 permits only the binary image-level tumor label")
    if split_manifest is None:
        raise ValueError("B4 requires the frozen split manifest")
    return BTXRDImageLabelOnlyDataset(
        root,
        split=split,
        split_manifest=split_manifest,
        expected_split_sha256=FROZEN_SPLIT_SHA256,
        image_size=image_size,
        augment=augment,
        expected_cohort=2981 if split == "train" else 371,
        use_clahe=use_clahe,
        preprocessing_mode=preprocessing_mode,
        normalization=normalization,
    )
