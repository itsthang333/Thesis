from __future__ import annotations

"""Canonical BTXRD dataset constructors used by every CLI entry point."""

from pathlib import Path
from typing import Sequence

from datasets.btxrd import BTXRDClassificationDataset, BTXRDSegmentationDataset


def build_classification_dataset(
    root: str | Path,
    split: str,
    target_columns: Sequence[str],
    image_size: int,
    use_clahe: bool = False,
    augment: bool = False,
    preprocessing_mode: str = "none",
    normalization: str = "imagenet",
    split_manifest: str | Path | None = None,
):
    return BTXRDClassificationDataset(
        root=root,
        split=split,
        target_columns=target_columns,
        image_size=image_size,
        use_clahe=use_clahe,
        augment=augment,
        preprocessing_mode=preprocessing_mode,
        normalization=normalization,
        split_manifest=split_manifest,
    )


def build_segmentation_dataset(
    root: str | Path,
    split: str,
    image_size: int,
    augment: bool = False,
    use_clahe: bool = False,
    pred_mask_dir: str | Path | None = None,
    split_manifest: str | Path | None = None,
):
    return BTXRDSegmentationDataset(
        root=root,
        split=split,
        image_size=image_size,
        augment=augment,
        use_clahe=use_clahe,
        pred_mask_dir=pred_mask_dir,
        split_manifest=split_manifest,
    )
