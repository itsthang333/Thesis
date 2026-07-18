from __future__ import annotations

"""Dataset selection helpers so scripts can target RAM-H1200 or BTXRD via --dataset.

Both datasets are consumed identically downstream (image-level classification
target for Stage 1/LayerCAM, binary GT mask for evaluation/U-Net), so scripts
only need to pick which loader class to instantiate.
"""

from pathlib import Path
from typing import Sequence

from config import SUPPORTED_DATASETS
from datasets.btxrd import BTXRDClassificationDataset, BTXRDSegmentationDataset
from datasets.ramh1200 import RAMH1200ClassificationDataset, RAMH1200SegmentationDataset


def _check_dataset_name(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}.")
    return dataset


def build_classification_dataset(
    dataset: str,
    root: str | Path,
    split: str,
    target_columns: Sequence[str],
    image_size: int,
    use_clahe: bool = False,
    augment: bool = False,
    preprocessing_mode: str = "none",
    normalization: str = "imagenet",
):
    dataset = _check_dataset_name(dataset)
    if dataset == "ramh1200":
        return RAMH1200ClassificationDataset(
            root=root,
            split=split,
            target_columns=target_columns,
            image_size=image_size,
            use_clahe=use_clahe,
            augment=augment,
            preprocessing_mode=preprocessing_mode,
            normalization=normalization,
        )
    return BTXRDClassificationDataset(
        root=root,
        split=split,
        target_columns=target_columns,
        image_size=image_size,
        use_clahe=use_clahe,
        augment=augment,
        preprocessing_mode=preprocessing_mode,
        normalization=normalization,
    )


def build_segmentation_dataset(
    dataset: str,
    root: str | Path,
    split: str,
    image_size: int,
    augment: bool = False,
    use_clahe: bool = False,
    annotation_name: str = "_annotations_bone_rle.coco.json",
    pseudo_mask_root: str | Path | None = None,
    require_all_pseudo_masks: bool = True,
):
    dataset = _check_dataset_name(dataset)
    if dataset == "ramh1200":
        return RAMH1200SegmentationDataset(
            root=root,
            split=split,
            image_size=image_size,
            augment=augment,
            use_clahe=use_clahe,
            annotation_name=annotation_name,
        )
    return BTXRDSegmentationDataset(
        root=root,
        split=split,
        image_size=image_size,
        augment=augment,
        use_clahe=use_clahe,
        pseudo_mask_root=pseudo_mask_root,
        require_all_pseudo_masks=require_all_pseudo_masks,
    )
