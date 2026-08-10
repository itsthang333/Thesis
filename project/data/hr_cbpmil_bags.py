from __future__ import annotations

"""Annotation-free candidate-bag data path for HR-CBPMIL-IE+."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    from ..config import IMAGENET_MEAN, IMAGENET_STD
    from ..frozen_io import locate_verified_image, sha256_file
    from ..selectors.hr_cbpmil_ie_plus import duplicate_cluster_ids
except ImportError:  # pragma: no cover - script entrypoints with PYTHONPATH=project
    from config import IMAGENET_MEAN, IMAGENET_STD
    from frozen_io import locate_verified_image, sha256_file
    from selectors.hr_cbpmil_ie_plus import duplicate_cluster_ids


def load_candidate_masks(candidate_root: Path, manifest_row: dict[str, str]) -> np.ndarray:
    path = candidate_root / manifest_row["diagnostic_path"]
    if not path.is_file() or sha256_file(path) != manifest_row["diagnostic_sha256"]:
        raise ValueError(f"Frozen candidate payload/hash mismatch: {manifest_row['image_name']}")
    with np.load(path, allow_pickle=False) as payload:
        masks = np.asarray(payload["sam_masks"], dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[1:] != (320, 320):
        raise ValueError(f"Candidate payload geometry differs: {manifest_row['image_name']}")
    if not (1 <= len(masks) <= 243):
        raise ValueError(f"Candidate count outside [1,243]: {manifest_row['image_name']}")
    return masks


def build_cluster_cache(
    candidate_root: Path,
    candidate_rows: dict[str, dict[str, str]],
    output_path: Path,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for stem in sorted(candidate_rows):
        masks = load_candidate_masks(candidate_root, candidate_rows[stem])
        cache[stem] = duplicate_cluster_ids(masks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cache)
    return cache


def load_cluster_cache(
    path: Path,
    candidate_rows: dict[str, dict[str, str]],
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        cache = {key: payload[key].astype(np.int32) for key in payload.files}
    if set(cache) != set(candidate_rows):
        raise ValueError("Duplicate-cluster cache cohort differs from candidate manifest")
    for stem, values in cache.items():
        if len(values) != int(candidate_rows[stem]["candidate_count"]) or np.any(values < 0):
            raise ValueError(f"Duplicate-cluster cache differs for {stem}")
    return cache


class HRCBPMILBagDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        dataset_root: Path,
        candidate_root: Path,
        candidate_rows: dict[str, dict[str, str]],
        cluster_cache: dict[str, np.ndarray],
        augment: bool,
    ) -> None:
        self.rows = list(rows)
        self.dataset_root = dataset_root
        self.candidate_root = candidate_root
        self.candidate_rows = candidate_rows
        self.cluster_cache = cluster_cache
        self.augment = bool(augment)
        expected = {Path(row["image_id"]).stem for row in rows}
        if expected != set(candidate_rows) or expected != set(cluster_cache):
            raise ValueError("Image, candidate and duplicate-cluster cohorts differ")
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        stem = Path(row["image_id"]).stem
        path = locate_verified_image(self.dataset_root, row)
        image = Image.open(path).convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
        masks = load_candidate_masks(self.candidate_root, self.candidate_rows[stem])
        if self.augment and bool(torch.rand(()) < 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            masks = masks[:, :, ::-1].copy()
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return {
            "image_id": row["image_id"],
            "image": tensor,
            "candidate_masks": torch.from_numpy(masks),
            "cluster_ids": torch.from_numpy(self.cluster_cache[stem]),
            "binary_label": int(row["tumor"]),
            "class10_label": int(row["tumor_type"]),
        }


def collate_hr_cbpmil_bags(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    maximum = max(len(sample["candidate_masks"]) for sample in samples)
    batch = len(samples)
    masks = torch.zeros((batch, maximum, 320, 320), dtype=torch.uint8)
    clusters = torch.full((batch, maximum), -1, dtype=torch.int32)
    valid = torch.zeros((batch, maximum), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = len(sample["candidate_masks"])
        masks[index, :count] = sample["candidate_masks"]
        clusters[index, :count] = sample["cluster_ids"]
        valid[index, :count] = True
    return {
        "image_id": [str(sample["image_id"]) for sample in samples],
        "image": torch.stack([sample["image"] for sample in samples]),
        "candidate_masks": masks,
        "candidate_valid": valid,
        "cluster_ids": clusters,
        "binary_label": torch.tensor([sample["binary_label"] for sample in samples]),
        "class10_label": torch.tensor([sample["class10_label"] for sample in samples]),
    }


def write_data_boundary_receipt(path: Path, *, train_images: int, val_images: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": "hr_cbpmil_ie_plus_annotation_free_data_boundary_v1",
                "train_images": int(train_images),
                "validation_images": int(val_images),
                "spatial_ground_truth_read": False,
                "test_images_read": 0,
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
