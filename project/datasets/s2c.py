from __future__ import annotations

"""Image-label dataset paired with an integrity-checked SAM segment cache."""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.btxrd import BTXRDClassificationDataset
from pseudo.sam_segment_cache import (
    load_cached_proposal_bank,
    load_cached_segments,
    validate_sam_segment_cache,
)


class BTXRDS2CDataset(Dataset):
    def __init__(
        self,
        *,
        root: str | Path,
        split: str,
        split_manifest: str | Path,
        segment_cache_dir: str | Path,
        image_size: int,
        augment: bool = False,
        normalization: str = "imagenet",
    ) -> None:
        self.base = BTXRDClassificationDataset(
            root=root,
            split=split,
            target_columns=("tumor",),
            image_size=image_size,
            augment=False,
            normalization=normalization,
            split_manifest=split_manifest,
        )
        self.samples = self.base.samples
        self.split = split
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.segment_cache_dir = Path(segment_cache_dir).resolve()
        cache_info = validate_sam_segment_cache(
            self.segment_cache_dir,
            self.samples,
            split=split,
        )
        self.cache_rows: dict[str, dict[str, str]] = cache_info["rows"]  # type: ignore[assignment]
        self.cache_summary = {key: value for key, value in cache_info.items() if key != "rows"}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        image, tumor, image_name = self.base[index]
        stem = Path(str(image_name)).stem
        row = self.cache_rows[stem]
        segment_map, quality = load_cached_segments(self.segment_cache_dir / row["cache_path"])
        proposal_masks, proposal_quality = load_cached_proposal_bank(
            self.segment_cache_dir / row["cache_path"]
        )
        segments = torch.from_numpy(segment_map).to(torch.float32)[None, None]
        segments = F.interpolate(
            segments,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0, 0].to(torch.long)
        proposal_tensor = torch.from_numpy(proposal_masks)
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            segments = torch.flip(segments, dims=(-1,))
            proposal_tensor = torch.flip(proposal_tensor, dims=(-1,))
        return {
            "image": image,
            "tumor": tumor[0].to(torch.float32),
            "segments": segments,
            "quality": torch.from_numpy(quality),
            "proposal_masks": proposal_tensor,
            "proposal_quality": torch.from_numpy(proposal_quality),
            "image_name": str(image_name),
        }


def collate_s2c_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "tumor": torch.stack([item["tumor"] for item in batch]),
        "segments": torch.stack([item["segments"] for item in batch]),
        "quality": [item["quality"] for item in batch],
        "proposal_masks": [item["proposal_masks"] for item in batch],
        "proposal_quality": [item["proposal_quality"] for item in batch],
        "image_name": [item["image_name"] for item in batch],
    }

