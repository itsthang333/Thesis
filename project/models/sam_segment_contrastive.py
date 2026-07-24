from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SamSegmentMapStore:
    """Validated in-memory S2C region maps keyed by the dataset image ID."""

    REQUIRED_COLUMNS = {
        "image_id",
        "source_image_sha256",
        "region_map_path",
        "region_map_sha256",
        "regions",
        "map_width",
        "map_height",
        "map_dtype",
    }

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[dict[str, object]],
        *,
        expected_manifest_sha256: str,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "region_map_manifest.csv"
        if not self.root.is_dir():
            raise FileNotFoundError(f"SAM segment-map root does not exist: {self.root}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"SAM segment-map manifest does not exist: {self.manifest_path}"
            )
        if not expected_manifest_sha256:
            raise ValueError(
                "A frozen --sam-segment-map-manifest-sha256 is required when SSC is enabled"
            )
        actual_manifest_sha256 = sha256_file(self.manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "SAM segment-map manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
            )
        self.manifest_sha256 = actual_manifest_sha256

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"SAM segment-map manifest is empty: {self.manifest_path}")
        missing_columns = self.REQUIRED_COLUMNS - set(rows[0])
        if missing_columns:
            raise ValueError(
                f"SAM segment-map manifest is missing columns: {sorted(missing_columns)}"
            )
        by_name: dict[str, dict[str, str]] = {}
        for row in rows:
            image_id = row["image_id"].strip()
            if not image_id or image_id in by_name:
                raise ValueError(
                    f"Duplicate or empty image_id in SAM segment-map manifest: {image_id!r}"
                )
            by_name[image_id] = row

        expected = {str(sample["image_id"]): sample for sample in samples}
        if set(by_name) != set(expected):
            missing = sorted(set(expected) - set(by_name))
            extra = sorted(set(by_name) - set(expected))
            raise ValueError(
                "SAM segment-map/train population mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

        self._maps: dict[str, np.ndarray] = {}
        expected_shape: tuple[int, int] | None = None
        for image_id in sorted(expected):
            row = by_name[image_id]
            expected_source_sha = str(expected[image_id].get("image_sha256", ""))
            if (
                not expected_source_sha
                or row["source_image_sha256"] != expected_source_sha
            ):
                raise ValueError(f"Source image SHA-256 mismatch for {image_id}")
            relative_path = Path(row["region_map_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe region-map path for {image_id}: {relative_path}")
            map_path = (self.root / relative_path).resolve()
            if self.root not in map_path.parents:
                raise ValueError(f"Region map escapes root for {image_id}: {map_path}")
            if not map_path.is_file():
                raise FileNotFoundError(f"Missing SAM region map for {image_id}: {map_path}")
            if sha256_file(map_path) != row["region_map_sha256"]:
                raise ValueError(f"SAM region-map SHA-256 mismatch for {image_id}")
            with Image.open(map_path) as image:
                region_map = np.asarray(image, dtype=np.uint16)
            if region_map.ndim != 2:
                raise ValueError(
                    f"SAM region map must be 2D for {image_id}, got {region_map.shape}"
                )
            declared_shape = (int(row["map_height"]), int(row["map_width"]))
            if region_map.shape != declared_shape:
                raise ValueError(
                    f"SAM region-map shape mismatch for {image_id}: "
                    f"{region_map.shape} != {declared_shape}"
                )
            if row["map_dtype"] != "uint16":
                raise ValueError(
                    f"SAM region-map dtype metadata must be uint16 for {image_id}"
                )
            if expected_shape is None:
                expected_shape = region_map.shape
            elif region_map.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent SAM region-map shape: {region_map.shape} != {expected_shape}"
                )
            positive = np.unique(region_map)
            positive = positive[positive > 0]
            declared_regions = int(row["regions"])
            if declared_regions <= 0:
                raise ValueError(f"SAM produced no usable region for {image_id}")
            expected_ids = np.arange(1, declared_regions + 1, dtype=np.uint16)
            if not np.array_equal(positive, expected_ids):
                raise ValueError(
                    f"SAM region IDs are not contiguous for {image_id}: "
                    f"declared={declared_regions}, present={positive.tolist()[:10]}"
                )
            self._maps[image_id] = region_map.copy()

        self.map_shape = expected_shape

    def __len__(self) -> int:
        return len(self._maps)

    def load_batch(
        self, image_ids: Sequence[str], *, device: torch.device
    ) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        for value in image_ids:
            image_id = str(value)
            if image_id not in self._maps:
                raise KeyError(f"No validated SAM segment map for {image_id}")
            arrays.append(self._maps[image_id])
        stacked = np.stack(arrays, axis=0).astype(np.int64, copy=False)
        return torch.from_numpy(stacked).to(device=device, non_blocking=True)


def sam_segment_contrastive_loss(
    features: torch.Tensor,
    region_maps: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """S2C SSC loss using detached per-region prototypes.

    This is the `torch_scatter`-free equivalent of the official implementation:
    normalized classifier features are averaged inside each SAM segment, those
    prototypes are detached, and every covered pixel is classified by cosine
    similarity to the prototypes from its own image. Region ID 0 is ignored.
    """
    if features.ndim != 4:
        raise ValueError(f"features must have shape [B,D,H,W], got {features.shape}")
    if region_maps.ndim != 3:
        raise ValueError(
            f"region_maps must have shape [B,H,W], got {region_maps.shape}"
        )
    if features.shape[0] != region_maps.shape[0]:
        raise ValueError("Feature/map batch sizes differ")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")

    target_size = tuple(int(value) for value in region_maps.shape[-2:])
    features = F.interpolate(
        features.float(), size=target_size, mode="bilinear", align_corners=False
    )
    features = F.normalize(features, dim=1)

    loss_sum = features.new_zeros(())
    valid_pixels = 0
    for sample_index in range(features.shape[0]):
        sample_features = features[sample_index].flatten(1)
        sample_regions = region_maps[sample_index].reshape(-1).long()
        valid = sample_regions > 0
        if not bool(valid.any()):
            raise ValueError(
                f"SAM segment map {sample_index} has no positive region IDs"
            )
        ids = sample_regions[valid]
        unique = torch.unique(ids, sorted=True)
        expected = torch.arange(
            1, int(unique.numel()) + 1, device=ids.device, dtype=ids.dtype
        )
        if not torch.equal(unique, expected):
            raise ValueError(
                f"SAM region IDs must be contiguous 1..N, got {unique.tolist()[:10]}"
            )

        region_count = int(unique.numel())
        detached = sample_features[:, valid].detach()
        prototypes = sample_features.new_zeros(
            (sample_features.shape[0], region_count)
        )
        targets = ids - 1
        prototypes.index_add_(1, targets, detached)
        counts = torch.bincount(targets, minlength=region_count).to(
            dtype=prototypes.dtype
        )
        prototypes = prototypes / counts.clamp_min(1).unsqueeze(0)
        prototypes = F.normalize(prototypes, dim=0)

        logits = (
            prototypes.transpose(0, 1) @ sample_features[:, valid]
        ) * float(temperature)
        loss_sum = loss_sum + F.cross_entropy(
            logits.transpose(0, 1), targets, reduction="sum"
        )
        valid_pixels += int(valid.sum().item())

    if valid_pixels == 0:
        raise ValueError("SSC batch has no valid SAM-covered pixels")
    return loss_sum / valid_pixels
