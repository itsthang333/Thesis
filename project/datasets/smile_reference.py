from __future__ import annotations

"""Split-locked, annotation-free query/reference inputs for SMILE."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ALLOWED_SPLITS = frozenset({"train", "val"})
REFERENCE_SETS = ("primary", "swap")
REFERENCES_PER_SET = 4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SmileRecord:
    image_id: str
    split: str
    tumor: int
    tumor_type: int
    group_id: str
    view: str
    image_sha256: str


@dataclass(frozen=True)
class SmileReference:
    query_id: str
    query_split: str
    query_group_id: str
    query_view: str
    query_image_sha256: str
    retrieval_rank: int
    reference_set: str
    reference_set_rank: int
    reference_id: str
    reference_group_id: str
    reference_view: str
    reference_image_sha256: str
    descriptor_cosine: float


def load_smile_records(
    split_manifest: str | Path,
    *,
    expected_sha256: str,
) -> list[SmileRecord]:
    """Load train/val image labels while ignoring all out-of-scope row fields."""

    path = Path(split_manifest).expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("canonical split manifest SHA-256 mismatch")
    records: list[SmileRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_id",
            "split",
            "eligible",
            "tumor",
            "tumor_type",
            "group_id",
            "view",
            "image_sha256",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("split manifest lacks SMILE fields")
        for row in reader:
            split = str(row.get("split", "")).strip()
            # Scope decision happens before any image/test metadata is parsed.
            if split not in ALLOWED_SPLITS or str(row.get("eligible", "")).strip() != "1":
                continue
            image_id = str(row["image_id"]).strip()
            if not image_id or image_id in seen:
                raise ValueError(f"duplicate/empty image_id: {image_id!r}")
            seen.add(image_id)
            tumor_text = str(row["tumor"]).strip()
            if tumor_text not in {"0", "1"}:
                raise ValueError(f"invalid tumor label: {image_id}")
            tumor = int(tumor_text)
            tumor_type = int(row["tumor_type"])
            if not 0 <= tumor_type < 10 or bool(tumor_type) != bool(tumor):
                raise ValueError(f"binary/subtype label mismatch: {image_id}")
            group_id = str(row["group_id"]).strip()
            view = str(row["view"]).strip().lower()
            image_sha = str(row["image_sha256"]).strip().lower()
            if not group_id or not view or len(image_sha) != 64:
                raise ValueError(f"missing group/view/hash: {image_id}")
            records.append(
                SmileRecord(
                    image_id=image_id,
                    split=split,
                    tumor=tumor,
                    tumor_type=tumor_type,
                    group_id=group_id,
                    view=view,
                    image_sha256=image_sha,
                )
            )
    counts = {
        (split, tumor): sum(r.split == split and r.tumor == tumor for r in records)
        for split in ALLOWED_SPLITS
        for tumor in (0, 1)
    }
    expected = {
        ("train", 0): 1493,
        ("train", 1): 1488,
        ("val", 0): 187,
        ("val", 1): 184,
    }
    if counts != expected:
        raise ValueError(f"canonical train/val counts changed: {counts}")
    group_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(record.group_id, set()).add(record.split)
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"group split leakage: {leaking[:5]}")
    return sorted(records, key=lambda item: item.image_id)


def load_smile_references(
    reference_manifest: str | Path,
    *,
    expected_sha256: str,
    records: list[SmileRecord],
) -> dict[str, dict[str, list[SmileReference]]]:
    path = Path(reference_manifest).expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("reference manifest SHA-256 mismatch")
    record_by_id = {record.image_id: record for record in records}
    train_normal = {
        record.image_id: record
        for record in records
        if record.split == "train" and record.tumor == 0
    }
    grouped: dict[str, dict[str, list[SmileReference]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(SmileReference.__dataclass_fields__)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("reference manifest lacks provenance fields")
        for row in reader:
            query_split = str(row["query_split"]).strip()
            if query_split not in ALLOWED_SPLITS:
                raise ValueError("reference manifest contains out-of-scope query")
            item = SmileReference(
                query_id=str(row["query_id"]).strip(),
                query_split=query_split,
                query_group_id=str(row["query_group_id"]).strip(),
                query_view=str(row["query_view"]).strip().lower(),
                query_image_sha256=str(row["query_image_sha256"]).strip().lower(),
                retrieval_rank=int(row["retrieval_rank"]),
                reference_set=str(row["reference_set"]).strip(),
                reference_set_rank=int(row["reference_set_rank"]),
                reference_id=str(row["reference_id"]).strip(),
                reference_group_id=str(row["reference_group_id"]).strip(),
                reference_view=str(row["reference_view"]).strip().lower(),
                reference_image_sha256=str(row["reference_image_sha256"]).strip().lower(),
                descriptor_cosine=float(row["descriptor_cosine"]),
            )
            query = record_by_id.get(item.query_id)
            reference = train_normal.get(item.reference_id)
            if query is None or reference is None:
                raise ValueError("unknown query/reference ID")
            if (
                item.query_split != query.split
                or item.query_group_id != query.group_id
                or item.query_view != query.view
                or item.query_image_sha256 != query.image_sha256
                or item.reference_group_id != reference.group_id
                or item.reference_view != reference.view
                or item.reference_image_sha256 != reference.image_sha256
                or item.reference_view != item.query_view
                or item.reference_group_id == item.query_group_id
                or item.reference_set not in REFERENCE_SETS
            ):
                raise ValueError("reference contract violation")
            grouped.setdefault(item.query_id, {}).setdefault(item.reference_set, []).append(item)
    if set(grouped) != set(record_by_id):
        raise ValueError("reference query population differs from train/val")
    for query_id, sets in grouped.items():
        if set(sets) != set(REFERENCE_SETS):
            raise ValueError(f"incomplete reference sets: {query_id}")
        used: set[str] = set()
        for set_name in REFERENCE_SETS:
            values = sorted(sets[set_name], key=lambda item: item.reference_set_rank)
            if (
                len(values) != REFERENCES_PER_SET
                or [item.reference_set_rank for item in values] != [1, 2, 3, 4]
                or used.intersection(item.reference_id for item in values)
            ):
                raise ValueError(f"invalid/disjoint reference set: {query_id}")
            used.update(item.reference_id for item in values)
            sets[set_name] = values
    return dict(sorted(grouped.items()))


def median_letterbox_grayscale(
    image: np.ndarray,
    *,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("expected a finite grayscale image")
    low, high = np.percentile(array, (1.0, 99.0))
    normalized = (
        np.zeros_like(array)
        if high <= low + 1e-6
        else np.clip((array - low) / (high - low), 0.0, 1.0)
    ).astype(np.float32)
    source = Image.fromarray(normalized, mode="F")
    scale = min(output_size / source.width, output_size / source.height)
    width = min(output_size, max(1, int(round(source.width * scale))))
    height = min(output_size, max(1, int(round(source.height * scale))))
    resized = source.resize((width, height), Image.Resampling.BILINEAR)
    pad_x = (output_size - width) // 2
    pad_y = (output_size - height) // 2
    canvas = np.full((output_size, output_size), float(np.median(normalized)), dtype=np.float32)
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = np.asarray(resized)
    valid = np.zeros_like(canvas, dtype=np.float32)
    valid[pad_y : pad_y + height, pad_x : pad_x + width] = 1.0
    geometry = {
        "original_height": int(array.shape[0]),
        "original_width": int(array.shape[1]),
        "resized_height": int(height),
        "resized_width": int(width),
        "pad_y": int(pad_y),
        "pad_x": int(pad_x),
    }
    return torch.from_numpy(canvas)[None], torch.from_numpy(valid)[None], geometry


def imagenet_normalize_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError("grayscale tensor must end in 1HW")
    rgb = image.expand(*image.shape[:-3], 3, *image.shape[-2:])
    mean = rgb.new_tensor((0.485, 0.456, 0.406))
    std = rgb.new_tensor((0.229, 0.224, 0.225))
    shape = (1,) * (rgb.ndim - 3) + (3, 1, 1)
    return (rgb - mean.reshape(shape)) / std.reshape(shape)


class BTXRDSMILEReferenceDataset(Dataset):
    def __init__(
        self,
        *,
        root: str | Path,
        split_manifest: str | Path,
        split_manifest_sha256: str,
        reference_manifest: str | Path,
        reference_manifest_sha256: str,
        split: str,
        image_size: int = 512,
        verify_image_hashes: bool = False,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError("SMILE only permits train/val")
        requested = Path(root).expanduser().resolve()
        options = (requested / "images", requested / "BTXRD" / "images")
        self.images_dir = next((path for path in options if path.is_dir()), None)
        if self.images_dir is None:
            raise FileNotFoundError("BTXRD images directory not found")
        self.records = load_smile_records(split_manifest, expected_sha256=split_manifest_sha256)
        self.record_by_id = {record.image_id: record for record in self.records}
        self.samples = [record for record in self.records if record.split == split]
        self.references = load_smile_references(
            reference_manifest,
            expected_sha256=reference_manifest_sha256,
            records=self.records,
        )
        self.image_size = int(image_size)
        if verify_image_hashes:
            required = {sample.image_id for sample in self.samples}
            for sample in self.samples:
                for set_name in REFERENCE_SETS:
                    required.update(item.reference_id for item in self.references[sample.image_id][set_name])
            for image_id in sorted(required):
                if sha256_file(self.images_dir / image_id) != self.record_by_id[image_id].image_sha256:
                    raise ValueError(f"image SHA mismatch: {image_id}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read(self, image_id: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        with Image.open(self.images_dir / image_id) as handle:
            array = np.asarray(handle.convert("L"), dtype=np.float32)
        return median_letterbox_grayscale(array, output_size=self.image_size)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        query, query_valid, geometry = self._read(sample.image_id)
        result: dict[str, object] = {
            "query": query,
            "query_valid": query_valid,
            "tumor": torch.tensor(float(sample.tumor)),
            "subtype": torch.tensor(sample.tumor_type, dtype=torch.long),
            "image_id": sample.image_id,
            "group_id": sample.group_id,
            **{key: torch.tensor(value, dtype=torch.long) for key, value in geometry.items()},
        }
        for set_name in REFERENCE_SETS:
            assignments = self.references[sample.image_id][set_name]
            loaded = [self._read(item.reference_id) for item in assignments]
            result[f"{set_name}_references"] = torch.stack([item[0] for item in loaded])
            result[f"{set_name}_reference_valid"] = torch.stack([item[1] for item in loaded])
            result[f"{set_name}_reference_ids"] = [item.reference_id for item in assignments]
        return result


def collate_smile_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    tensor_fields = (
        "query",
        "query_valid",
        "tumor",
        "subtype",
        "primary_references",
        "primary_reference_valid",
        "swap_references",
        "swap_reference_valid",
        "original_height",
        "original_width",
        "resized_height",
        "resized_width",
        "pad_y",
        "pad_x",
    )
    result: dict[str, object] = {
        field: torch.stack([item[field] for item in batch]) for field in tensor_fields
    }
    for field in (
        "image_id",
        "group_id",
        "primary_reference_ids",
        "swap_reference_ids",
    ):
        result[field] = [item[field] for item in batch]
    return result

