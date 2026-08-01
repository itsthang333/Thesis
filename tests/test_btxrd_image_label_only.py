from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path

from PIL import Image
import pytest

from datasets.btxrd_image_label_only import BTXRDImageLabelOnlyDataset


def _write_manifest(path: Path, image_id: str, image_sha256: str) -> str:
    fields = ("image_id", "group_id", "split", "eligible", "tumor", "image_sha256")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "image_id": image_id,
                "group_id": "g1",
                "split": "val",
                "eligible": "1",
                "tumor": "1",
                "image_sha256": image_sha256,
            }
        )
    return sha256(path.read_bytes()).hexdigest()


def test_image_label_only_dataset_never_requires_annotation_directory(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "tumor.png"
    Image.new("RGB", (8, 8), "black").save(image_path)
    split_path = tmp_path / "split.csv"
    split_sha = _write_manifest(
        split_path,
        image_path.name,
        sha256(image_path.read_bytes()).hexdigest(),
    )

    dataset = BTXRDImageLabelOnlyDataset(
        tmp_path,
        split="val",
        split_manifest=split_path,
        expected_split_sha256=split_sha,
        image_size=8,
        augment=False,
    )
    image, target, image_id = dataset[0]
    assert not (tmp_path / "Annotations").exists()
    assert tuple(image.shape) == (3, 8, 8)
    assert target.tolist() == [1.0]
    assert image_id == "tumor.png"


def test_image_label_only_dataset_rejects_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train/val only"):
        BTXRDImageLabelOnlyDataset(
            tmp_path,
            split="test",
            split_manifest=tmp_path / "missing.csv",
            expected_split_sha256="0" * 64,
            image_size=8,
            augment=False,
        )
