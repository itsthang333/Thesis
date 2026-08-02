from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from project.datasets.smile_reference import load_smile_records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_record_loader_filters_test_before_metadata_parse(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    fields = [
        "image_id",
        "split",
        "eligible",
        "tumor",
        "tumor_type",
        "group_id",
        "view",
        "image_sha256",
    ]
    rows = []
    counters = {("train", 0): 1493, ("train", 1): 1488, ("val", 0): 187, ("val", 1): 184}
    index = 0
    for (split, tumor), count in counters.items():
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "image_id": f"IMG{index:06d}.jpeg",
                    "split": split,
                    "eligible": "1",
                    "tumor": str(tumor),
                    "tumor_type": "1" if tumor else "0",
                    "group_id": f"{split}-g{index}",
                    "view": "frontal",
                    "image_sha256": "a" * 64,
                }
            )
    # Deliberately malformed non-permitted row: fields after split must never be parsed.
    rows.append({"image_id": "", "split": "test", "eligible": "x"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    records = load_smile_records(path, expected_sha256=_sha(path))
    assert len(records) == 3352
    assert all(record.split in {"train", "val"} for record in records)


def test_record_loader_rejects_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    fields = ["image_id", "split", "eligible", "tumor", "tumor_type", "group_id", "view", "image_sha256"]
    counters = {("train", 0): 1493, ("train", 1): 1488, ("val", 0): 187, ("val", 1): 184}
    rows = []
    index = 0
    for (split, tumor), count in counters.items():
        for offset in range(count):
            index += 1
            rows.append(
                {
                    "image_id": f"IMG{index:06d}.jpeg",
                    "split": split,
                    "eligible": "1",
                    "tumor": str(tumor),
                    "tumor_type": "1" if tumor else "0",
                    "group_id": "leak" if (split == "train" and offset == 0) or (split == "val" and offset == 0) else f"{split}-{index}",
                    "view": "frontal",
                    "image_sha256": "b" * 64,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="group split leakage"):
        load_smile_records(path, expected_sha256=_sha(path))

