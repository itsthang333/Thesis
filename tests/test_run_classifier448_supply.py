from __future__ import annotations

import csv
from pathlib import Path

import pytest

from project.run_classifier448_supply import load_image_label_counts


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image_id", "split", "eligible", "tumor")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_rejects_noncanonical_counts(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    _write_manifest(
        path,
        [{"image_id": "a.jpeg", "split": "train", "eligible": "1", "tumor": "0"}],
    )
    with pytest.raises(ValueError, match="Canonical train/validation counts"):
        load_image_label_counts(path)


def test_rejects_nonbinary_label_before_count_check(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    _write_manifest(
        path,
        [{"image_id": "a.jpeg", "split": "train", "eligible": "1", "tumor": "2"}],
    )
    with pytest.raises(ValueError, match="Non-binary"):
        load_image_label_counts(path)
