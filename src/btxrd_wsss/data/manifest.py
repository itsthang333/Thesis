from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    split: str
    fold: int | None
    group_id: str
    class_index: int
    class_indices: tuple[int, ...]
    is_tumor: bool
    anatomy: str
    view: str


def _integer(row: dict[str, str], key: str, default: int | None = None) -> int | None:
    value = row.get(key, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer {key}={value!r}") from exc


def read_manifest(path: str | Path, *, data_root: str | Path) -> list[ImageRecord]:
    path = Path(path)
    data_root = Path(data_root)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    records: list[ImageRecord] = []
    seen: set[str] = set()
    for row in rows:
        image_id = row.get("image_id", "").strip()
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image_id: {image_id!r}")
        seen.add(image_id)
        relative = row.get("image_path", "").strip() or f"images/{image_id}"
        image_path = Path(relative)
        if not image_path.is_absolute():
            image_path = data_root / image_path
        class_index = _integer(row, "class_index", 0)
        assert class_index is not None
        class_indices_text = row.get("class_indices", "").strip()
        class_indices = tuple(int(value) for value in class_indices_text.split("|") if value)
        if not class_indices:
            class_indices = (class_index,)
        records.append(
            ImageRecord(
                image_id=image_id,
                image_path=image_path,
                split=row.get("split", "train").strip().lower(),
                fold=_integer(row, "fold"),
                group_id=row.get("group_id", image_id).strip() or image_id,
                class_index=class_index,
                class_indices=class_indices,
                is_tumor=bool(_integer(row, "tumor", int(class_index > 0))),
                anatomy=row.get("anatomy", "unknown").strip().lower() or "unknown",
                view=row.get("view", "unknown").strip().lower() or "unknown",
            )
        )
    return records
