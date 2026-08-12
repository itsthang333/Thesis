from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import pandas as pd

ANATOMY_COLUMNS = (
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip bone",
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
)
VIEW_COLUMNS = ("frontal", "lateral", "oblique")


def _active(row: pd.Series, columns: tuple[str, ...]) -> str:
    values = [column for column in columns if int(row.get(column, 0)) == 1]
    return "|".join(values) if values else "unknown"


def _image_number(value: str) -> int:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else -1


def _class_indices(row: pd.Series, tumor_columns: list[str]) -> tuple[int, ...]:
    active = [index + 1 for index, column in enumerate(tumor_columns) if int(row[column]) == 1]
    return tuple(active) if active else (0,)


def _group_rows(frame: pd.DataFrame, tumor_columns: list[str]) -> list[dict[str, object]]:
    ordered = frame.copy()
    ordered["_number"] = ordered["image_id"].map(_image_number)
    ordered = ordered.sort_values(["_number", "image_id"], kind="stable")
    records: list[dict[str, object]] = []
    previous_number: int | None = None
    previous_signature: tuple[object, ...] | None = None
    group_index = -1
    for _, row in ordered.iterrows():
        anatomy = _active(row, ANATOMY_COLUMNS)
        view = _active(row, VIEW_COLUMNS)
        class_indices = _class_indices(row, tumor_columns)
        # Primary class is used only for stratification. The full multi-label target is retained.
        class_index = max(class_indices)
        signature = (
            row.get("center"),
            row.get("age"),
            str(row.get("gender", "")).strip().lower(),
            anatomy,
            class_index,
        )
        number = int(row["_number"])
        if (
            previous_number is None
            or number != previous_number + 1
            or signature != previous_signature
        ):
            group_index += 1
        records.append(
            {
                "image_id": str(row["image_id"]),
                "group_id": f"btxrd-group-{group_index:06d}",
                "class_index": class_index,
                "class_indices": "|".join(map(str, class_indices)),
                "tumor": int(class_index > 0),
                "anatomy": anatomy,
                "view": view,
            }
        )
        previous_number, previous_signature = number, signature
    return records


def _assign_splits(records: list[dict[str, object]], seed: int) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["group_id"]), []).append(record)
    by_class: dict[int, list[str]] = {}
    for group_id, rows in groups.items():
        labels = {int(row["class_index"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Group {group_id} crosses class labels")
        by_class.setdefault(labels.pop(), []).append(group_id)
    rng = random.Random(seed)
    for _class_index, group_ids in sorted(by_class.items()):
        rng.shuffle(group_ids)
        for position, group_id in enumerate(group_ids):
            bucket = position % 10
            split = "test" if bucket == 0 else "val" if bucket == 1 else "train"
            for row in groups[group_id]:
                row["split"] = split
                row["fold"] = "" if split != "train" else position % 5


def build_manifest(
    *,
    data_root: str | Path,
    output_path: str | Path,
    tumor_columns: list[str],
    seed: int = 42,
) -> Path:
    data_root, output_path = Path(data_root), Path(output_path)
    table = data_root / "dataset.xlsx"
    if not table.exists():
        table = data_root / "dataset.csv"
    frame = pd.read_excel(table) if table.suffix == ".xlsx" else pd.read_csv(table)
    missing = {"image_id", *tumor_columns} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset table is missing columns: {sorted(missing)}")
    records = _group_rows(frame, tumor_columns)
    _assign_splits(records, seed)
    for record in records:
        image_path = data_root / "images" / str(record["image_id"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        record["image_path"] = str(Path("images") / str(record["image_id"])).replace("\\", "/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_id",
        "image_path",
        "group_id",
        "split",
        "fold",
        "class_index",
        "class_indices",
        "tumor",
        "anatomy",
        "view",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return output_path
