from __future__ import annotations

import csv
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from datasets.common import (
    IMAGE_EXTENSIONS,
    apply_clahe,
    make_classification_transform,
    make_segmentation_image_transform,
    make_segmentation_mask_transform,
)

DEFAULT_CSV_NAMES = ("dataset.csv", "dataset.xlsx")
DEFAULT_ANNOTATIONS_DIR = "Annotations"
DEFAULT_IMAGES_DIR = "images"

DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
DEFAULT_SPLIT_SEED = 42

TUMOR_TYPE_COLUMNS = (
    "osteochondroma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "osteofibroma",
    "synovial osteochondroma",
    "other bt",
    "osteosarcoma",
    "other mt",
)
TUMOR_TYPE_CLASS_NAMES = ("normal",) + TUMOR_TYPE_COLUMNS


def resolve_btxrd_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = [root, root / "BTXRD"]
    for candidate in candidates:
        if (candidate / DEFAULT_IMAGES_DIR).exists() and (candidate / DEFAULT_ANNOTATIONS_DIR).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find BTXRD dataset layout. Expected 'images/' and 'Annotations/' under one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _load_dataset_table(btxrd_root: Path) -> list[dict[str, object]]:
    csv_path = btxrd_root / "dataset.csv"
    xlsx_path = btxrd_root / "dataset.xlsx"

    if csv_path.exists():
        import csv as csv_module

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv_module.DictReader(handle)
            return [dict(row) for row in reader]

    if xlsx_path.exists():
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading dataset.xlsx requires pandas + openpyxl. "
                "Install with: pip install pandas openpyxl "
                "(or export dataset.csv instead)."
            ) from exc
        frame = pd.read_excel(xlsx_path)
        return frame.to_dict(orient="records")

    raise FileNotFoundError(
        f"Neither dataset.csv nor dataset.xlsx found under {btxrd_root}"
    )


def _row_flag(row: dict[str, object], column: str) -> int:
    if column not in row:
        raise ValueError(f"BTXRD metadata is missing required column {column!r}")
    value = row[column]
    if value is None or value == "":
        raise ValueError(f"BTXRD metadata column {column!r} contains an empty value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"BTXRD metadata column {column!r} must contain 0 or 1, got {value!r}"
        ) from exc
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(
            f"BTXRD metadata column {column!r} must contain 0 or 1, got {value!r}"
        )
    return int(numeric)


def _row_tumor_type_index(row: dict[str, object]) -> int:
    matches = [i + 1 for i, column in enumerate(TUMOR_TYPE_COLUMNS) if _row_flag(row, column)]
    if len(matches) > 1:
        names = [TUMOR_TYPE_COLUMNS[index - 1] for index in matches]
        raise ValueError(f"BTXRD row has multiple tumor types set: {names}")
    return matches[0] if matches else 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_flag(row: dict[str, str], column: str, image_id: str) -> int:
    value = row.get(column, "")
    if value not in {"0", "1"}:
        raise ValueError(f"Split manifest row {image_id}: {column!r} must be 0 or 1, got {value!r}")
    return int(value)


@lru_cache(maxsize=8)
def _load_split_manifest_records_verified(
    btxrd_root_text: str,
    manifest_path_text: str,
    manifest_mtime_ns: int,
    manifest_size: int,
) -> list[dict[str, object]]:
    """Validate every manifest source hash once per unchanged manifest and process.

    Training constructs train, validation, and mask-audit datasets in the same
    process. Re-hashing all full-resolution radiographs for each construction
    adds minutes without increasing integrity evidence. The manifest stat is
    part of the cache key, while a new process always performs a fresh check.
    """
    del manifest_mtime_ns, manifest_size  # cache-key provenance only
    btxrd_root = Path(btxrd_root_text)
    manifest_path = Path(manifest_path_text)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"BTXRD split manifest is empty: {manifest_path}")

    required = {
        "image_id", "group_id", "split", "eligible", "tumor", "benign",
        "malignant", "tumor_type", "image_sha256", "dataset_table",
        "dataset_table_sha256",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"BTXRD split manifest is missing required columns: {missing}")

    metadata_name = str(rows[0]["dataset_table"])
    metadata_path = btxrd_root / metadata_name
    if metadata_path.is_file() and rows[0]["dataset_table_sha256"]:
        actual_hash = _sha256_file(metadata_path)
        if actual_hash != rows[0]["dataset_table_sha256"]:
            raise ValueError(
                f"BTXRD source metadata hash does not match split manifest: "
                f"expected {rows[0]['dataset_table_sha256']}, got {actual_hash}"
            )

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    image_hashes: dict[str, str] = {}
    images_dir = btxrd_root / DEFAULT_IMAGES_DIR
    annotations_dir = btxrd_root / DEFAULT_ANNOTATIONS_DIR
    for row in rows:
        image_id = str(row["image_id"]).strip()
        split = str(row["split"]).strip()
        if not image_id or image_id in seen_ids:
            raise ValueError(f"BTXRD split manifest contains duplicate/empty image_id: {image_id!r}")
        seen_ids.add(image_id)
        if split not in {"train", "val", "test", "excluded"}:
            raise ValueError(f"BTXRD split manifest has unknown split {split!r} for {image_id}")
        if _manifest_flag(row, "eligible", image_id) == 0:
            continue
        if split == "excluded":
            raise ValueError(f"Eligible image {image_id} cannot have split='excluded'")
        image_path = images_dir / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Split manifest references missing BTXRD image: {image_path}")
        tumor = _manifest_flag(row, "tumor", image_id)
        benign = _manifest_flag(row, "benign", image_id)
        malignant = _manifest_flag(row, "malignant", image_id)
        tumor_type = int(row["tumor_type"])
        if not 0 <= tumor_type < len(TUMOR_TYPE_CLASS_NAMES):
            raise ValueError(f"Invalid tumor_type {tumor_type} for {image_id}")
        if bool(tumor_type) != bool(tumor) or benign + malignant != tumor:
            raise ValueError(f"Inconsistent class flags in split manifest for {image_id}")
        group_id = str(row["group_id"]).strip()
        if not group_id:
            raise ValueError(f"Missing group_id in split manifest for {image_id}")
        group_splits.setdefault(group_id, set()).add(split)
        image_hash = str(row["image_sha256"]).strip()
        if image_hash:
            actual_image_hash = _sha256_file(image_path)
            if actual_image_hash != image_hash:
                raise ValueError(
                    f"BTXRD image hash does not match split manifest for {image_id}: "
                    f"expected {image_hash}, got {actual_image_hash}"
                )
            image_hashes.setdefault(image_hash, image_id)
            if image_hashes[image_hash] != image_id:
                raise ValueError(
                    f"Exact duplicate image hash appears in eligible manifest: "
                    f"{image_hashes[image_hash]} and {image_id}"
                )
        annotation_hash = str(row.get("annotation_sha256", "")).strip()
        if annotation_hash:
            annotation_path = annotations_dir / f"{Path(image_id).stem}.json"
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Split manifest references missing annotation for {image_id}: {annotation_path}"
                )
            actual_annotation_hash = _sha256_file(annotation_path)
            if actual_annotation_hash != annotation_hash:
                raise ValueError(
                    f"BTXRD annotation hash does not match split manifest for {image_id}: "
                    f"expected {annotation_hash}, got {actual_annotation_hash}"
                )
        records.append(
            {
                "image_id": image_id,
                "tumor": tumor,
                "benign": benign,
                "malignant": malignant,
                "tumor_type": tumor_type,
                "split": split,
                "group_id": group_id,
                "group_source": str(row.get("group_source", "")),
                "group_confidence": str(row.get("group_confidence", "")),
                "center": row.get("center", ""),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
                "anatomy": row.get("anatomy", ""),
                "view": row.get("view", ""),
                "image_sha256": image_hash,
                "annotation_sha256": annotation_hash,
                "split_manifest": str(manifest_path),
            }
        )
    leaking_groups = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if leaking_groups:
        raise ValueError(f"Split manifest has group overlap: {list(leaking_groups.items())[:5]}")
    if not records:
        raise ValueError(f"BTXRD split manifest has no eligible records: {manifest_path}")
    return sorted(records, key=lambda record: str(record["image_id"]))


def _load_split_manifest_records(
    btxrd_root: Path,
    split_manifest: str | Path,
) -> list[dict[str, object]]:
    manifest_path = Path(split_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BTXRD split manifest does not exist: {manifest_path}")
    manifest_stat = manifest_path.stat()
    cached = _load_split_manifest_records_verified(
        str(btxrd_root.resolve()),
        str(manifest_path),
        manifest_stat.st_mtime_ns,
        manifest_stat.st_size,
    )
    # Dataset instances should not be able to mutate the process-wide cache.
    return [dict(record) for record in cached]


def load_btxrd_records(
    btxrd_root: str | Path,
    split_manifest: str | Path | None = None,
) -> list[dict[str, object]]:
    btxrd_root = resolve_btxrd_root(btxrd_root)
    if split_manifest is not None:
        return _load_split_manifest_records(btxrd_root, split_manifest)
    rows = _load_dataset_table(btxrd_root)
    records: list[dict[str, object]] = []
    seen_image_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            raise ValueError(f"BTXRD metadata row {row_number} has no image_id")
        if image_id in seen_image_ids:
            raise ValueError(f"BTXRD metadata contains duplicate image_id {image_id!r}")
        seen_image_ids.add(image_id)
        try:
            tumor = _row_flag(row, "tumor")
            benign = _row_flag(row, "benign")
            malignant = _row_flag(row, "malignant")
            tumor_type = _row_tumor_type_index(row)
        except ValueError as exc:
            raise ValueError(f"Invalid BTXRD metadata row {row_number} ({image_id}): {exc}") from exc
        if bool(tumor_type) != bool(tumor):
            raise ValueError(
                f"Invalid BTXRD metadata row {row_number} ({image_id}): tumor={tumor} "
                f"but tumor-type count is {int(bool(tumor_type))}"
            )
        if benign + malignant != tumor:
            raise ValueError(
                f"Invalid BTXRD metadata row {row_number} ({image_id}): "
                f"benign + malignant must equal tumor, got {benign} + {malignant} != {tumor}"
            )
        records.append(
            {
                "image_id": image_id,
                "tumor": tumor,
                "benign": benign,
                "malignant": malignant,
                "tumor_type": tumor_type,
            }
        )
    return records


def split_btxrd_records(
    records: Sequence[dict[str, object]],
    split: str,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[dict[str, object]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test.")

    if records and "split" in records[0]:
        if any(str(record.get("split")) not in {"train", "val", "test"} for record in records):
            raise ValueError("BTXRD records contain an invalid manifest split")
        selected = [record for record in records if str(record["split"]) == split]
        selected.sort(key=lambda item: str(item["image_id"]))
        return selected

    def stratum_key(record: dict[str, object]) -> int:
        return int(record["tumor_type"])

    groups: dict[int, list[dict[str, object]]] = {i: [] for i in range(len(TUMOR_TYPE_CLASS_NAMES))}
    for record in records:
        groups[stratum_key(record)].append(record)

    train_ratio, val_ratio, _test_ratio = ratios
    selected: list[dict[str, object]] = []
    rng = random.Random(seed)
    for group_records in groups.values():
        ordered = sorted(group_records, key=lambda item: item["image_id"])
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        if split == "train":
            selected.extend(ordered[:n_train])
        elif split == "val":
            selected.extend(ordered[n_train : n_train + n_val])
        else:
            selected.extend(ordered[n_train + n_val :])

    selected.sort(key=lambda item: item["image_id"])
    return selected


def _decode_labelme_polygon_mask(
    annotation_path: Path,
    height: int,
    width: int,
) -> np.ndarray:
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, outline=1, fill=1)

    return np.array(mask_image, dtype=bool)


class BTXRDSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        augment: bool = False,
        use_clahe: bool = False,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        split_seed: int = DEFAULT_SPLIT_SEED,
        pred_mask_dir: str | Path | None = None,
        split_manifest: str | Path | None = None,
    ) -> None:
        self.btxrd_root = resolve_btxrd_root(root)
        self.images_dir = self.btxrd_root / DEFAULT_IMAGES_DIR
        self.annotations_dir = self.btxrd_root / DEFAULT_ANNOTATIONS_DIR
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe
        self.pred_mask_dir = Path(pred_mask_dir) if pred_mask_dir is not None else None
        self.pseudo_manifest_info: dict[str, object] | None = None

        records = load_btxrd_records(self.btxrd_root, split_manifest=split_manifest)
        self.samples = split_btxrd_records(records, split=split, ratios=split_ratios, seed=split_seed)
        self.split = split

        if not self.samples:
            raise FileNotFoundError(
                f"No BTXRD samples found for split '{split}'. Check dataset root: {self.btxrd_root}"
            )

        missing_images = [
            sample["image_id"] for sample in self.samples if not (self.images_dir / sample["image_id"]).exists()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} BTXRD images referenced in dataset.csv/xlsx are missing under "
                f"{self.images_dir}, e.g. {missing_images[:5]}"
            )

        if self.pred_mask_dir is not None:
            if not self.pred_mask_dir.is_dir():
                raise FileNotFoundError(f"Pseudo-mask directory does not exist: {self.pred_mask_dir}")
            missing_masks = [
                str(sample["image_id"])
                for sample in self.samples
                if not self._pred_mask_path(str(sample["image_id"])).is_file()
            ]
            if missing_masks:
                raise FileNotFoundError(
                    f"{len(missing_masks)} pseudo-masks are missing for BTXRD split {split!r} under "
                    f"{self.pred_mask_dir}. A complete generation run must write one PNG per image, "
                    f"including explicit all-zero masks for skipped normal/low-confidence images; "
                    f"e.g. {missing_masks[:5]}"
                )
            from pseudo.manifest import validate_pseudo_mask_manifest

            self.pseudo_manifest_info = validate_pseudo_mask_manifest(
                self.pred_mask_dir,
                self.samples,
                split=split,
                image_size=None,
            )
            source_image_size = int(self.pseudo_manifest_info["source_image_size"])
            self.pseudo_manifest_info["consumer_image_size"] = int(image_size)
            self.pseudo_manifest_info["resized_for_consumer"] = (
                source_image_size != int(image_size)
            )

        self.image_transform = make_segmentation_image_transform(image_size)
        self.mask_transform = make_segmentation_mask_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _annotation_path(self, image_id: str) -> Path:
        return self.annotations_dir / f"{Path(image_id).stem}.json"

    def _pred_mask_path(self, image_id: str) -> Path:
        assert self.pred_mask_dir is not None
        return self.pred_mask_dir / f"{Path(image_id).stem}.png"

    def _build_mask(self, sample: dict[str, object], image_size: tuple[int, int]) -> Image.Image:
        width, height = image_size
        if self.pred_mask_dir is not None:
            pred_mask_path = self._pred_mask_path(str(sample["image_id"]))
            if not pred_mask_path.is_file():
                raise FileNotFoundError(f"Pseudo-mask disappeared after dataset validation: {pred_mask_path}")
            return Image.open(pred_mask_path).convert("L").resize((width, height), Image.NEAREST)
        elif not sample["tumor"]:
            mask = np.zeros((height, width), dtype=bool)
        else:
            annotation_path = self._annotation_path(str(sample["image_id"]))
            if not annotation_path.exists():
                raise FileNotFoundError(
                    f"Expected BTXRD annotation for tumor image but none found: {annotation_path}"
                )
            mask = _decode_labelme_polygon_mask(annotation_path, height=height, width=width)
        return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.images_dir / str(sample["image_id"])
        image = Image.open(image_path).convert("RGB")
        mask = self._build_mask(sample, image.size)

        if self.use_clahe:
            image = apply_clahe(image)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        return image_tensor, mask_tensor, str(sample["image_id"])


class BTXRDClassificationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        target_columns: Sequence[str] = ("tumor",),
        image_size: int = 512,
        use_clahe: bool = False,
        augment: bool = False,
        preprocessing_mode: str = "none",
        normalization: str = "imagenet",
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        split_seed: int = DEFAULT_SPLIT_SEED,
        split_manifest: str | Path | None = None,
    ) -> None:
        self.btxrd_root = resolve_btxrd_root(root)
        self.images_dir = self.btxrd_root / DEFAULT_IMAGES_DIR
        self.target_columns = list(target_columns)
        if self.target_columns not in (["tumor"], ["tumor_type"]):
            raise ValueError(
                "BTXRDClassificationDataset supports target_columns=['tumor'] (binary "
                "tumor-vs-normal WSSS label) or ['tumor_type'] (10-class single-label: "
                f"{list(TUMOR_TYPE_CLASS_NAMES)})."
            )
        self.is_tumor_type = self.target_columns == ["tumor_type"]
        self.use_clahe = use_clahe
        self.augment = augment
        self.preprocessing_mode = "clahe" if use_clahe and preprocessing_mode == "none" else preprocessing_mode

        records = load_btxrd_records(self.btxrd_root, split_manifest=split_manifest)
        self.samples = split_btxrd_records(records, split=split, ratios=split_ratios, seed=split_seed)
        self.split = split

        if not self.samples:
            raise FileNotFoundError(
                f"No BTXRD samples found for split '{split}'. Check dataset root: {self.btxrd_root}"
            )

        missing_images = [
            sample["image_id"] for sample in self.samples if not (self.images_dir / sample["image_id"]).exists()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} BTXRD images referenced in dataset.csv/xlsx are missing under "
                f"{self.images_dir}, e.g. {missing_images[:5]}"
            )

        self.image_transform = make_classification_transform(
            image_size,
            augment=augment,
            preprocessing_mode=self.preprocessing_mode,
            normalization=normalization,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.images_dir / str(sample["image_id"])
        image = Image.open(image_path).convert("RGB")
        if self.use_clahe and self.preprocessing_mode != "clahe":
            image = apply_clahe(image)
        if self.is_tumor_type:
            # Long scalar class index for nn.CrossEntropyLoss, not a
            # multi-hot float vector -- this is single-label multi-class,
            # not multi-label like the ["tumor"] binary mode.
            target = torch.tensor(int(sample["tumor_type"]), dtype=torch.long)
        else:
            target = torch.tensor([float(sample["tumor"])], dtype=torch.float32)
        return self.image_transform(image), target, str(sample["image_id"])
