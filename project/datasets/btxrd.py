from __future__ import annotations

import json
import random
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

# 9 mutually-exclusive tumor-type columns in dataset.csv, plus "normal" (no
# tumor) as class 0. Order fixed here so class indices are stable across
# training runs and checkpoints. Verified against a real dataset.csv sample:
# sum of these 9 columns (1868) matches the tumor==1 count (1867) to within 1
# image, i.e. each tumor image has exactly one of these columns set to 1.
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
ANATOMY_REGION_COLUMNS = ("upper limb", "lower limb", "pelvis")
ANATOMY_REGION_NAMES = ANATOMY_REGION_COLUMNS


def resolve_btxrd_root(root: str | Path) -> Path:
    """Return the directory that directly contains images/, Annotations/, dataset.csv|xlsx."""
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
    """Load dataset.csv or dataset.xlsx into a list of row dicts, without requiring pandas."""
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
    value = row.get(column, 0)
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _row_tumor_type_index(row: dict[str, object]) -> int:
    """Return the 10-class tumor-type index: 0=normal, 1..9=TUMOR_TYPE_COLUMNS.

    If a tumor image has more than one tumor-type column set to 1 (a data
    quality edge case -- observed once in a real dataset.csv sample, where the
    9 tumor-type columns summed to 1868 vs. tumor==1 count of 1867), the
    first matching column in TUMOR_TYPE_COLUMNS order wins, deterministically.
    """
    for i, column in enumerate(TUMOR_TYPE_COLUMNS):
        if _row_flag(row, column):
            return i + 1
    return 0


def _row_anatomy_region_index(row: dict[str, object]) -> int:
    """Return the one coarse region available for both tumor and normal rows.

    Fine-grained bone columns are excluded because BTXRD populates them only
    for tumor images, which would leak the target label.
    """
    active = [i for i, column in enumerate(ANATOMY_REGION_COLUMNS) if _row_flag(row, column)]
    if len(active) != 1:
        image_id = str(row.get("image_id", "<unknown>"))
        raise ValueError(
            f"BTXRD image {image_id} must have exactly one coarse anatomy region; "
            f"found {[ANATOMY_REGION_COLUMNS[i] for i in active]}"
        )
    return active[0]


def load_btxrd_records(btxrd_root: str | Path) -> list[dict[str, object]]:
    """Return one record per image with the fields this project cares about."""
    btxrd_root = resolve_btxrd_root(btxrd_root)
    rows = _load_dataset_table(btxrd_root)
    records: list[dict[str, object]] = []
    for row in rows:
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        records.append(
            {
                "image_id": image_id,
                "tumor": _row_flag(row, "tumor"),
                "benign": _row_flag(row, "benign"),
                "malignant": _row_flag(row, "malignant"),
                "tumor_type": _row_tumor_type_index(row),
                "anatomy_region": _row_anatomy_region_index(row),
            }
        )
    return records


def split_btxrd_records(
    records: Sequence[dict[str, object]],
    split: str,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[dict[str, object]]:
    """Deterministic stratified 80/10/10 split by the 10-class tumor_type label.

    BTXRD ships with no predefined train/val/test split, so this project derives
    one locally. Stratifying on the full 10-class label (normal + 9 tumor
    types, see TUMOR_TYPE_CLASS_NAMES) keeps even the rarest classes (44-51
    images, e.g. osteofibroma/other_mt/synovial_osteochondroma) represented in
    every split -- verified this never rounds a class down to 0 images in any
    split for the real BTXRD class distribution (smallest split count is 4).
    A coarser normal/benign/malignant split was used before tumor_type
    existed; this supersedes it since tumor_type strictly refines that
    grouping (every benign/malignant image maps to exactly one tumor_type).
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test.")

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
    """Rasterize every polygon shape in a LabelMe JSON into one binary tumor mask.

    BTXRD stores two shapes per tumor instance (a rectangle bbox and a polygon
    outline) sharing the same label. Only polygons are rasterized; rectangles
    are redundant with the polygon's bounding extent and are intentionally
    ignored here (image-level labels drive the WSSS pipeline; this mask is only
    used as ground truth for evaluation, not as a shape input).
    """
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
    """BTXRD bone-tumor segmentation dataset.

    Ground-truth masks come from LabelMe polygon annotations (tumor lesion
    outline). Images without a tumor (tumor=0) have no annotation file and get
    an all-zero mask, which also lets pseudo-mask evaluation penalize false
    positives on normal images.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        augment: bool = False,
        use_clahe: bool = False,
        pseudo_mask_root: str | Path | None = None,
        require_all_pseudo_masks: bool = True,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        split_seed: int = DEFAULT_SPLIT_SEED,
    ) -> None:
        self.btxrd_root = resolve_btxrd_root(root)
        self.images_dir = self.btxrd_root / DEFAULT_IMAGES_DIR
        self.annotations_dir = self.btxrd_root / DEFAULT_ANNOTATIONS_DIR
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe
        self.pseudo_mask_root = Path(pseudo_mask_root) if pseudo_mask_root is not None else None

        records = load_btxrd_records(self.btxrd_root)
        self.samples = split_btxrd_records(records, split=split, ratios=split_ratios, seed=split_seed)
        self.split = split

        if self.pseudo_mask_root is not None:
            missing_pseudo = [
                sample["image_id"] for sample in self.samples
                if not (self.pseudo_mask_root / f"{Path(str(sample['image_id'])).stem}.png").exists()
            ]
            if missing_pseudo and require_all_pseudo_masks:
                raise FileNotFoundError(
                    f"Missing {len(missing_pseudo)} pseudo masks under {self.pseudo_mask_root}, "
                    f"e.g. {missing_pseudo[:5]}"
                )
            if missing_pseudo:
                missing_stems = {Path(str(name)).stem for name in missing_pseudo}
                self.samples = [
                    sample for sample in self.samples
                    if Path(str(sample["image_id"])).stem not in missing_stems
                ]

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

        self.image_transform = make_segmentation_image_transform(image_size)
        self.mask_transform = make_segmentation_mask_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _annotation_path(self, image_id: str) -> Path:
        return self.annotations_dir / f"{Path(image_id).stem}.json"

    def _build_mask(self, sample: dict[str, object], image_size: tuple[int, int]) -> Image.Image:
        if self.pseudo_mask_root is not None:
            pseudo_path = self.pseudo_mask_root / f"{Path(str(sample['image_id'])).stem}.png"
            # Crucially, this branch never opens annotations_dir. It is the
            # training path for weak supervision; LabelMe GT remains confined
            # to the fully-supervised/evaluation branch below.
            return Image.open(pseudo_path).convert("L")
        width, height = image_size
        if not sample["tumor"]:
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
    """Image-only BTXRD dataset for classifier training and Stage 2 pseudo-mask generation.

    Two supported label modes, selected via target_columns:
      - ["tumor"]: binary tumor-vs-normal image-level label for WSSS (the
        original mode -- classifier trained only on this weak label, LayerCAM/
        SAM localize the tumor without ever consuming polygon/bbox annotations).
      - ["tumor_type"]: 10-class single-label classification (normal + 9
        mutually-exclusive tumor types, see TUMOR_TYPE_CLASS_NAMES). Still WSSS
        in spirit -- the segmentation pipeline never sees polygon/bbox
        annotations either way -- but the classifier/LayerCAM class-conditioning
        signal is now the specific tumor type instead of just tumor-vs-normal.
    """

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
        include_anatomy_target: bool = False,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        split_seed: int = DEFAULT_SPLIT_SEED,
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
        self.include_anatomy_target = bool(include_anatomy_target)
        if self.include_anatomy_target and not self.is_tumor_type:
            raise ValueError("include_anatomy_target requires target_columns=['tumor_type']")
        self.use_clahe = use_clahe
        self.augment = augment
        self.preprocessing_mode = "clahe" if use_clahe and preprocessing_mode == "none" else preprocessing_mode

        records = load_btxrd_records(self.btxrd_root)
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
        transformed = self.image_transform(image)
        if self.include_anatomy_target:
            region = torch.tensor(int(sample["anatomy_region"]), dtype=torch.long)
            return transformed, target, region, str(sample["image_id"])
        return transformed, target, str(sample["image_id"])
