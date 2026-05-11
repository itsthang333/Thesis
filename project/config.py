from __future__ import annotations

from dataclasses import dataclass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 25
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_ANATOMY_COLUMNS = ("hand", "leg", "hip", "shoulder")


@dataclass
class ClassifierConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_EPOCHS
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = DEFAULT_SEED


@dataclass
class SegmentationConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_EPOCHS
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = DEFAULT_SEED

