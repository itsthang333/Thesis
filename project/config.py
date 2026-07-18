from __future__ import annotations

from dataclasses import dataclass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 25
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_ANATOMY_COLUMNS = ("hand",)

SUPPORTED_DATASETS = ("ramh1200", "btxrd")
DEFAULT_DATASET = "ramh1200"
DATASET_TARGET_COLUMNS = {
    "ramh1200": ("hand",),
    "btxrd": ("tumor",),
}


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


@dataclass(frozen=True)
class BtxrdBestPipelineConfig:
    """Frozen, hardware-independent BTXRD WSSS configuration.

    The values are the single pipeline selected after the validation A/B
    runs.  Device and checkpoint paths are intentionally not part of this
    object: those are deployment-machine inputs, while the model/prompt/CAM
    behavior must remain identical when the project is moved to another
    workstation.
    """

    name: str = "btxrd_best"
    target_columns: tuple[str, ...] = ("tumor_type",)
    classifier_image_size: int = 320
    classifier_batch_size: int = 4
    classifier_epochs: int = 25
    classifier_lr: float = 1e-4
    classifier_weight_decay: float = 1e-4
    classifier_seed: int = 42
    classifier_puzzle_alpha_max: float = 0.0
    classifier_attention_alpha_max: float = 0.0
    cam_percentile: float = 90.0
    cam_percentile_values: tuple[float, ...] = (85.0, 90.0, 95.0)
    cam_contrast_normal: bool = True
    cam_contrast_weight: float = 1.0
    cam_aggregation: str = "class"
    layercam_weights: tuple[float, ...] = (0.20, 0.30, 0.50)
    layercam_gradient_mode: str = "positive"
    confidence_threshold: float = 0.5
    max_points: int = 5
    min_component_area: int = 100
    mask_score_threshold: float = 0.4
    morphology_fusion_mode: str = "components"
    sam_version: str = "v1"
    sam_model_type: str = "vit_b"
    sam_image_size: int = 512
    sam_preserve_aspect: bool = False
    sam_prompt_mode: str = "box_point"
    sam_prompt_ensemble: bool = True
    sam_single_mask: bool = False
    max_bone_components: int = 3
    all_cam_components: bool = True
    component_topk: int = 1
    points_per_component: int = 5
    bbox_padding_ratio: float = 0.02
    negative_points_per_component: int = 4
    prompt_border_margin: int = 2
    max_box_area_ratio: float = 0.35
    selection_method: str = "coverage_mass_sam"
    best_per_component: bool = True
    fusion_topk: int = 3
    support_clip_kernel: int = 5
    closing_kernel: int = 0
    opening_kernel: int = 0
    min_size: int = 40
    max_hole_area: int = 0
    guidance_threshold: float = 0.4


BTXRD_BEST_PIPELINE = BtxrdBestPipelineConfig()

