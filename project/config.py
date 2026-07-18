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
    classifier_epochs: int = 6
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

# btxrd_hybrid: same CAM/SAM/mask-selection recipe as btxrd_best (validated to
# give the higher oracle_dice of the two independent pipelines -- contrastive
# CAM, percentile ensemble, multi-component, SAM prompt ensemble at 512px),
# combined with the classifier training recipe validated on the other
# pipeline (25 epochs with early stopping, PuzzleCAM + Teacher-Student
# attention distillation) instead of btxrd_best's 6-epoch pure-CE classifier
# (val_f1=0.4251, visibly still improving at epoch 6 on the confusion
# matrix). btxrd_best's classifier CAM/SAM stages produced a much higher
# oracle_best_single_dice (0.52 vs 0.34) than the other pipeline's, but its
# own classifier was undertrained relative to the other pipeline's
# (val_f1=0.6774) -- this profile keeps the winning downstream recipe and
# swaps in the winning training recipe, to be evaluated on its own oracle
# diagnostics rather than assumed additive.
@dataclass(frozen=True)
class BtxrdHybridPipelineConfig(BtxrdBestPipelineConfig):
    name: str = "btxrd_hybrid"
    classifier_epochs: int = 25
    classifier_early_stop_patience: int = 7
    classifier_puzzle_alpha_max: float = 4.0
    classifier_attention_alpha_max: float = 0.01
    teacher_warmup_epochs: int = 3
    teacher_ema_decay: float = 0.999
    teacher_cam_percentile: float = 96.0


BTXRD_HYBRID_PIPELINE = BtxrdHybridPipelineConfig()

# btxrd_anatomy: anatomy-matched contrastive learning profile. Adds a region
# classification head, a region-conditioned tumor head, and an anatomy-
# matched contrastive loss on top of btxrd_hybrid's classifier recipe (same
# epochs/puzzle/attention settings), plus anatomy-conditioned CAM downstream.
# Loss weights below are the fixed values from this project's own anatomy-
# aware design doc (L = L_type + 0.2*L_anatomy + 0.1*L_region-tumor +
# 0.05*L_contrast, 3-epoch contrastive warmup) -- CLI flags with the same
# names still exist for A/B ablation, but this profile is what pins them to
# those exact values instead of leaving them at the CLI's off-by-default 0.0.
@dataclass(frozen=True)
class BtxrdAnatomyPipelineConfig(BtxrdHybridPipelineConfig):
    name: str = "btxrd_anatomy"
    classifier_anatomy_region_alpha: float = 0.2
    classifier_anatomy_region_tumor_alpha: float = 0.1
    classifier_anatomy_contrastive_alpha: float = 0.05
    classifier_anatomy_contrastive_warmup_epochs: int = 3
    cam_anatomy_beta: float = 0.5
    cam_anatomy_weight: float = 1.0
    anatomy_consistency_weight: float = 0.2
    unet_consistency_weight: float = 0.1
    unet_confidence_boundary_width: int = 3
    unet_consistency_confidence_threshold: float = 0.80


BTXRD_ANATOMY_PIPELINE = BtxrdAnatomyPipelineConfig()

