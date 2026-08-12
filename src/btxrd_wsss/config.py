from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    resume: bool
    selector_holdout_fold: int


@dataclass(frozen=True)
class DataConfig:
    root: str
    manifest: str
    image_dir: str
    table_names: list[str]
    tumor_columns: list[str]
    tile_sizes: list[int]
    tile_overlap: float
    pad_multiple: int


@dataclass(frozen=True)
class HRNetConfig:
    backbone: str
    pretrained: bool
    full_long_side: int
    network_tile_size: int
    tiles_per_image: int
    inference_tile_batch: int
    calibration_normal_images: int
    dense_channels: int
    dropout: float
    output_classes: int
    topk_fractions: list[float]
    epochs: int
    early_stopping_patience: int
    batch_size: int
    gradient_accumulation: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    warmup_epochs: int
    normal_suppression_weight: float
    full_tile_consistency_weight: float
    tile_fusion_max_weight: float


@dataclass(frozen=True)
class BiomedCLIPConfig:
    model_id: str
    crop_fraction: float
    positions_per_axis: int
    top_k_tiles: int
    frozen: bool


@dataclass(frozen=True)
class ProposalConfig:
    hrnet_full_percentiles: list[float]
    hrnet_tile_percentiles: list[float]
    biomedclip_percentiles: list[float]
    minimum_native_area: int
    max_components_per_threshold: int
    box_padding: list[float]
    positive_point_counts: list[int]
    negative_points: int
    source_quotas: dict[str, int]


@dataclass(frozen=True)
class SAMConfig:
    model_type: str
    checkpoint: str
    image_size: int
    initial_roi_scale: float
    expansion_roi_scale: float
    multimask: bool
    duplicate_iou: float
    expansion_roi_quotas: dict[str, int]
    gallery_minimum_quotas: dict[str, int]
    maximum_raw_candidates: int
    maximum_selected_candidates: int
    minimum_tiny_candidates: int
    minimum_small_candidates: int
    tiny_area_ratio: float
    small_area_ratio: float
    diversity_weight: float


@dataclass(frozen=True)
class RadDINOConfig:
    model_id: str
    input_size: int
    batch_size: int
    context_scales: list[float]
    selected_layers: list[int]
    frozen: bool


@dataclass(frozen=True)
class G1Config:
    projection_dim: int
    hidden_dim: int
    geometry_dim: int
    bag_temperature: float
    context_radius: int
    epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    negative_instance_weight: float
    self_guided_winner_loss: bool


@dataclass(frozen=True)
class SelectionConfig:
    hrnet_weights: dict[str, float]
    biomedclip_weights: dict[str, float]
    source_confidence_floor: float
    g1_rank_weight: float
    upstream_rank_weight: float
    minimum_mask_area: int
    maximum_mask_area_ratio: float
    minimum_stability: float
    minimum_component_coverage: float
    add_multifocal_unions: bool
    maximum_union_masks: int
    maximum_components_per_union: int


@dataclass(frozen=True)
class EvaluationConfig:
    threshold: float
    small_area_ratio: float
    medium_area_ratio: float
    save_native_maps: bool
    save_stage_jsonl: bool


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    precision: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    maximum_dollars_per_hour: float
    disk_gb: int


@dataclass(frozen=True)
class PipelineConfig:
    experiment: ExperimentConfig
    data: DataConfig
    hrnet: HRNetConfig
    biomedclip: BiomedCLIPConfig
    proposals: ProposalConfig
    sam: SAMConfig
    rad_dino: RadDINOConfig
    g1: G1Config
    selection: SelectionConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig


SECTIONS: dict[str, type[Any]] = {
    "experiment": ExperimentConfig,
    "data": DataConfig,
    "hrnet": HRNetConfig,
    "biomedclip": BiomedCLIPConfig,
    "proposals": ProposalConfig,
    "sam": SAMConfig,
    "rad_dino": RadDINOConfig,
    "g1": G1Config,
    "selection": SelectionConfig,
    "evaluation": EvaluationConfig,
    "runtime": RuntimeConfig,
}


def _construct(cls: type[T], payload: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(payload) - allowed
    missing = allowed - set(payload)
    if unknown or missing:
        raise ValueError(
            f"Invalid {cls.__name__}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return cls(**payload)


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(SECTIONS):
        actual = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(f"Config sections must be exactly {sorted(SECTIONS)}, got {actual}")
    sections = {name: _construct(cls, raw[name]) for name, cls in SECTIONS.items()}
    config = PipelineConfig(**sections)
    validate_config(config)
    return config


def _require_probability(value: float, name: str, *, open_zero: bool = False) -> None:
    valid = 0 < value <= 1 if open_zero else 0 <= value <= 1
    if not valid:
        bracket = "(0,1]" if open_zero else "[0,1]"
        raise ValueError(f"{name} must lie in {bracket}")


def validate_config(config: PipelineConfig) -> None:
    sources = {"hrnet_full", "hrnet_tile", "biomedclip"}
    if config.hrnet.output_classes != len(config.data.tumor_columns) + 1:
        raise ValueError("HRNet output_classes must equal normal plus tumor columns")
    if config.data.tile_sizes != [512, 1024]:
        raise ValueError("The frozen one-day method requires native tile sizes [512,1024]")
    if not 0 <= config.data.tile_overlap < 1:
        raise ValueError("tile_overlap must lie in [0,1)")
    if config.hrnet.tiles_per_image < 1:
        raise ValueError("tiles_per_image must be positive")
    if config.hrnet.inference_tile_batch < 1:
        raise ValueError("inference_tile_batch must be positive")
    if config.hrnet.calibration_normal_images < 1:
        raise ValueError("calibration_normal_images must be positive")
    _require_probability(config.hrnet.tile_fusion_max_weight, "tile_fusion_max_weight")
    if any(not 0 < value <= 1 for value in config.hrnet.topk_fractions):
        raise ValueError("top-k fractions must lie in (0,1]")
    if (
        set(config.proposals.source_quotas) != sources
        or set(config.sam.expansion_roi_quotas) != sources
        or set(config.sam.gallery_minimum_quotas) != sources
    ):
        raise ValueError("Proposal and SAM quotas must cover exactly the three frozen sources")
    for name, values in (
        ("hrnet_full_percentiles", config.proposals.hrnet_full_percentiles),
        ("hrnet_tile_percentiles", config.proposals.hrnet_tile_percentiles),
        ("biomedclip_percentiles", config.proposals.biomedclip_percentiles),
    ):
        if any(not 0 < value < 1 for value in values):
            raise ValueError(f"{name} must use probabilities in (0,1)")
    _require_probability(config.sam.duplicate_iou, "duplicate_iou", open_zero=True)
    _require_probability(config.sam.tiny_area_ratio, "tiny_area_ratio", open_zero=True)
    _require_probability(config.sam.small_area_ratio, "small_area_ratio", open_zero=True)
    if config.sam.tiny_area_ratio >= config.sam.small_area_ratio:
        raise ValueError("tiny_area_ratio must be smaller than small_area_ratio")
    if config.sam.maximum_selected_candidates > config.sam.maximum_raw_candidates:
        raise ValueError("Selected SAM gallery cannot exceed the raw gallery")
    if sum(config.sam.gallery_minimum_quotas.values()) > config.sam.maximum_selected_candidates:
        raise ValueError("Minimum source quotas exceed selected gallery size")
    if config.rad_dino.batch_size < 1:
        raise ValueError("RAD-DINO batch_size must be positive")
    if not (config.biomedclip.frozen and config.rad_dino.frozen):
        raise ValueError("BiomedCLIP and RAD-DINO must remain frozen in this protocol")
    if config.g1.self_guided_winner_loss:
        raise ValueError("Self-guided winner loss is disabled to avoid confirmation bias")
    _require_probability(config.selection.source_confidence_floor, "source_confidence_floor")
    _require_probability(
        config.selection.maximum_mask_area_ratio, "maximum_mask_area_ratio", open_zero=True
    )
    _require_probability(config.selection.minimum_stability, "minimum_stability")
    _require_probability(config.selection.minimum_component_coverage, "minimum_component_coverage")
    if abs(config.selection.g1_rank_weight + config.selection.upstream_rank_weight - 1) > 1e-9:
        raise ValueError("Final rank weights must sum to one")
    if abs(sum(config.selection.hrnet_weights.values()) - 1) > 1e-9:
        raise ValueError("HRNet upstream weights must sum to one")
    if abs(sum(config.selection.biomedclip_weights.values()) - 1) > 1e-9:
        raise ValueError("BiomedCLIP upstream weights must sum to one")
    if config.runtime.maximum_dollars_per_hour > 0.60:
        raise ValueError("One-day RTX 5090 configuration refuses prices above $0.60/hour")
