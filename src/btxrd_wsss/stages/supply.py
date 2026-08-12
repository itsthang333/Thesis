from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from btxrd_wsss.artifacts import (
    descriptor_path,
    gallery_path,
    load_gallery,
    load_source_maps,
    save_descriptors,
    save_gallery,
    save_source_maps,
    source_map_path,
)
from btxrd_wsss.config import PipelineConfig
from btxrd_wsss.data.images import load_native_grayscale
from btxrd_wsss.data.manifest import ImageRecord, read_manifest
from btxrd_wsss.evaluation.classification import classification_metrics
from btxrd_wsss.evaluation.stage_report import StageReportWriter
from btxrd_wsss.models.biomedclip import FrozenBiomedCLIP
from btxrd_wsss.models.rad_dino_g1 import FrozenRadDINODescriptor
from btxrd_wsss.pipeline.proposals import ProposalGenerator
from btxrd_wsss.pipeline.sam_gallery import (
    SAMViTBROIBackend,
    build_adaptive_gallery,
    select_diverse_gallery,
)
from btxrd_wsss.stages.hrnet import (
    calibrate_hrnet,
    empirical_cdf,
    infer_hrnet_sources,
    load_trained_hrnet,
)


def _records(
    config: PipelineConfig, splits: set[str] | None, limit: int | None = None
) -> list[ImageRecord]:
    records = read_manifest(config.data.manifest, data_root=config.data.root)
    records = records if not splits else [record for record in records if record.split in splits]
    if limit is not None and len(records) > limit:
        records = random.Random(config.experiment.seed).sample(records, limit)
    return records


def generate_source_maps(
    config: PipelineConfig, *, splits: set[str] | None = None, limit: int | None = None
) -> None:
    output_dir = Path(config.experiment.output_dir)
    hrnet = load_trained_hrnet(config)
    calibration_path = output_dir / "calibration/hrnet_normal_cdf.npz"
    if not calibration_path.exists():
        calibrate_hrnet(config, hrnet)
    with np.load(calibration_path) as payload:
        references = {source: payload[source] for source in ("hrnet_full", "hrnet_tile")}
    biomed = FrozenBiomedCLIP.from_pretrained(config.biomedclip.model_id, config.runtime.device)
    report = StageReportWriter(output_dir, "source_maps")
    completed = report.completed_ids() if config.experiment.resume else set()
    for record in _records(config, splits, limit):
        if record.image_id in completed and source_map_path(output_dir, record.image_id).exists():
            continue
        started = time.perf_counter()
        native = load_native_grayscale(record.image_path)
        raw_maps, raw_confidences = infer_hrnet_sources(hrnet, native, record.image_id, config)
        maps = {source: empirical_cdf(raw_maps[source], references[source]) for source in raw_maps}
        with Image.open(record.image_path) as handle:
            semantic = biomed.localize(
                handle,
                crop_fraction=config.biomedclip.crop_fraction,
                positions_per_axis=config.biomedclip.positions_per_axis,
                top_k_tiles=config.biomedclip.top_k_tiles,
            )
        maps["biomedclip"] = semantic.saliency
        confidence = 1 / (1 + math.exp(-10 * semantic.contrast_score))
        confidences = {
            source: float(np.clip(raw_confidences[source] * float(maps[source].max()), 0, 1))
            for source in ("hrnet_full", "hrnet_tile")
        }
        confidences["biomedclip"] = float(confidence)
        save_source_maps(output_dir, record.image_id, maps, confidences)
        report.append(
            {
                "image_id": record.image_id,
                "split": record.split,
                "label": int(record.is_tumor),
                "native_height": native.shape[0],
                "native_width": native.shape[1],
                "confidence": confidences,
                "peak": {key: float(value.max()) for key, value in maps.items()},
                "seconds": time.perf_counter() - started,
            }
        )
    rows = report.records()
    if rows:
        labels = np.asarray([row["label"] for row in rows])
        classification = {
            source: classification_metrics(
                labels, np.asarray([row["confidence"][source] for row in rows])
            )
            for source in ("hrnet_full", "hrnet_tile", "biomedclip")
        }
        report.write_numeric_summary({"classification": classification})


def generate_sam_galleries(
    config: PipelineConfig, *, splits: set[str] | None = None, limit: int | None = None
) -> None:
    output_dir = Path(config.experiment.output_dir)
    backend = SAMViTBROIBackend(config.sam.checkpoint, config.runtime.device, config.sam.model_type)
    generator = ProposalGenerator(config.proposals)
    report = StageReportWriter(output_dir, "sam_gallery")
    completed = report.completed_ids() if config.experiment.resume else set()
    for record in _records(config, splits, limit):
        if record.image_id in completed and gallery_path(output_dir, record.image_id).exists():
            continue
        started = time.perf_counter()
        image = load_native_grayscale(record.image_path)
        maps, confidences = load_source_maps(output_dir, record.image_id)
        proposals = generator.generate_all(maps, image_id=record.image_id, confidences=confidences)
        raw = build_adaptive_gallery(image, proposals, backend, config=config.sam)
        selected = select_diverse_gallery(
            raw,
            maps,
            sam_config=config.sam,
            selection_config=config.selection,
        )
        save_gallery(output_dir / "raw", record.image_id, raw)
        save_gallery(output_dir, record.image_id, selected)
        source_counts = {
            source: sum(item.proposal_source == source for item in selected)
            for source in confidences
        }
        size_counts = {
            "tiny": sum(
                item.mask.sum() / item.mask.size < config.sam.tiny_area_ratio for item in selected
            ),
            "small": sum(
                config.sam.tiny_area_ratio
                <= item.mask.sum() / item.mask.size
                < config.sam.small_area_ratio
                for item in selected
            ),
        }
        report.append(
            {
                "image_id": record.image_id,
                "split": record.split,
                "proposal_count": len(proposals),
                "raw_candidate_count": len(raw),
                "selected_candidate_count": len(selected),
                "source_counts": source_counts,
                "size_counts": size_counts,
                "seconds": time.perf_counter() - started,
            }
        )
    report.write_numeric_summary()


def generate_rad_dino_descriptors(
    config: PipelineConfig, *, splits: set[str] | None = None, limit: int | None = None
) -> None:
    output_dir = Path(config.experiment.output_dir)
    extractor = FrozenRadDINODescriptor(
        config.rad_dino.model_id,
        input_size=config.rad_dino.input_size,
        selected_layers=config.rad_dino.selected_layers,
        projection_dim=config.g1.projection_dim,
        batch_size=config.rad_dino.batch_size,
        device=config.runtime.device,
        seed=config.experiment.seed,
    )
    report = StageReportWriter(output_dir, "rad_dino")
    completed = report.completed_ids() if config.experiment.resume else set()
    for record in _records(config, splits, limit):
        if record.image_id in completed and descriptor_path(output_dir, record.image_id).exists():
            continue
        started = time.perf_counter()
        candidates = load_gallery(output_dir, record.image_id)
        if candidates:
            with Image.open(record.image_path) as image:
                batch = extractor.extract(image, candidates, config.rad_dino.context_scales)
            save_descriptors(output_dir, record.image_id, batch.values, batch.candidate_ids)
            dimension = int(batch.values.shape[1])
        else:
            save_descriptors(output_dir, record.image_id, np.empty((0, 0), np.float32), ())
            dimension = 0
        report.append(
            {
                "image_id": record.image_id,
                "split": record.split,
                "candidate_count": len(candidates),
                "descriptor_dimension": dimension,
                "seconds": time.perf_counter() - started,
            }
        )
    report.write_numeric_summary()
