from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from btxrd_wsss.artifacts import load_gallery, load_source_maps
from btxrd_wsss.config import PipelineConfig
from btxrd_wsss.data.manifest import read_manifest
from btxrd_wsss.evaluation.candidates import candidate_gallery_metrics
from btxrd_wsss.evaluation.ground_truth import load_labelme_mask
from btxrd_wsss.evaluation.segmentation import segmentation_metrics
from btxrd_wsss.evaluation.stage_report import StageReportWriter
from btxrd_wsss.pipeline.sam_gallery import select_diverse_gallery


def _summary(records_path: Path) -> dict[str, object]:
    rows = [
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line
    ]
    numeric: dict[str, list[float]] = {}

    def collect(prefix: str, payload: dict[str, object]) -> None:
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                collect(name, value)
            elif (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and np.isfinite(value)
            ):
                numeric.setdefault(name, []).append(float(value))

    for row in rows:
        collect("", row)
    return {
        "image_count": len(rows),
        "mean": {key: float(np.mean(value)) for key, value in numeric.items()},
    }


def evaluate_spatial_stages(
    config: PipelineConfig,
    *,
    splits: set[str] | None = None,
) -> None:
    """Evaluation-only entry point; annotations never enter a training or selection stage."""
    splits = {"val", "test"} if splits is None else splits
    output_dir = Path(config.experiment.output_dir)
    records = [
        record
        for record in read_manifest(config.data.manifest, data_root=config.data.root)
        if record.split in splits
    ]
    report = StageReportWriter(output_dir, "spatial_audit")
    completed = report.completed_ids() if config.experiment.resume else set()
    for record in records:
        if record.image_id in completed:
            continue
        annotation = Path(config.data.root) / "Annotations" / f"{record.image_id}.json"
        if not annotation.exists():
            continue
        with Image.open(record.image_path) as image:
            target = load_labelme_mask(annotation, height=image.height, width=image.width)
        maps, _ = load_source_maps(output_dir, record.image_id)
        source_metrics: dict[str, object] = {}
        for source, evidence in maps.items():
            metrics = segmentation_metrics(evidence >= config.evaluation.threshold, target)
            peak = np.unravel_index(np.argmax(evidence), evidence.shape)
            source_metrics[source] = {
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "recall": metrics["recall"],
                "complete_miss": metrics["complete_miss"],
                "pointing_accuracy": bool(target[peak]) if target.any() else None,
            }
        raw = load_gallery(output_dir / "raw", record.image_id)
        selected = load_gallery(output_dir, record.image_id)
        cap_metrics: dict[str, object] = {}
        for cap in (24, 36, 48, 72):
            cap_config = replace(config.sam, maximum_selected_candidates=cap)
            subset = select_diverse_gallery(
                raw, maps, sam_config=cap_config, selection_config=config.selection
            )
            cap_metrics[str(cap)] = candidate_gallery_metrics(subset, target)
        final_path = output_dir / "final_masks" / f"{record.image_id}.png"
        final_metrics = None
        if final_path.exists():
            with Image.open(final_path) as image:
                final_metrics = segmentation_metrics(np.asarray(image) > 0, target)
        area_ratio = float(target.mean())
        lesion_size = (
            "tiny"
            if area_ratio < config.sam.tiny_area_ratio
            else "small"
            if area_ratio < config.sam.small_area_ratio
            else "large"
        )
        report.append(
            {
                "image_id": record.image_id,
                "split": record.split,
                "lesion_size": lesion_size,
                "target_area_ratio": area_ratio,
                "sources": source_metrics,
                "raw_gallery": candidate_gallery_metrics(raw, target),
                "selected_gallery": candidate_gallery_metrics(selected, target),
                "cap_ablation": cap_metrics,
                "final": final_metrics,
            }
        )
    if report.records_path.exists():
        report.write_summary(_summary(report.records_path))
