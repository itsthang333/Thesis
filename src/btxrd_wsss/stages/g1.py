from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from btxrd_wsss.artifacts import load_descriptors, load_gallery
from btxrd_wsss.config import PipelineConfig
from btxrd_wsss.data.manifest import ImageRecord, read_manifest
from btxrd_wsss.evaluation.stage_report import StageReportWriter
from btxrd_wsss.io import atomic_json, seed_everything
from btxrd_wsss.models.rad_dino_g1 import G1Scorer, g1_mil_loss, smooth_bag_logit
from btxrd_wsss.pipeline.selection import select_final, unions_with_logits


def _selector_records(config: PipelineConfig) -> tuple[list[ImageRecord], list[ImageRecord]]:
    records = read_manifest(config.data.manifest, data_root=config.data.root)
    holdout = [
        record
        for record in records
        if record.split == "train" and record.fold == config.experiment.selector_holdout_fold
    ]
    train, validation = [], []
    for record in holdout:
        value = int(hashlib.sha1(record.group_id.encode()).hexdigest()[:8], 16) % 5
        (validation if value == 0 else train).append(record)
    if not train or not validation:
        raise ValueError("Selector holdout cannot produce non-empty G1 train/validation sets")
    return train, validation


def _load_bag(output_dir: Path, record: ImageRecord) -> torch.Tensor | None:
    values, identifiers = load_descriptors(output_dir, record.image_id)
    if not len(values):
        return None
    gallery = load_gallery(output_dir, record.image_id)
    if identifiers != tuple(item.candidate_id for item in gallery):
        raise RuntimeError(f"Descriptor/gallery mismatch for {record.image_id}")
    return torch.from_numpy(values)


@torch.inference_mode()
def _evaluate(
    model: G1Scorer,
    records: list[ImageRecord],
    output_dir: Path,
    device: torch.device,
    temperature: float,
) -> dict[str, float]:
    labels, probabilities = [], []
    model.eval()
    for record in records:
        bag = _load_bag(output_dir, record)
        if bag is None:
            probability = 0.0
        else:
            logit = smooth_bag_logit(model(bag.to(device)), temperature)
            probability = float(torch.sigmoid(logit).cpu())
        labels.append(int(record.is_tumor))
        probabilities.append(probability)
    metrics = {"auprc": float(average_precision_score(labels, probabilities))}
    metrics["auroc"] = (
        float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else float("nan")
    )
    return metrics


def train_g1(config: PipelineConfig) -> Path:
    seed_everything(config.experiment.seed)
    output_dir = Path(config.experiment.output_dir)
    train, validation = _selector_records(config)
    example = None
    for record in train:
        example = _load_bag(output_dir, record)
        if example is not None:
            break
    if example is None:
        raise ValueError("No non-empty descriptor bags are available for G1")
    device = torch.device(config.runtime.device)
    model = G1Scorer(example.shape[1], config.g1.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.g1.learning_rate, weight_decay=config.g1.weight_decay
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "g1_best.pt"
    best, patience = -np.inf, 0
    history: list[dict[str, float]] = []
    for epoch in range(config.g1.epochs):
        model.train()
        random.Random(config.experiment.seed + epoch).shuffle(train)
        losses = []
        for record in train:
            bag = _load_bag(output_dir, record)
            if bag is None:
                continue
            logits = model(bag.to(device))
            loss, _ = g1_mil_loss(
                logits,
                torch.tensor(float(record.is_tumor), device=device),
                temperature=config.g1.bag_temperature,
                negative_instance_weight=config.g1.negative_instance_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = _evaluate(model, validation, output_dir, device, config.g1.bag_temperature)
        row = {"epoch": float(epoch + 1), "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        if metrics["auprc"] > best:
            best, patience = metrics["auprc"], 0
            torch.save(
                {"model": model.state_dict(), "input_dim": example.shape[1], "metrics": metrics},
                checkpoint,
            )
        else:
            patience += 1
        atomic_json(checkpoint_dir / "g1_history.json", history)
        if patience >= config.g1.early_stopping_patience:
            break
    return checkpoint


def load_g1(config: PipelineConfig) -> G1Scorer:
    device = torch.device(config.runtime.device)
    payload = torch.load(
        Path(config.experiment.output_dir) / "checkpoints/g1_best.pt",
        map_location=device,
        weights_only=True,
    )
    model = G1Scorer(int(payload["input_dim"]), config.g1.hidden_dim).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def run_final_selection(config: PipelineConfig, *, splits: set[str] | None = None) -> None:
    output_dir = Path(config.experiment.output_dir)
    records = read_manifest(config.data.manifest, data_root=config.data.root)
    if splits:
        records = [record for record in records if record.split in splits]
    model = load_g1(config)
    device = next(model.parameters()).device
    report = StageReportWriter(output_dir, "final_selection")
    completed = report.completed_ids() if config.experiment.resume else set()
    masks_dir = output_dir / "final_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        mask_path = masks_dir / f"{record.image_id}.png"
        if record.image_id in completed and mask_path.exists():
            continue
        candidates = load_gallery(output_dir, record.image_id)
        values, identifiers = load_descriptors(output_dir, record.image_id)
        if not candidates:
            with Image.open(record.image_path) as handle:
                mask = np.zeros((handle.height, handle.width), bool)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            report.append(
                {
                    "image_id": record.image_id,
                    "candidate_id": "empty",
                    "bag_probability": 0.0,
                    "uncertainty": 1.0,
                }
            )
            continue
        if identifiers != tuple(item.candidate_id for item in candidates):
            raise RuntimeError(f"Descriptor/gallery mismatch for {record.image_id}")
        with torch.inference_mode():
            logits = model(torch.from_numpy(values).to(device)).float().cpu().numpy()
        candidates, logits = unions_with_logits(candidates, logits, config.selection)
        selection = select_final(
            record.image_id,
            candidates,
            logits,
            config.selection,
            bag_temperature=config.g1.bag_temperature,
        )
        if selection.bag_probability < config.evaluation.threshold:
            selection = replace(selection, candidate_id="empty", mask=np.zeros_like(selection.mask))
        Image.fromarray(selection.mask.astype(np.uint8) * 255).save(mask_path)
        report.append(
            {
                "image_id": record.image_id,
                "split": record.split,
                "candidate_id": selection.candidate_id,
                "probability": selection.probability,
                "bag_probability": selection.bag_probability,
                "uncertainty": selection.uncertainty,
                "evidence": selection.evidence,
                "predicted_area_ratio": float(selection.mask.mean()),
            }
        )
    report.write_numeric_summary()
