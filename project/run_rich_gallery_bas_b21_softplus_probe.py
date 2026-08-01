from __future__ import annotations

"""Bounded image-label-only mechanics probe for the BAS binary correction.

This runner changes exactly one scientific component relative to BAS-B2: the
terminal nonnegative class-map activation is Softplus instead of ReLU. It runs
five canonical train passes, freezes label-safe validation maps and candidate
area diagnostics, and stops. It never opens validation polygons or test data.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.factory import build_classification_dataset
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.bas_candidate_localizer import (
    BASResNet50Localizer,
    candidate_activation_evidence,
)
from models.rich_gallery_g2_objective import average_percentile_rank
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_bas_candidate_descriptor_b1 import (
    EXPECTED_BACKBONE_LR,
    EXPECTED_BATCH_SIZE,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_MOMENTUM,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_WEIGHT_DECAY,
    _active_state_dict,
    _binary_metrics,
    _optimizer,
    _train,
    _verify_g1_stage_a,
)


EXPERIMENT_ID = "EXP-20260802-codex-rich-gallery-bas-b21-softplus-probe-v1"
EXPECTED_EPOCHS = 5
EXPECTED_ACTIVATION = "softplus"
TRAIN_IMAGES = 2981
TRAIN_NORMALS = 1493
TRAIN_TUMORS = 1488

GATES = {
    "final_full_ce_max": 0.69,
    "final_accuracy_prior_distance_min": 0.01,
    "validation_auroc_min": 0.55,
    "activation_range_mean_min": 1.0e-3,
    "tumor_nondegenerate_fraction_min": 0.50,
    "tumor_bas_area_spearman_mean_max": 0.98,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--g1-stage-a-root", type=Path, required=True)
    parser.add_argument("--expected-g1-stage-a-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--backbone-lr", type=float, default=EXPECTED_BACKBONE_LR)
    parser.add_argument("--weight-decay", type=float, default=EXPECTED_WEIGHT_DECAY)
    parser.add_argument("--momentum", type=float, default=EXPECTED_MOMENTUM)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write empty BAS probe table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@torch.inference_mode()
def _validation_probe(
    model: BASResNet50Localizer,
    loader: DataLoader,
) -> tuple[dict[str, np.ndarray], dict[str, float], list[dict[str, object]]]:
    model.eval()
    activations: dict[str, np.ndarray] = {}
    labels: list[int] = []
    probabilities: list[float] = []
    ranges: list[float] = []
    tumor_ranges: list[float] = []
    rows: list[dict[str, object]] = []
    for images, targets, image_ids in loader:
        images = images.cuda(non_blocking=True)
        logits, maps = model.classify_and_tumor_activation(images)
        flip_logits, flip_maps = model.classify_and_tumor_activation(images.flip(-1))
        logits = 0.5 * (logits + flip_logits)
        maps = 0.5 * (maps + flip_maps.flip(-1))
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        for index, image_id in enumerate(image_ids):
            key = str(image_id)
            activation = maps[index, 0].float().cpu().numpy().astype(np.float32)
            if key in activations or not np.isfinite(activation).all():
                raise RuntimeError(f"invalid/duplicate BAS activation: {key}")
            label = int(targets[index].reshape(-1)[0].item())
            probability = float(probs[index].item())
            activation_range = float(np.ptp(activation))
            activations[key] = activation
            labels.append(label)
            probabilities.append(probability)
            ranges.append(activation_range)
            if label:
                tumor_ranges.append(activation_range)
            rows.append(
                {
                    "image_id": key,
                    "tumor": label,
                    "tumor_probability": probability,
                    "activation_range": activation_range,
                }
            )
    if len(rows) != 371 or len(tumor_ranges) != 184:
        raise RuntimeError("BAS Softplus validation probe cohort mismatch")
    diagnostics = {
        **_binary_metrics(np.asarray(labels), np.asarray(probabilities)),
        "activation_range_mean": float(np.mean(ranges)),
        "tumor_nondegenerate_activation_fraction": float(
            np.mean(np.asarray(tumor_ranges) > 1.0e-4)
        ),
    }
    return activations, diagnostics, rows


def _freeze_label_safe_diagnostics(
    *,
    output_dir: Path,
    val_rows: list[dict[str, str]],
    activations: Mapping[str, np.ndarray],
    g1_rows: Mapping[str, Mapping[str, str]],
    g1_stage_a_root: Path,
    candidate_rows: Mapping[str, Mapping[str, str]],
    candidate_root: Path,
) -> tuple[str, dict[str, float]]:
    activation_root = output_dir / "activation_maps"
    activation_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    tumor_correlations: list[float] = []
    for row in val_rows:
        image_id = row["image_id"]
        stem = Path(image_id).stem
        activation = np.asarray(activations[image_id], dtype=np.float32)
        activation_path = activation_root / f"{stem}.npy"
        np.save(activation_path, activation, allow_pickle=False)

        frozen = g1_rows[image_id]
        score_path = g1_stage_a_root / frozen["score_path"]
        if sha256_file(score_path) != frozen["score_sha256"]:
            raise ValueError(f"G1 score payload changed: {image_id}")
        candidate_row = candidate_rows[stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(score_path, allow_pickle=False) as scored:
            candidate_indices = scored["candidate_indices"].astype(np.int64)
        with np.load(candidate_path, allow_pickle=False) as candidate:
            masks = candidate["sam_masks"].astype(np.float32)[candidate_indices]
        tensor_masks = torch.from_numpy(masks)[None]
        valid = torch.ones((1, len(masks)), dtype=torch.bool)
        with torch.inference_mode():
            _coverage, _purity, harmonic = candidate_activation_evidence(
                torch.from_numpy(activation)[None, None],
                tensor_masks,
                valid,
            )
        bas_scores = harmonic[0].numpy().astype(np.float64)
        areas = masks.mean(axis=(1, 2), dtype=np.float64)
        correlation = _rank_correlation(bas_scores, areas)
        tumor = int(row["tumor"])
        if tumor:
            tumor_correlations.append(correlation)
        rows.append(
            {
                "image_id": image_id,
                "group_id": row["group_id"],
                "tumor": tumor,
                "activation_path": str(activation_path.relative_to(output_dir)).replace(
                    "\\", "/"
                ),
                "activation_sha256": sha256_file(activation_path),
                "activation_min": float(activation.min()),
                "activation_max": float(activation.max()),
                "activation_mean": float(activation.mean()),
                "activation_std": float(activation.std()),
                "candidate_count": len(candidate_indices),
                "bas_area_spearman": correlation,
            }
        )
    if len(rows) != 371 or len(tumor_correlations) != 184:
        raise RuntimeError("BAS Softplus probe validation cohort mismatch")
    manifest_sha = _write_csv(output_dir / "activation_manifest.csv", rows)
    diagnostics = {
        "tumor_bas_area_spearman_mean": float(np.mean(tumor_correlations)),
        "tumor_bas_area_spearman_median": float(np.median(tumor_correlations)),
        "tumor_bas_area_spearman_fraction_above_0_9": float(
            np.mean(np.asarray(tumor_correlations) > 0.9)
        ),
    }
    return manifest_sha, diagnostics


def _mechanics_gate(
    history: list[dict[str, float | int]],
    validation: Mapping[str, float],
    area: Mapping[str, float],
) -> dict[str, object]:
    final = history[-1]
    class_prior_accuracy = TRAIN_NORMALS / TRAIN_IMAGES
    values = {
        "final_full_ce": float(final["full_ce"]),
        "final_accuracy": float(final["accuracy"]),
        "final_accuracy_prior_distance": abs(
            float(final["accuracy"]) - class_prior_accuracy
        ),
        "validation_auroc": float(validation["auroc"]),
        "activation_range_mean": float(validation["activation_range_mean"]),
        "tumor_nondegenerate_fraction": float(
            validation["tumor_nondegenerate_activation_fraction"]
        ),
        "tumor_bas_area_spearman_mean": float(
            area["tumor_bas_area_spearman_mean"]
        ),
    }
    checks = {
        "final_full_ce": values["final_full_ce"] <= GATES["final_full_ce_max"],
        "final_accuracy_prior_distance": values["final_accuracy_prior_distance"]
        >= GATES["final_accuracy_prior_distance_min"],
        "validation_auroc": values["validation_auroc"]
        >= GATES["validation_auroc_min"],
        "activation_range_mean": values["activation_range_mean"]
        >= GATES["activation_range_mean_min"],
        "tumor_nondegenerate_fraction": values["tumor_nondegenerate_fraction"]
        >= GATES["tumor_nondegenerate_fraction_min"],
        "tumor_bas_area_spearman_mean": values["tumor_bas_area_spearman_mean"]
        <= GATES["tumor_bas_area_spearman_mean_max"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "values": values,
        "thresholds": GATES,
        "meaning": "technical_continuation_gate_not_spatial_efficacy",
    }


def main() -> None:
    args = parse_args()
    if (
        args.epochs != EXPECTED_EPOCHS
        or args.batch_size != EXPECTED_BATCH_SIZE
        or args.backbone_lr != EXPECTED_BACKBONE_LR
        or args.weight_decay != EXPECTED_WEIGHT_DECAY
        or args.momentum != EXPECTED_MOMENTUM
        or args.seed != 42
        or args.expected_pretrained_sha256 != EXPECTED_PRETRAINED_SHA256
    ):
        raise ValueError("BAS Softplus mechanics probe differs from frozen controls")
    if args.output_dir.exists():
        raise FileExistsError("BAS Softplus probe output directory must not exist")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")
    if sha256_file(args.pretrained_checkpoint) != args.expected_pretrained_sha256:
        raise ValueError("ImageNet ResNet-50 SHA-256 mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("BAS Softplus probe requires exactly two CUDA devices")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    _seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    train_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="train",
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(train_rows) != TRAIN_IMAGES or len(val_rows) != 371:
        raise RuntimeError("canonical BAS Softplus probe cohort mismatch")
    g1_freeze, g1_rows = _verify_g1_stage_a(
        args.g1_stage_a_root,
        expected_freeze_sha256=args.expected_g1_stage_a_freeze_sha256,
        expected_split_sha256=args.expected_split_sha256,
        expected_val_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_val_pseudo_sha256=args.val_pseudo_manifest_sha256,
    )
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all" or len(candidate_rows) != 371:
        raise ValueError("BAS Softplus probe requires all validation candidate bags")

    train_dataset = build_classification_dataset(
        root=args.dataset_root,
        split="train",
        target_columns=("tumor",),
        image_size=EXPECTED_IMAGE_SIZE,
        augment=True,
        normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    val_dataset = build_classification_dataset(
        root=args.dataset_root,
        split="val",
        target_columns=("tumor",),
        image_size=EXPECTED_IMAGE_SIZE,
        augment=False,
        normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    state = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
    model = BASResNet50Localizer(
        pretrained=False,
        backbone_state_dict=state,
        classifier_activation=EXPECTED_ACTIVATION,
    ).cuda()
    parallel = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = _optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history = _train(parallel, train_loader, optimizer, scaler, epochs=args.epochs)
    history_sha = _write_csv(args.output_dir / "training_history.csv", history)
    activations, validation_diagnostics, validation_rows = _validation_probe(
        model,
        val_loader,
    )
    validation_prediction_sha = _write_csv(
        args.output_dir / "validation_predictions.csv",
        validation_rows,
    )
    activation_manifest_sha, area_diagnostics = _freeze_label_safe_diagnostics(
        output_dir=args.output_dir,
        val_rows=val_rows,
        activations=activations,
        g1_rows=g1_rows,
        g1_stage_a_root=args.g1_stage_a_root,
        candidate_rows=candidate_rows,
        candidate_root=args.val_candidate_root,
    )
    gate = _mechanics_gate(history, validation_diagnostics, area_diagnostics)
    checkpoint_path = args.output_dir / "bas_b21_softplus_probe_fp16.pt"
    torch.save(
        {
            "model_state_dict": _active_state_dict(model),
            "experiment_id": EXPERIMENT_ID,
            "classifier_activation": EXPECTED_ACTIVATION,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "epochs": args.epochs,
            "training_labels": "image_level_normal_tumor_only",
            "mechanics_gate_pass": gate["pass"],
            "validation_gt_read": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    summary = {
        "stage": "rich_gallery_bas_b21_softplus_mechanics_probe_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "classifier_activation": EXPECTED_ACTIVATION,
        "scientific_delta_from_b2": "terminal_relu_to_softplus_only",
        "epochs": args.epochs,
        "training_history_sha256": history_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "activation_manifest_sha256": activation_manifest_sha,
        "validation_predictions_sha256": validation_prediction_sha,
        "validation_diagnostics": validation_diagnostics,
        "candidate_area_diagnostics": area_diagnostics,
        "mechanics_gate": gate,
        "full_training_authorized": bool(gate["pass"]),
        "spatial_efficacy_evaluated": False,
        "g1_stage_a_freeze_sha256": args.expected_g1_stage_a_freeze_sha256,
        "g1_stage_a_source_commit": g1_freeze["source_commit"],
        "candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "validation_images": 371,
        "validation_tumors": 184,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = args.output_dir / "probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "stage": "rich_gallery_bas_b21_softplus_mechanics_probe_freeze_v1",
        "probe_summary_sha256": sha256_file(summary_path),
        "training_history_sha256": history_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "activation_manifest_sha256": activation_manifest_sha,
        "validation_predictions_sha256": validation_prediction_sha,
        "candidate_scores_or_choices_frozen": False,
        "full_training_authorized": bool(gate["pass"]),
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "mechanics_probe_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "freeze": freeze}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
