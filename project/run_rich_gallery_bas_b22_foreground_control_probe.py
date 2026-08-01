from __future__ import annotations

"""Matched label-safe probe for the B2.1 zero-map objective correction."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from analyze_rich_gallery_bas_b21_softplus_failure import activation_collapse_features
from datasets.factory import build_classification_dataset
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.bas_candidate_localizer import (
    BASResNet50Localizer,
    ForegroundControlLossConfig,
    foreground_control_area_loss,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_bas_candidate_descriptor_b1 import (
    EXPECTED_BACKBONE_LR,
    EXPECTED_BATCH_SIZE,
    EXPECTED_FOREGROUND_WEIGHT,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_MOMENTUM,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_WEIGHT_DECAY,
    _active_state_dict,
    _optimizer,
    _verify_g1_stage_a,
)
from run_rich_gallery_bas_b21_softplus_probe import (
    _freeze_label_safe_diagnostics,
    _validation_probe,
)


EXPERIMENT_ID = "EXP-20260802-codex-rich-gallery-bas-b22-foreground-control-probe-v1"
EXPECTED_EPOCHS = 5
EXPECTED_ACTIVATION = "softplus"
EXPECTED_FGC_WEIGHT = 1.5
EXPECTED_AREA_WEIGHT = 1.2
EXPECTED_REFERENCE_RATIO = 0.5
TRAIN_IMAGES = 2981
TRAIN_NORMALS = 1493

GATES = {
    "final_full_ce_max": 0.69,
    "final_foreground_ce_max": 0.68,
    "final_accuracy_prior_distance_min": 0.01,
    "validation_auroc_min": 0.55,
    "activation_range_mean_min": 1.0e-3,
    "tumor_nondegenerate_fraction_min": 0.50,
    "tumor_argmax_border_fraction_max": 0.50,
    "tumor_top_1_percent_mass_median_max": 0.75,
    "tumor_effective_support_median_min": 0.003,
    "tumor_candidate_area_spearman_mean_max": 0.98,
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
        raise ValueError("cannot write empty B2.2 table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    *,
    epochs: int,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    config = ForegroundControlLossConfig(
        foreground_control_weight=EXPECTED_FGC_WEIGHT,
        area_weight=EXPECTED_AREA_WEIGHT,
        reference_ratio=EXPECTED_REFERENCE_RATIO,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "full_ce": 0.0, "foreground_ce": 0.0, "fgc_area": 0.0}
        correct = 0
        images_seen = 0
        for images, targets, _image_ids in loader:
            images = images.cuda(non_blocking=True)
            labels = targets.reshape(-1).long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                output = model(images, labels)
                full_ce = F.cross_entropy(output.class_logits, labels)
                foreground_ce = F.cross_entropy(output.foreground_logits, labels)
                fgc_area = foreground_control_area_loss(output, labels, config=config)
                loss = full_ce + EXPECTED_FOREGROUND_WEIGHT * foreground_ce + fgc_area
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("B2.2 training loss is non-finite")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = int(labels.numel())
            totals["loss"] += float(loss.detach()) * batch
            totals["full_ce"] += float(full_ce.detach()) * batch
            totals["foreground_ce"] += float(foreground_ce.detach()) * batch
            totals["fgc_area"] += float(fgc_area.detach()) * batch
            correct += int((output.class_logits.detach().argmax(dim=1) == labels).sum())
            images_seen += batch
        if images_seen != TRAIN_IMAGES:
            raise RuntimeError("B2.2 epoch did not cover canonical training cohort")
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": totals["loss"] / images_seen,
            "full_ce": totals["full_ce"] / images_seen,
            "foreground_ce": totals["foreground_ce"] / images_seen,
            "fgc_area": totals["fgc_area"] / images_seen,
            "accuracy": correct / images_seen,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def _spatial_mechanics(
    activations: Mapping[str, np.ndarray],
    validation_rows: list[Mapping[str, object]],
) -> dict[str, float]:
    tumor = []
    for row in validation_rows:
        if int(row["tumor"]) == 1:
            tumor.append(activation_collapse_features(activations[str(row["image_id"])]))
    if len(tumor) != 184:
        raise RuntimeError("B2.2 tumor mechanics cohort mismatch")
    return {
        "tumor_argmax_border_fraction": float(np.mean([x["argmax_border"] for x in tumor])),
        "tumor_top_1_percent_mass_median": float(
            np.median([x["top_1_percent_mass_fraction"] for x in tumor])
        ),
        "tumor_effective_support_median": float(
            np.median([x["effective_support_fraction"] for x in tumor])
        ),
        "tumor_activation_max_mean": float(np.mean([x["activation_max"] for x in tumor])),
        "tumor_sigmoid_gradient_max_mean": float(
            np.mean([x["sigmoid_gradient_max"] for x in tumor])
        ),
    }


def _mechanics_gate(
    history: list[Mapping[str, float | int]],
    validation: Mapping[str, float],
    spatial: Mapping[str, float],
    area: Mapping[str, float],
) -> dict[str, object]:
    final = history[-1]
    values = {
        "final_full_ce": float(final["full_ce"]),
        "final_foreground_ce": float(final["foreground_ce"]),
        "final_accuracy_prior_distance": abs(float(final["accuracy"]) - TRAIN_NORMALS / TRAIN_IMAGES),
        "validation_auroc": float(validation["auroc"]),
        "activation_range_mean": float(validation["activation_range_mean"]),
        "tumor_nondegenerate_fraction": float(validation["tumor_nondegenerate_activation_fraction"]),
        "tumor_argmax_border_fraction": float(spatial["tumor_argmax_border_fraction"]),
        "tumor_top_1_percent_mass_median": float(spatial["tumor_top_1_percent_mass_median"]),
        "tumor_effective_support_median": float(spatial["tumor_effective_support_median"]),
        "tumor_candidate_area_spearman_mean": float(area["tumor_bas_area_spearman_mean"]),
    }
    checks = {
        "final_full_ce": values["final_full_ce"] <= GATES["final_full_ce_max"],
        "final_foreground_ce": values["final_foreground_ce"] <= GATES["final_foreground_ce_max"],
        "final_accuracy_prior_distance": values["final_accuracy_prior_distance"] >= GATES["final_accuracy_prior_distance_min"],
        "validation_auroc": values["validation_auroc"] >= GATES["validation_auroc_min"],
        "activation_range_mean": values["activation_range_mean"] >= GATES["activation_range_mean_min"],
        "tumor_nondegenerate_fraction": values["tumor_nondegenerate_fraction"] >= GATES["tumor_nondegenerate_fraction_min"],
        "tumor_argmax_border_fraction": values["tumor_argmax_border_fraction"] <= GATES["tumor_argmax_border_fraction_max"],
        "tumor_top_1_percent_mass_median": values["tumor_top_1_percent_mass_median"] <= GATES["tumor_top_1_percent_mass_median_max"],
        "tumor_effective_support_median": values["tumor_effective_support_median"] >= GATES["tumor_effective_support_median_min"],
        "tumor_candidate_area_spearman_mean": values["tumor_candidate_area_spearman_mean"] <= GATES["tumor_candidate_area_spearman_mean_max"],
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "values": values,
        "thresholds": GATES,
        "meaning": "label_safe_mechanics_gate_not_spatial_efficacy",
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
        raise ValueError("B2.2 differs from frozen matched controls")
    if args.output_dir.exists():
        raise FileExistsError("B2.2 output directory must not exist")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")
    if sha256_file(args.pretrained_checkpoint) != args.expected_pretrained_sha256:
        raise ValueError("ImageNet ResNet-50 SHA-256 mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("B2.2 requires exactly two CUDA devices")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    _seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    train_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="train"
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if len(train_rows) != TRAIN_IMAGES or len(val_rows) != 371:
        raise RuntimeError("canonical B2.2 cohort mismatch")
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
        raise ValueError("B2.2 requires all validation candidate bags")

    train_dataset = build_classification_dataset(
        root=args.dataset_root, split="train", target_columns=("tumor",),
        image_size=EXPECTED_IMAGE_SIZE, augment=True, normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    val_dataset = build_classification_dataset(
        root=args.dataset_root, split="val", target_columns=("tumor",),
        image_size=EXPECTED_IMAGE_SIZE, augment=False, normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    state = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
    model = BASResNet50Localizer(
        pretrained=False, backbone_state_dict=state,
        classifier_activation=EXPECTED_ACTIVATION,
    ).cuda()
    parallel = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = _optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history = _train(parallel, train_loader, optimizer, scaler, epochs=args.epochs)
    history_sha = _write_csv(args.output_dir / "training_history.csv", history)
    activations, validation_diagnostics, validation_rows = _validation_probe(model, val_loader)
    prediction_sha = _write_csv(args.output_dir / "validation_predictions.csv", validation_rows)
    spatial_diagnostics = _spatial_mechanics(activations, validation_rows)
    activation_manifest_sha, area_diagnostics = _freeze_label_safe_diagnostics(
        output_dir=args.output_dir, val_rows=val_rows, activations=activations,
        g1_rows=g1_rows, g1_stage_a_root=args.g1_stage_a_root,
        candidate_rows=candidate_rows, candidate_root=args.val_candidate_root,
    )
    gate = _mechanics_gate(history, validation_diagnostics, spatial_diagnostics, area_diagnostics)
    checkpoint_path = args.output_dir / "bas_b22_foreground_control_probe_fp16.pt"
    torch.save(
        {
            "model_state_dict": _active_state_dict(model),
            "experiment_id": EXPERIMENT_ID,
            "classifier_activation": EXPECTED_ACTIVATION,
            "objective": "continuous_foreground_control_ratio",
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "epochs": args.epochs,
            "mechanics_gate_pass": gate["pass"],
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    summary = {
        "stage": "rich_gallery_bas_b22_foreground_control_mechanics_probe_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "classifier_activation": EXPECTED_ACTIVATION,
        "scientific_delta_from_b21": "hard_gated_background_ratio_to_continuous_foreground_control",
        "objective_weights": {"full_ce": 1.0, "foreground_ce": 0.5, "foreground_control": 1.5, "area": 1.2, "reference_ratio": 0.5},
        "epochs": args.epochs,
        "training_history_sha256": history_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "activation_manifest_sha256": activation_manifest_sha,
        "validation_predictions_sha256": prediction_sha,
        "validation_diagnostics": validation_diagnostics,
        "spatial_mechanics_diagnostics": spatial_diagnostics,
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "stage": "rich_gallery_bas_b22_foreground_control_mechanics_probe_freeze_v1",
        "probe_summary_sha256": sha256_file(summary_path),
        "training_history_sha256": history_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "activation_manifest_sha256": activation_manifest_sha,
        "validation_predictions_sha256": prediction_sha,
        "candidate_scores_or_choices_frozen": False,
        "full_training_authorized": bool(gate["pass"]),
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "mechanics_probe_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "freeze": freeze}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
