from __future__ import annotations

"""Train BAS from image labels and freeze rich-gallery candidate choices.

Validation segmentation annotations are deliberately absent from this module.
The runner consumes only the canonical train/validation image-label split, the
already frozen G1/upstream score vectors, immutable candidate masks and an
ImageNet initialization.  Spatial evaluation is a separate post-freeze step.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from datasets.factory import build_classification_dataset
from evaluation.classification_metrics import binary_auroc
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.bas_candidate_localizer import (
    BASLossConfig,
    BASResNet50Localizer,
    bas_activation_suppression_loss,
    candidate_activation_evidence,
)
from models.rich_gallery_g2_objective import (
    average_percentile_rank,
    rank_fusion_scores,
    stable_select,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPERIMENT_ID = "EXP-20260801-codex-rich-gallery-bas-b1-v1"
EXPECTED_PRETRAINED_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
EXPECTED_IMAGE_SIZE = 224
EXPECTED_EPOCHS = 100
EXPECTED_BATCH_SIZE = 32
EXPECTED_BACKBONE_LR = 1.0e-3
EXPECTED_WEIGHT_DECAY = 5.0e-4
EXPECTED_MOMENTUM = 0.9
EXPECTED_AREA_WEIGHT = 1.2
EXPECTED_FOREGROUND_WEIGHT = 0.5

VARIANTS = (
    "g1_upstream_baseline",
    "bas_only",
    "g1_bas_two_way",
    "upstream_bas_two_way",
    "g1_upstream_bas_three_way",
)


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


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown rich-gallery source: {value!r}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = {
        "body_weight": [],
        "body_bias": [],
        "head_weight": [],
        "head_bias": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = "classifier_head" in name or "localization_head" in name
        bias = name.endswith(".bias")
        key = ("head" if head else "body") + ("_bias" if bias else "_weight")
        groups[key].append(parameter)
    if not all(groups.values()):
        raise RuntimeError("BAS optimizer parameter partition is incomplete")
    return torch.optim.SGD(
        [
            {"params": groups["body_weight"], "lr": args.backbone_lr},
            {"params": groups["body_bias"], "lr": args.backbone_lr * 2.0},
            {"params": groups["head_weight"], "lr": args.backbone_lr * 10.0},
            {"params": groups["head_bias"], "lr": args.backbone_lr * 20.0},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def _train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    *,
    epochs: int,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "full_ce": 0.0, "foreground_ce": 0.0, "bas": 0.0}
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
                bas = bas_activation_suppression_loss(
                    output,
                    labels,
                    config=BASLossConfig(area_weight=EXPECTED_AREA_WEIGHT),
                )
                loss = full_ce + EXPECTED_FOREGROUND_WEIGHT * foreground_ce + bas
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("BAS training loss is non-finite")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = int(labels.numel())
            totals["loss"] += float(loss.detach()) * batch
            totals["full_ce"] += float(full_ce.detach()) * batch
            totals["foreground_ce"] += float(foreground_ce.detach()) * batch
            totals["bas"] += float(bas.detach()) * batch
            correct += int((output.class_logits.detach().argmax(dim=1) == labels).sum())
            images_seen += batch
        if images_seen != 2981:
            raise RuntimeError("BAS epoch did not cover the canonical train cohort")
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": totals["loss"] / images_seen,
            "full_ce": totals["full_ce"] / images_seen,
            "foreground_ce": totals["foreground_ce"] / images_seen,
            "bas": totals["bas"] / images_seen,
            "accuracy": correct / images_seen,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= 0.5
    positive = labels == 1
    negative = ~positive
    tp = int(np.logical_and(predictions, positive).sum())
    fp = int(np.logical_and(predictions, negative).sum())
    fn = int(np.logical_and(~predictions, positive).sum())
    tn = int(np.logical_and(~predictions, negative).sum())
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2.0 * precision * sensitivity / max(1.0e-12, precision + sensitivity)
    return {
        "auroc": float(binary_auroc(labels, probabilities)),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


@torch.inference_mode()
def _validation_activations(
    model: BASResNet50Localizer,
    loader: DataLoader,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    activations: dict[str, np.ndarray] = {}
    labels: list[int] = []
    probabilities: list[float] = []
    ranges: list[float] = []
    tumor_ranges: list[float] = []
    for images, targets, image_ids in loader:
        images = images.cuda(non_blocking=True)
        logits, maps = model.classify_and_tumor_activation(images)
        flip_logits, flip_maps = model.classify_and_tumor_activation(images.flip(-1))
        logits = 0.5 * (logits + flip_logits)
        maps = 0.5 * (maps + flip_maps.flip(-1))
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        for index, image_id in enumerate(image_ids):
            key = str(image_id)
            value = maps[index, 0].float().cpu().numpy().astype(np.float32)
            if key in activations or not np.isfinite(value).all():
                raise RuntimeError(f"invalid/duplicate BAS activation: {key}")
            activations[key] = value
            label = int(targets[index].reshape(-1)[0].item())
            activation_range = float(value.max() - value.min())
            labels.append(label)
            probabilities.append(float(probs[index].item()))
            ranges.append(activation_range)
            if label:
                tumor_ranges.append(activation_range)
    if len(activations) != 371 or len(tumor_ranges) != 184:
        raise RuntimeError("BAS validation activation cohort mismatch")
    diagnostic = {
        **_binary_metrics(np.asarray(labels), np.asarray(probabilities)),
        "activation_range_mean": float(np.mean(ranges)),
        "tumor_nondegenerate_activation_fraction": float(
            np.mean(np.asarray(tumor_ranges) > 1.0e-4)
        ),
    }
    return activations, diagnostic


def _verify_g1_stage_a(
    root: Path,
    *,
    expected_freeze_sha256: str,
    expected_split_sha256: str,
    expected_val_manifest_sha256: str,
    expected_val_pseudo_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    freeze_path = root / "prediction_freeze.json"
    if sha256_file(freeze_path) != expected_freeze_sha256:
        raise ValueError("G1/G2 Stage-A freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_g2_selector_pair_stage_a_v1"
        or freeze.get("split_sha256") != expected_split_sha256
        or freeze.get("val_candidate_manifest_sha256") != expected_val_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256") != expected_val_pseudo_sha256
        or freeze.get("g1_reproduction_max_selected_index_delta") != 0
        or freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1/G2 Stage-A safety/provenance mismatch")
    manifest_path = root / "stage_a_selection_manifest.csv"
    if sha256_file(manifest_path) != freeze.get("selection_manifest_sha256"):
        raise ValueError("G1/G2 selection manifest changed")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["image_id"]: row
        for row in rows
        if row["variant"] == "g1_frozen__rank_fusion"
    }
    if len(selected) != 371:
        raise ValueError("G1 rank-fusion Stage-A cohort mismatch")
    return freeze, selected


def build_variant_scores(
    g1_logits: np.ndarray,
    upstream_scores: np.ndarray,
    bas_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    g1 = np.asarray(g1_logits, dtype=np.float64)
    upstream = np.asarray(upstream_scores, dtype=np.float64)
    bas = np.asarray(bas_scores, dtype=np.float64)
    if g1.ndim != 1 or g1.shape != upstream.shape or g1.shape != bas.shape:
        raise ValueError("G1/upstream/BAS score vectors must align")
    if not (np.isfinite(g1).all() and np.isfinite(upstream).all() and np.isfinite(bas).all()):
        raise ValueError("candidate score vectors must be finite")
    g1_rank = average_percentile_rank(g1)
    upstream_rank = average_percentile_rank(upstream)
    bas_rank = average_percentile_rank(bas)
    return {
        "g1_upstream_baseline": 0.5 * (g1_rank + upstream_rank),
        "bas_only": bas_rank,
        "g1_bas_two_way": 0.5 * (g1_rank + bas_rank),
        "upstream_bas_two_way": 0.5 * (upstream_rank + bas_rank),
        "g1_upstream_bas_three_way": (g1_rank + upstream_rank + bas_rank) / 3.0,
    }


def _freeze_selections(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
    g1_rows: Mapping[str, Mapping[str, str]],
    candidate_rows: Mapping[str, Mapping[str, str]],
    activations: Mapping[str, np.ndarray],
) -> tuple[str, str, dict[str, float]]:
    score_root = args.output_dir / "stage_a_scores"
    activation_root = args.output_dir / "activation_maps"
    score_root.mkdir(parents=True, exist_ok=False)
    activation_root.mkdir(parents=True, exist_ok=False)
    selection_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    baseline_reproduced = 0
    changed_primary = 0
    correlations: list[float] = []
    for row in val_rows:
        image_id = row["image_id"]
        stem = Path(image_id).stem
        frozen = g1_rows[image_id]
        score_path = args.g1_stage_a_root / frozen["score_path"]
        if sha256_file(score_path) != frozen["score_sha256"]:
            raise ValueError(f"G1 score payload changed: {image_id}")
        candidate_row = candidate_rows[stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"rich candidate payload changed: {image_id}")
        with np.load(score_path, allow_pickle=False) as scored:
            candidate_indices = scored["candidate_indices"].astype(np.int64)
            g1_logits = scored["g1_frozen_candidate_logits"].astype(np.float64)
            upstream = scored["upstream_scores"].astype(np.float64)
        with np.load(candidate_path, allow_pickle=False) as candidate:
            masks = candidate["sam_masks"].astype(np.float32)[candidate_indices]
            sources = candidate["proposal_source_ids"].astype(str)[candidate_indices]
        if not (len(candidate_indices) == len(g1_logits) == len(upstream) == len(masks)):
            raise ValueError(f"rich candidate/G1 arrays misalign: {image_id}")
        activation = np.asarray(activations[image_id], dtype=np.float32)
        tensor_masks = torch.from_numpy(masks)[None]
        valid = torch.ones((1, len(masks)), dtype=torch.bool)
        with torch.inference_mode():
            _coverage, _purity, harmonic = candidate_activation_evidence(
                torch.from_numpy(activation)[None, None],
                tensor_masks,
                valid,
            )
        bas_scores = harmonic[0].numpy().astype(np.float64)
        variants = build_variant_scores(g1_logits, upstream, bas_scores)
        baseline_local = stable_select(variants["g1_upstream_baseline"], g1_logits)
        if int(candidate_indices[baseline_local]) != int(frozen["selected_candidate_index"]):
            raise ValueError(f"G1/upstream baseline does not reproduce: {image_id}")
        baseline_reproduced += 1
        primary_local = stable_select(variants["g1_upstream_bas_three_way"], g1_logits)
        changed_primary += int(primary_local != baseline_local)
        bas_rank = variants["bas_only"]
        upstream_rank = average_percentile_rank(upstream)
        if np.std(bas_rank) > 0 and np.std(upstream_rank) > 0:
            correlations.append(float(np.corrcoef(bas_rank, upstream_rank)[0, 1]))
        activation_path = activation_root / f"{stem}.npy"
        np.save(activation_path, activation.astype(np.float32), allow_pickle=False)
        activation_rows.append(
            {
                "image_id": image_id,
                "group_id": row["group_id"],
                "tumor": int(row["tumor"]),
                "activation_path": str(activation_path.relative_to(args.output_dir)).replace("\\", "/"),
                "activation_sha256": sha256_file(activation_path),
                "activation_min": float(activation.min()),
                "activation_max": float(activation.max()),
            }
        )
        frozen_score_path = score_root / f"{stem}.npz"
        np.savez_compressed(
            frozen_score_path,
            candidate_indices=candidate_indices.astype(np.int32),
            g1_logits=g1_logits.astype(np.float32),
            upstream_scores=upstream.astype(np.float32),
            bas_scores=bas_scores.astype(np.float32),
            **{name: values.astype(np.float32) for name, values in variants.items()},
        )
        frozen_score_sha = sha256_file(frozen_score_path)
        for variant in VARIANTS:
            local = stable_select(variants[variant], g1_logits)
            selection_rows.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "tumor": int(row["tumor"]),
                    "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                    "candidate_count": len(candidate_indices),
                    "selected_local_index": local,
                    "selected_candidate_index": int(candidate_indices[local]),
                    "selected_source": canonical_source(sources[local]),
                    "selected_g1_logit": float(g1_logits[local]),
                    "selected_upstream_score": float(upstream[local]),
                    "selected_bas_score": float(bas_scores[local]),
                    "selected_variant_score": float(variants[variant][local]),
                    "score_path": str(frozen_score_path.relative_to(args.output_dir)).replace("\\", "/"),
                    "score_sha256": frozen_score_sha,
                }
            )
    if baseline_reproduced != 371 or len(selection_rows) != 371 * len(VARIANTS):
        raise RuntimeError("rich BAS Stage-A cohort/reproduction mismatch")
    selection_sha = _write_csv(args.output_dir / "selection_manifest.csv", selection_rows)
    activation_sha = _write_csv(args.output_dir / "activation_manifest.csv", activation_rows)
    diagnostics = {
        "g1_upstream_baseline_reproduced": float(baseline_reproduced),
        "three_way_changed_selection_fraction": changed_primary / 371.0,
        "mean_bas_upstream_rank_correlation": float(np.mean(correlations)),
        "correlation_images": float(len(correlations)),
    }
    return selection_sha, activation_sha, diagnostics


def _active_state_dict(model: BASResNet50Localizer) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name.startswith("background_"):
            continue
        detached = value.detach().cpu()
        result[name] = detached.half() if detached.is_floating_point() else detached
    return result


def main() -> None:
    args = parse_args()
    if (
        args.expected_pretrained_sha256 != EXPECTED_PRETRAINED_SHA256
        or args.epochs != EXPECTED_EPOCHS
        or args.batch_size != EXPECTED_BATCH_SIZE
        or args.backbone_lr != EXPECTED_BACKBONE_LR
        or args.weight_decay != EXPECTED_WEIGHT_DECAY
        or args.momentum != EXPECTED_MOMENTUM
        or args.seed != 42
    ):
        raise ValueError("rich-gallery BAS execution differs from frozen controls")
    if args.output_dir.exists():
        raise FileExistsError("rich-gallery BAS output directory must not exist")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")
    if sha256_file(args.pretrained_checkpoint) != args.expected_pretrained_sha256:
        raise ValueError("ImageNet ResNet-50 SHA-256 mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("rich-gallery BAS requires exactly two CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"rich-gallery BAS requires T4 x2, got {device_names}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("canonical train/validation cohort mismatch")
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
        raise ValueError("rich-gallery BAS requires all validation candidate bags")
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
    model = BASResNet50Localizer(pretrained=False, backbone_state_dict=state).cuda()
    parallel = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = _optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history = _train(parallel, train_loader, optimizer, scaler, epochs=args.epochs)
    history_sha = _write_csv(args.output_dir / "training_history.csv", history)
    activations, classification_diagnostic = _validation_activations(model, val_loader)
    checkpoint_path = args.output_dir / "bas_localizer_final_fp16.pt"
    torch.save(
        {
            "model_state_dict": _active_state_dict(model),
            "experiment_id": EXPERIMENT_ID,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "pretrained_sha256": args.expected_pretrained_sha256,
            "epochs": args.epochs,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    selection_sha, activation_sha, selector_diagnostic = _freeze_selections(
        args,
        val_rows,
        g1_rows,
        candidate_rows,
        activations,
    )
    diagnostics = {
        **classification_diagnostic,
        **selector_diagnostic,
        "diagnostics_do_not_block_spatial_evaluation": True,
    }
    diagnostic_path = args.output_dir / "label_safe_diagnostics.json"
    diagnostic_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "stage": "rich_gallery_bas_b1_stage_a_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "g1_stage_a_freeze_sha256": args.expected_g1_stage_a_freeze_sha256,
        "g1_checkpoint_sha256": g1_freeze["g1_checkpoint_sha256"],
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "bas_checkpoint_sha256": checkpoint_sha,
        "training_history_sha256": history_sha,
        "selection_manifest_sha256": selection_sha,
        "activation_manifest_sha256": activation_sha,
        "label_safe_diagnostics_sha256": sha256_file(diagnostic_path),
        "validation_images": 371,
        "selection_rows": 371 * len(VARIANTS),
        "variants": list(VARIANTS),
        "baseline_reproduction_images": int(
            selector_diagnostic["g1_upstream_baseline_reproduced"]
        ),
        "candidate_choices_frozen_before_validation_gt": True,
        "training_labels": "image_level_normal_tumor_only",
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"cuda_device_count": 2, "cuda_device_names": device_names},
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
