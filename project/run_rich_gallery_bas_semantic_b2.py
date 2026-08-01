"""Train and freeze the rich-gallery B2 BAS semantic descriptor pair."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from audit_rich_gallery_stage_a_transport import (
    EXPECTED_BASELINE_VARIANT,
    audit as audit_transport,
    audit_g1_baseline_row,
    load_npz_mapping,
    safe_transport_path,
)
from datasets.factory import build_classification_dataset
from mae_reconstruction_io import load_split_rows_without_annotations
from models.bas_candidate_localizer import (
    BASLossConfig,
    BASResNet50Localizer,
    bas_activation_suppression_loss,
)
from models.rich_gallery_bas_residual import (
    average_percentile_rank,
    bas_candidate_scores,
    canonical_source,
    score_rich_gallery_bas_pair,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from pseudo.manifest import sha256_file


EXPERIMENT_ID = "EXP-20260801-codex-b2-rich-gallery-bas-residual-v1"
CONTROL_ARM = "g1_upstream_control"
SEMANTIC_ARM = "g1_upstream_bas_semantic"
EXPECTED_PRETRAINED_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
EXPECTED_IMAGE_SIZE = 448
EXPECTED_EPOCHS = 100
EXPECTED_BATCH_SIZE = 32
EXPECTED_BACKBONE_LR = 1.0e-3
EXPECTED_WEIGHT_DECAY = 5.0e-4
EXPECTED_MOMENTUM = 0.9
EXPECTED_AREA_WEIGHT = 1.2
EXPECTED_FOREGROUND_WEIGHT = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--input-size", type=int, default=EXPECTED_IMAGE_SIZE)
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
        raise RuntimeError("B2 optimizer parameter partition is incomplete")
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


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, EXPECTED_EPOCHS + 1):
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
                raise RuntimeError("B2 training loss is non-finite")
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
            raise RuntimeError("B2 training epoch did not cover the frozen cohort")
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


def _active_state_dict(model: BASResNet50Localizer) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name.startswith("background_"):
            continue
        detached = value.detach().cpu()
        result[name] = detached.half() if detached.is_floating_point() else detached
    return result


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    positive = labels == 1
    negative = ~positive
    tp = int(np.logical_and(predictions, positive).sum())
    fp = int(np.logical_and(predictions, negative).sum())
    fn = int(np.logical_and(~predictions, positive).sum())
    tn = int(np.logical_and(~predictions, negative).sum())
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "f1": float(2.0 * precision * sensitivity / max(1.0e-12, precision + sensitivity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


@torch.inference_mode()
def _validation_activations(
    model: BASResNet50Localizer,
    loader: DataLoader,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    activations: dict[str, np.ndarray] = {}
    labels: list[int] = []
    probabilities: list[float] = []
    tumor_ranges: list[float] = []
    for images, targets, image_ids in loader:
        images = images.cuda(non_blocking=True)
        logits, maps = model.classify_and_tumor_activation(images)
        flipped_logits, flipped_maps = model.classify_and_tumor_activation(
            torch.flip(images, dims=(-1,))
        )
        logits = 0.5 * (logits + flipped_logits)
        maps = 0.5 * (maps + torch.flip(flipped_maps, dims=(-1,)))
        batch_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        for index, image_id in enumerate(image_ids):
            value = maps[index, 0].float().cpu().numpy()
            if not np.isfinite(value).all():
                raise RuntimeError(f"B2 activation is non-finite: {image_id}")
            activations[str(image_id)] = value
            label = int(targets[index].reshape(-1)[0].item())
            labels.append(label)
            probabilities.append(float(batch_probabilities[index].cpu()))
            if label == 1:
                tumor_ranges.append(float(np.ptp(value)))
    if len(activations) != 371 or len(labels) != 371 or len(tumor_ranges) != 184:
        raise RuntimeError("B2 validation activation cohort mismatch")
    metrics = _binary_metrics(
        np.asarray(labels, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
    )
    nondegenerate = sum(value > 1.0e-4 for value in tumor_ranges)
    gate = {
        **metrics,
        "validation_images": 371,
        "validation_tumor_images": 184,
        "finite_activation_maps": 371,
        "tumor_nondegenerate_activation_maps": nondegenerate,
        "tumor_nondegenerate_fraction": nondegenerate / 184.0,
        "minimum_auroc": 0.75,
        "minimum_sensitivity": 0.60,
        "minimum_specificity": 0.60,
        "minimum_tumor_nondegenerate_fraction": 0.95,
    }
    gate["classification_gate_pass"] = bool(
        metrics["auroc"] >= 0.75
        and metrics["sensitivity"] >= 0.60
        and metrics["specificity"] >= 0.60
        and nondegenerate / 184.0 >= 0.95
    )
    return activations, gate


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(left)
    right_rank = average_percentile_rank(right)
    if float(np.std(left_rank)) <= 1.0e-12 or float(np.std(right_rank)) <= 1.0e-12:
        # A constant BAS rank has no new ordering information. Treat it as
        # maximally redundant so the GT-blind complementarity gate rejects the
        # arm cleanly instead of turning an expected scientific failure into a
        # runtime error.
        return 1.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("candidate rank correlation is non-finite")
    return value


def pack_binary_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or not array.size:
        raise ValueError("prediction mask must be one nonempty 2D array")
    return (
        np.packbits(array.reshape(-1), bitorder="little"),
        np.asarray(array.shape, dtype=np.int32),
    )


def unpack_binary_mask(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
    dimensions = tuple(int(value) for value in np.asarray(shape).reshape(-1))
    if len(dimensions) != 2 or min(dimensions) <= 0:
        raise ValueError("packed prediction shape is invalid")
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")
    required = int(np.prod(dimensions))
    if len(bits) < required:
        raise ValueError("packed prediction is truncated")
    return bits[:required].reshape(dimensions).astype(bool)


@dataclass(frozen=True)
class ScoredImage:
    image_id: str
    group_id: str
    tumor: int
    candidate_payload_sha256: str
    candidate_count: int
    candidate_indices: np.ndarray
    source_ids: np.ndarray
    g1_logits: np.ndarray
    upstream_scores: np.ndarray
    coverage: np.ndarray
    purity: np.ndarray
    bas_scores: np.ndarray
    baseline_rank: np.ndarray
    semantic_rank: np.ndarray
    baseline_local_index: int
    semantic_local_index: int
    baseline_mask: np.ndarray
    semantic_mask: np.ndarray
    baseline_source: str
    semantic_source: str


def score_one_image(
    activation: np.ndarray,
    baseline_row: Mapping[str, str],
    candidate_payload: Mapping[str, object],
    stage_a_payload: Mapping[str, object],
) -> ScoredImage:
    aligned = audit_g1_baseline_row(baseline_row, candidate_payload, stage_a_payload)
    coverage, purity, harmonic = bas_candidate_scores(
        activation,
        aligned.candidate_masks,
    )
    pair = score_rich_gallery_bas_pair(
        aligned.g1_logits,
        aligned.upstream_scores,
        harmonic,
    )
    source_names = np.asarray(candidate_payload["proposal_source_ids"]).reshape(-1)
    baseline_original = int(aligned.candidate_indices[pair.baseline_local_index])
    semantic_original = int(aligned.candidate_indices[pair.bas_residual_local_index])
    return ScoredImage(
        image_id=str(baseline_row["image_id"]),
        group_id=str(baseline_row["group_id"]),
        tumor=int(baseline_row["tumor"]),
        candidate_payload_sha256=str(baseline_row["candidate_payload_sha256"]),
        candidate_count=len(aligned.candidate_indices),
        candidate_indices=aligned.candidate_indices,
        source_ids=aligned.source_ids,
        g1_logits=aligned.g1_logits,
        upstream_scores=aligned.upstream_scores,
        coverage=coverage,
        purity=purity,
        bas_scores=harmonic,
        baseline_rank=pair.baseline_rank,
        semantic_rank=pair.bas_residual_rank,
        baseline_local_index=pair.baseline_local_index,
        semantic_local_index=pair.bas_residual_local_index,
        baseline_mask=aligned.candidate_masks[pair.baseline_local_index],
        semantic_mask=aligned.candidate_masks[pair.bas_residual_local_index],
        baseline_source=canonical_source(source_names[baseline_original]),
        semantic_source=canonical_source(source_names[semantic_original]),
    )


def _write_evidence(
    output_dir: Path,
    scored: list[ScoredImage],
    activations: Mapping[str, np.ndarray],
) -> tuple[str, str]:
    score_root = output_dir / "candidate_score_evidence"
    activation_root = output_dir / "activation_evidence"
    score_root.mkdir(parents=True, exist_ok=False)
    activation_root.mkdir(parents=True, exist_ok=False)
    score_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    for item in scored:
        stem = Path(item.image_id).stem
        score_path = score_root / f"{stem}.npz"
        np.savez_compressed(
            score_path,
            candidate_indices=item.candidate_indices.astype(np.int32),
            source_ids=item.source_ids.astype(np.int16),
            g1_logits=item.g1_logits.astype(np.float32),
            upstream_scores=item.upstream_scores.astype(np.float32),
            coverage=item.coverage.astype(np.float32),
            purity=item.purity.astype(np.float32),
            bas_scores=item.bas_scores.astype(np.float32),
            baseline_rank=item.baseline_rank.astype(np.float32),
            semantic_rank=item.semantic_rank.astype(np.float32),
        )
        activation_path = activation_root / f"{stem}.npy"
        np.save(
            activation_path,
            np.asarray(activations[item.image_id], dtype=np.float32),
            allow_pickle=False,
        )
        score_rows.append(
            {
                "image_id": item.image_id,
                "candidate_payload_sha256": item.candidate_payload_sha256,
                "candidate_count": item.candidate_count,
                "score_path": score_path.relative_to(output_dir).as_posix(),
                "score_sha256": sha256_file(score_path),
            }
        )
        activation_rows.append(
            {
                "image_id": item.image_id,
                "activation_path": activation_path.relative_to(output_dir).as_posix(),
                "activation_sha256": sha256_file(activation_path),
                "activation_shape": "x".join(map(str, activations[item.image_id].shape)),
            }
        )
    return (
        _write_csv(output_dir / "candidate_score_manifest.csv", score_rows),
        _write_csv(output_dir / "activation_manifest.csv", activation_rows),
    )


def _materialize_pair(
    output_dir: Path,
    scored: list[ScoredImage],
) -> dict[str, str]:
    arm_hashes: dict[str, str] = {}
    for arm, local_attr, mask_attr, source_attr in (
        (CONTROL_ARM, "baseline_local_index", "baseline_mask", "baseline_source"),
        (SEMANTIC_ARM, "semantic_local_index", "semantic_mask", "semantic_source"),
    ):
        arm_root = output_dir / arm
        mask_root = arm_root / "predictions"
        mask_root.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, object]] = []
        for item in scored:
            local_index = int(getattr(item, local_attr))
            mask = np.asarray(getattr(item, mask_attr), dtype=bool)
            packed, shape = pack_binary_mask(mask)
            path = mask_root / f"{Path(item.image_id).stem}.npz"
            np.savez_compressed(path, packed_mask=packed, shape=shape)
            rows.append(
                {
                    "image_id": item.image_id,
                    "group_id": item.group_id,
                    "tumor": item.tumor,
                    "candidate_payload_sha256": item.candidate_payload_sha256,
                    "candidate_count": item.candidate_count,
                    "selected_local_index": local_index,
                    "selected_candidate_index": int(item.candidate_indices[local_index]),
                    "selected_source": str(getattr(item, source_attr)),
                    "prediction_path": path.relative_to(arm_root).as_posix(),
                    "prediction_sha256": sha256_file(path),
                }
            )
        manifest_path = arm_root / "prediction_manifest.csv"
        manifest_sha256 = _write_csv(manifest_path, rows)
        freeze = {
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "prediction_manifest_sha256": manifest_sha256,
            "validation_predictions": 371,
            "candidate_choices_frozen_before_validation_gt": True,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_images_read": 0,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        freeze_path.write_text(
            json.dumps(freeze, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        arm_hashes[arm] = sha256_file(freeze_path)
    return arm_hashes


def _load_baseline_rows(transport_root: Path) -> list[dict[str, str]]:
    rows = _load_csv(transport_root / "stage_a_selection_manifest.csv")
    baseline = [row for row in rows if row.get("variant") == EXPECTED_BASELINE_VARIANT]
    if len(baseline) != 371 or len({row["image_id"] for row in baseline}) != 371:
        raise ValueError("B2 frozen G1/upstream baseline cohort mismatch")
    return baseline


def _score_all_images(
    args: argparse.Namespace,
    activations: Mapping[str, np.ndarray],
    candidate_rows: Mapping[str, Mapping[str, str]],
) -> tuple[list[ScoredImage], dict[str, float | int]]:
    scored: list[ScoredImage] = []
    g1_correlations: list[float] = []
    upstream_correlations: list[float] = []
    positive_changed = 0
    positive_images = 0
    for row in _load_baseline_rows(args.transport_root):
        image_id = row["image_id"]
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        score_path = safe_transport_path(args.transport_root, row["score_path"])
        item = score_one_image(
            activations[image_id],
            row,
            load_npz_mapping(candidate_path),
            load_npz_mapping(score_path),
        )
        scored.append(item)
        if item.tumor == 1:
            positive_images += 1
            g1_correlations.append(_rank_correlation(item.bas_scores, item.g1_logits))
            upstream_correlations.append(
                _rank_correlation(item.bas_scores, item.upstream_scores)
            )
            positive_changed += int(
                item.baseline_local_index != item.semantic_local_index
            )
    if len(scored) != 371 or positive_images != 184:
        raise RuntimeError("B2 scored cohort mismatch")
    if len(g1_correlations) != 184 or len(upstream_correlations) != 184:
        raise RuntimeError("B2 rank-correlation cohort mismatch")
    diagnostics: dict[str, float | int] = {
        "mean_bas_g1_rank_correlation_positive_bags": float(np.mean(g1_correlations)),
        "mean_bas_upstream_rank_correlation_positive_bags": float(
            np.mean(upstream_correlations)
        ),
        "correlation_positive_bags": 184,
        "semantic_changed_positive_selections": positive_changed,
        "semantic_changed_positive_selection_fraction": positive_changed / 184.0,
    }
    return scored, diagnostics


def main() -> None:
    args = parse_args()
    if (
        args.epochs != EXPECTED_EPOCHS
        or args.batch_size != EXPECTED_BATCH_SIZE
        or args.input_size != EXPECTED_IMAGE_SIZE
        or args.backbone_lr != EXPECTED_BACKBONE_LR
        or args.weight_decay != EXPECTED_WEIGHT_DECAY
        or args.momentum != EXPECTED_MOMENTUM
        or args.seed != 42
        or args.expected_pretrained_sha256 != EXPECTED_PRETRAINED_SHA256
    ):
        raise ValueError("B2 execution differs from the frozen static recipe")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("B2 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"B2 requires Tesla T4 x2, got {device_names}")
    if sha256_file(args.pretrained_checkpoint) != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("B2 ImageNet initialization SHA-256 mismatch")
    if args.output_dir.exists():
        raise FileExistsError("B2 output directory already exists")
    args.output_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    _seed_everything(args.seed)

    transport_audit = audit_transport(args)
    transport_audit_path = args.output_dir / "transport_audit.json"
    transport_audit_path.write_text(
        json.dumps(transport_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_rows = {
        split: load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("B2 frozen cohort mismatch")
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in split_rows["val"]],
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
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
    ).cuda()
    parallel = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = _optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history = _train(parallel, train_loader, optimizer, scaler)
    history_sha256 = _write_csv(args.output_dir / "training_history.csv", history)
    activations, gate = _validation_activations(model, val_loader)

    checkpoint_path = args.output_dir / "bas_localizer_final_fp16.pt"
    torch.save(
        {
            "model_state_dict": _active_state_dict(model),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
            "input_size": EXPECTED_IMAGE_SIZE,
            "epochs": EXPECTED_EPOCHS,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    gate.update(
        {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_history_sha256": history_sha256,
            "transport_audit_sha256": sha256_file(transport_audit_path),
            "validation_candidate_manifest_sha256": candidate_audit["manifest_sha256"],
            "validation_candidate_summary_sha256": candidate_audit["summary_sha256"],
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_images_read": 0,
            "test_evaluated": False,
        }
    )
    gate_path = args.output_dir / "operational_gate.json"
    scored, complementarity = _score_all_images(args, activations, candidate_rows)
    score_manifest_sha256, activation_manifest_sha256 = _write_evidence(
        args.output_dir,
        scored,
        activations,
    )
    gate.update(
        {
            **complementarity,
            "maximum_mean_bas_g1_rank_correlation": 0.80,
            "maximum_mean_bas_upstream_rank_correlation": 0.80,
            "minimum_changed_positive_selection_fraction": 0.05,
            "candidate_score_manifest_sha256": score_manifest_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
        }
    )
    gate["complementarity_gate_pass"] = bool(
        float(complementarity["mean_bas_g1_rank_correlation_positive_bags"]) <= 0.80
        and float(complementarity["mean_bas_upstream_rank_correlation_positive_bags"])
        <= 0.80
        and float(complementarity["semantic_changed_positive_selection_fraction"])
        >= 0.05
    )
    gate["descriptor_diagnostic_gate_pass"] = bool(
        gate["classification_gate_pass"] and gate["complementarity_gate_pass"]
    )
    # These label-safe diagnostics explain a failure mode but cannot replace
    # the frozen spatial endpoint. Reaching this line already proves the
    # transport/runtime/cohort/finite-output operational contract, so both
    # physical arms are frozen regardless of the diagnostic point values.
    gate["operational_gate_pass"] = True
    gate["spatial_evaluation_authorized_after_pair_freeze"] = True
    gate["descriptor_diagnostics_do_not_block_spatial_evaluation"] = True
    gate["consumer_authorized"] = False
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arm_freezes = _materialize_pair(args.output_dir, scored)
    pair_freeze = {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "transport_prediction_freeze_sha256": args.expected_freeze_sha256,
        "transport_audit_sha256": sha256_file(transport_audit_path),
        "g1_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "bas_checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history_sha256": history_sha256,
        "operational_gate_sha256": sha256_file(gate_path),
        "candidate_score_manifest_sha256": score_manifest_sha256,
        "activation_manifest_sha256": activation_manifest_sha256,
        "arm_freezes": arm_freezes,
        "pair_physically_frozen_before_validation_gt": True,
        "training_labels": "image_level_normal_tumor_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    pair_path.write_text(
        json.dumps(pair_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "run_id": "btxrd_rich_gallery_bas_semantic_b2_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "cuda_device_count": 2,
            "cuda_device_names": device_names,
            "training_parallelism": "torch.nn.DataParallel",
        },
        "cohort": {"train": 2981, "validation": 371},
        "pair_freeze_sha256": sha256_file(pair_path),
        "arm_freezes": arm_freezes,
        "operational_gate": gate,
        "training_labels": "image_level_normal_tumor_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
