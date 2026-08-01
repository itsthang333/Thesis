from __future__ import annotations

"""Train and freeze the B1 BAS descriptor using image-level labels only."""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from datasets.factory import build_classification_dataset
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.bas_candidate_localizer import (
    BASLossConfig,
    BASResNet50Localizer,
    bas_activation_suppression_loss,
    candidate_activation_evidence,
    equal_rank_aggregate,
    within_bag_percentile_ranks,
)
from models.mask_bag_same_family_graph import (
    SameFamilyGraphConfig,
    score_same_family_graph_records,
)
from models.mask_bag_selector_cache import unpack_candidate_masks
from run_mask_bag_normal_prototype_arm import (
    _load_baseline_model,
    _load_cache_records,
    _verify_cache_freeze,
    _write_validation_outputs,
)
from run_mask_bag_same_family_graph_s3_arm import _verify_baseline_freeze
from run_rad_dino_mask_bag_mil_probe import _audit_candidate_input


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
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
        groups[("head" if head else "body") + ("_bias" if bias else "_weight")].append(parameter)
    if not all(groups.values()):
        raise RuntimeError("B1 optimizer parameter partition is incomplete")
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
                raise RuntimeError("B1 training loss is non-finite")
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
            raise RuntimeError("B1 training epoch did not cover the frozen cohort")
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
    ranges: list[float] = []
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
            local_range = float(np.ptp(value))
            activations[str(image_id)] = value
            label = int(targets[index].reshape(-1)[0].item())
            labels.append(label)
            probabilities.append(float(batch_probabilities[index].cpu()))
            ranges.append(local_range)
            if label == 1:
                tumor_ranges.append(local_range)
    if len(activations) != 371 or len(labels) != 371 or len(tumor_ranges) != 184:
        raise RuntimeError("B1 validation activation cohort mismatch")
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
    gate["operational_gate_pass"] = bool(
        metrics["auroc"] >= 0.75
        and metrics["sensitivity"] >= 0.60
        and metrics["specificity"] >= 0.60
        and nondegenerate / 184.0 >= 0.95
    )
    return activations, gate


def _npy_float16_sha256(values: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.asarray(values, dtype=np.float16), allow_pickle=False)
    return sha256(stream.getvalue()).hexdigest()


def _baseline_identity(
    records: list[dict[str, Any]],
    base_scored: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    accepted = {row["image_id"]: row for row in baseline_rows}
    rows: list[dict[str, object]] = []
    if len(records) != len(base_scored):
        raise ValueError("B1 baseline identity vectors do not align")
    for record, scored in zip(records, base_scored):
        reference = accepted[record["image_id"]]
        logits = np.asarray(scored["base_candidate_logits"], dtype=np.float32)
        candidate_indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        winner = int(np.argmax(logits))
        masks = unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        map_hash = _npy_float16_sha256(
            masks[winner] * float(scored["base_bag_probability"])
        )
        scalar_delta = max(
            abs(float(logits[winner]) - float(reference["selected_candidate_logit"])),
            abs(float(scored["base_bag_logit"]) - float(reference["bag_logit"])),
            abs(float(scored["base_bag_probability"]) - float(reference["bag_probability"])),
        )
        identity = bool(
            int(candidate_indices[winner]) == int(reference["selected_candidate_index"])
            and map_hash == reference["map_sha256"]
            and scalar_delta <= 5.0e-6
        )
        if not identity:
            raise RuntimeError(f"B1 accepted baseline identity failed: {record['image_id']}")
        rows.append(
            {
                "image_id": record["image_id"],
                "candidate_count": len(logits),
                "base_logits_sha256": sha256(logits.tobytes()).hexdigest(),
                "maximum_scalar_delta": scalar_delta,
                "selected_index_exact": 1,
                "map_sha256_exact": 1,
                "identity_pass": 1,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _score_arms(
    output_dir: Path,
    records: list[dict[str, Any]],
    base_scored: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
    activations: dict[str, np.ndarray],
    candidate_root: Path,
    candidate_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, float | int]]:
    accepted = {row["image_id"]: row for row in baseline_rows}
    evidence_root = output_dir / "activation_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    evidence_rows: list[dict[str, object]] = []
    arms: dict[str, list[dict[str, Any]]] = {
        "transferred_geometry_upstream": [],
        "three_way_geometry_upstream_bas": [],
    }
    correlations: list[float] = []
    changed_selections = 0
    if len(records) != len(base_scored):
        raise ValueError("B1 activation/base records do not align")
    for index, (record, base) in enumerate(zip(records, base_scored)):
        image_id = str(record["image_id"])
        masks = unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        activation_tensor = torch.from_numpy(activations[image_id])[None, None]
        mask_tensor = torch.from_numpy(masks)[None]
        valid = torch.ones((1, masks.shape[0]), dtype=torch.bool)
        coverage, purity, harmonic = candidate_activation_evidence(
            activation_tensor,
            mask_tensor,
            valid,
        )
        activation_rank = within_bag_percentile_ranks(harmonic, valid)[0]
        base_logits = torch.from_numpy(
            np.asarray(base["base_candidate_logits"], dtype=np.float32)
        )[None]
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if (
            sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]
            or candidate_row["diagnostic_sha256"]
            != record["candidate_payload_sha256"]
        ):
            raise ValueError(f"B1 upstream candidate provenance mismatch: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as candidate_payload:
            all_upstream = np.asarray(
                candidate_payload["selection_scores"], dtype=np.float32
            )
        kept = np.asarray(record["candidate_indices"], dtype=np.int64)
        if (
            all_upstream.ndim != 1
            or np.any(kept < 0)
            or np.any(kept >= len(all_upstream))
            or not np.isfinite(all_upstream).all()
        ):
            raise ValueError(f"B1 upstream score alignment mismatch: {image_id}")
        upstream = torch.from_numpy(all_upstream[kept])[None]
        base_rank = within_bag_percentile_ranks(base_logits, valid)[0]
        upstream_rank = within_bag_percentile_ranks(upstream, valid)[0]
        transferred = equal_rank_aggregate(
            (base_logits, upstream), valid
        )[0]
        three_way = equal_rank_aggregate(
            (base_logits, upstream, harmonic), valid
        )[0]
        if len(kept) > 1:
            correlation = float(
                np.corrcoef(
                    activation_rank.numpy().astype(np.float64),
                    upstream_rank.numpy().astype(np.float64),
                )[0, 1]
            )
            if np.isfinite(correlation):
                correlations.append(correlation)
        changed_selections += int(
            int(torch.argmax(transferred)) != int(torch.argmax(three_way))
        )
        relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        evidence_path = evidence_root / relative
        np.savez_compressed(
            evidence_path,
            activation=np.asarray(activations[image_id], dtype=np.float16),
            candidate_indices=np.asarray(record["candidate_indices"], dtype=np.int32),
            coverage=coverage[0].numpy().astype(np.float32),
            purity=purity[0].numpy().astype(np.float32),
            harmonic=harmonic[0].numpy().astype(np.float32),
            activation_rank=activation_rank.numpy().astype(np.float32),
            baseline_logits=np.asarray(base["base_candidate_logits"], dtype=np.float32),
            upstream_scores=upstream[0].numpy().astype(np.float32),
            upstream_rank=upstream_rank.numpy().astype(np.float32),
            transferred_rank=transferred.numpy().astype(np.float32),
            three_way_rank=three_way.numpy().astype(np.float32),
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_count": masks.shape[0],
                "evidence_path": str(relative),
                "evidence_sha256": sha256_file(evidence_path),
                "activation_range": float(np.ptp(activations[image_id])),
            }
        )
        common = {
            "image_id": image_id,
            "bag_logit": float(accepted[image_id]["bag_logit"]),
            "bag_probability": float(accepted[image_id]["bag_probability"]),
        }
        arms["transferred_geometry_upstream"].append(
            {**common, "candidate_logits": transferred.numpy()}
        )
        arms["three_way_geometry_upstream_bas"].append(
            {**common, "candidate_logits": three_way.numpy()}
        )
    if not correlations:
        raise RuntimeError("B1 complementarity correlation is undefined")
    diagnostics: dict[str, float | int] = {
        "mean_bas_upstream_rank_correlation": float(np.mean(correlations)),
        "correlation_images": len(correlations),
        "three_way_changed_selections": changed_selections,
        "three_way_changed_selection_fraction": changed_selections / 371.0,
    }
    return (
        arms,
        _write_csv(evidence_root / "activation_manifest.csv", evidence_rows),
        diagnostics,
    )


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
        raise ValueError("B1 execution differs from the frozen recipe")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("B1 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"B1 requires Tesla T4 x2, got {device_names}")
    if sha256_file(args.pretrained_checkpoint) != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("B1 ImageNet initialization SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    _seed_everything(args.seed)

    cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)
    split_rows = {
        split: load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("B1 frozen cohort mismatch")
    val_candidate_rows, val_candidate_audit = _audit_candidate_input(
        args.val_candidate_root,
        split_rows["val"],
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    cache, _validated = _load_cache_records(
        args, split_rows, cache_manifest_rows
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
            "epochs": EXPECTED_EPOCHS,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    gate.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "training_history_sha256": history_sha256,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
    )
    gate_path = args.output_dir / "operational_gate.json"
    if not gate["operational_gate_pass"]:
        gate_path.write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
        return

    del parallel, optimizer, scaler, state
    model.cpu()
    del model
    torch.cuda.empty_cache()
    baseline_freeze, baseline_rows = _verify_baseline_freeze(args)
    device = torch.device("cuda:0")
    base_model, base_config = _load_baseline_model(args, device=device)
    base_scored = score_same_family_graph_records(
        cache["val"],
        base_model,
        bag_temperature=base_config.bag_temperature,
        graph_config=SameFamilyGraphConfig(
            minimum_iou=1.0,
            minimum_containment=1.0,
            alpha=0.0,
            iterations=1,
        ),
        batch_size=16,
        device=device,
    )
    identity_rows = _baseline_identity(cache["val"], base_scored, baseline_rows)
    identity_sha256 = _write_csv(args.output_dir / "baseline_identity.csv", identity_rows)
    arms, activation_manifest_sha256, complementarity = _score_arms(
        args.output_dir,
        cache["val"],
        base_scored,
        baseline_rows,
        activations,
        args.val_candidate_root,
        val_candidate_rows,
    )
    gate.update(
        {
            **complementarity,
            "maximum_mean_bas_upstream_rank_correlation": 0.80,
            "minimum_changed_selection_fraction": 0.05,
            "val_candidate_manifest_sha256": val_candidate_audit[
                "manifest_sha256"
            ],
            "val_candidate_summary_sha256": val_candidate_audit[
                "summary_sha256"
            ],
        }
    )
    gate["complementarity_gate_pass"] = bool(
        float(complementarity["mean_bas_upstream_rank_correlation"]) <= 0.80
        and float(complementarity["three_way_changed_selection_fraction"])
        >= 0.05
    )
    gate["operational_gate_pass"] = bool(
        gate["operational_gate_pass"] and gate["complementarity_gate_pass"]
    )
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not gate["operational_gate_pass"]:
        print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
        return
    arm_freezes: dict[str, str] = {}
    for arm_name, scored in arms.items():
        arm_root = args.output_dir / arm_name
        arm_args = SimpleNamespace(output_dir=arm_root)
        prediction_sha256, score_sha256 = _write_validation_outputs(
            arm_args,
            cache["val"],
            scored,
        )
        freeze = {
            "experiment_id": "EXP-20260801-codex-b1-bas-candidate-descriptor-v1",
            "arm": arm_name,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
            "selector_cache_manifest_sha256": cache_freeze["selector_cache_manifest_sha256"],
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
            "baseline_prediction_manifest_sha256": baseline_freeze["prediction_manifest_sha256"],
            "bas_checkpoint_sha256": checkpoint_sha256,
            "training_history_sha256": history_sha256,
            "operational_gate_sha256": sha256_file(gate_path),
            "baseline_identity_sha256": identity_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
            "prediction_manifest_sha256": prediction_sha256,
            "candidate_score_manifest_sha256": score_sha256,
            "validation_predictions": 371,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        arm_freezes[arm_name] = sha256_file(freeze_path)
    pair_freeze = {
        "experiment_id": "EXP-20260801-codex-b1-bas-candidate-descriptor-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "arms": arm_freezes,
        "pair_physically_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    pair_path.write_text(json.dumps(pair_freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_manifest = {
        "run_id": "btxrd_bas_candidate_descriptor_b1_v1",
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
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
