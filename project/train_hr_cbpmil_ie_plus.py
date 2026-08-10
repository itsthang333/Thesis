from __future__ import annotations

"""Fixed one-shot HR-CBPMIL-IE+ training (image labels only, no test access)."""

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.hr_cbpmil_bags import (
    HRCBPMILBagDataset,
    build_cluster_cache,
    collate_hr_cbpmil_bags,
    load_cluster_cache,
    write_data_boundary_receipt,
)
from frozen_io import load_split_rows_without_annotations, sha256_file
from models.hr_cbpmil_ie_plus import HRCBPMILIEPlus, hr_cbpmil_loss
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--train-pseudo-manifest-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stop-after-epoch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    model_state = unwrap(model).state_dict()
    for name, value in ema.state_dict().items():
        source = model_state[name].detach()
        if torch.is_floating_point(value):
            value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            value.copy_(source)


def learning_rate_scale(global_step: int, steps_per_epoch: int, epochs: int) -> float:
    warmup_steps = 2 * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    if global_step <= warmup_steps:
        return max(global_step, 1) / max(warmup_steps, 1)
    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def label_safe_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(
                batch["image"], batch["candidate_masks"], batch["candidate_valid"], batch["cluster_ids"]
            )
        for image_id, label, probability in zip(
            batch["image_id"],
            batch["binary_label"].cpu().tolist(),
            output["image_probability"].float().cpu().tolist(),
            strict=True,
        ):
            rows.append({"image_id": image_id, "label": int(label), "probability": float(probability)})
    return {"images": len(rows), "predictions": rows}


def main() -> None:
    args = parse_args()
    if args.epochs != 30 or not (1 <= args.stop_after_epoch <= 30):
        raise ValueError("Scientific protocol fixes 30 epochs; stop-after only creates a resumable phase")
    if not torch.cuda.is_available():
        raise RuntimeError("HR-CBPMIL-IE+ training requires CUDA")
    if sha256_file(args.classifier_checkpoint) != args.expected_classifier_sha256:
        raise ValueError("Ten-class classifier checkpoint SHA-256 mismatch")
    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="train"
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    train_candidates, train_audit = validate_candidate_diagnostics_manifest(
        args.train_candidate_root,
        expected_image_names=[row["image_id"] for row in train_rows],
        split="train",
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
    )
    val_candidates, val_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
    )
    if train_audit.get("cohort") != "all" or val_audit.get("cohort") != "all":
        raise ValueError("Tumor and normal images require the identical full-gallery policy")
    write_data_boundary_receipt(
        args.output_dir / "data_boundary.json", train_images=len(train_rows), val_images=len(val_rows)
    )

    train_cluster_path = args.output_dir / "train_duplicate_clusters.npz"
    val_cluster_path = args.output_dir / "val_duplicate_clusters.npz"
    train_clusters = (
        load_cluster_cache(train_cluster_path, train_candidates)
        if train_cluster_path.is_file()
        else build_cluster_cache(args.train_candidate_root, train_candidates, train_cluster_path)
    )
    val_clusters = (
        load_cluster_cache(val_cluster_path, val_candidates)
        if val_cluster_path.is_file()
        else build_cluster_cache(args.val_candidate_root, val_candidates, val_cluster_path)
    )
    train_dataset = HRCBPMILBagDataset(
        train_rows,
        dataset_root=args.dataset_root,
        candidate_root=args.train_candidate_root,
        candidate_rows=train_candidates,
        cluster_cache=train_clusters,
        augment=True,
    )
    val_dataset = HRCBPMILBagDataset(
        val_rows,
        dataset_root=args.dataset_root,
        candidate_root=args.val_candidate_root,
        candidate_rows=val_candidates,
        cluster_cache=val_clusters,
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_hr_cbpmil_bags
    )

    device = torch.device("cuda:0")
    base_model = HRCBPMILIEPlus(args.classifier_checkpoint).to(device)
    ema = deepcopy(base_model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    model: nn.Module = base_model
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(base_model, device_ids=[0, 1])
    parameters = [
        {"params": base_model.backbone.features.parameters(), "lr": 1.0e-5},
        {"params": base_model.backbone.classifier10.parameters(), "lr": 1.0e-5},
        {
            "params": [
                parameter
                for name, parameter in base_model.named_parameters()
                if not name.startswith("backbone.features.")
                and not name.startswith("backbone.classifier10.")
            ],
            "lr": 1.0e-4,
        },
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    global_step = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    history: list[dict[str, object]] = []
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume.get("protocol_sha256") != args.protocol_sha256:
            raise ValueError("Resume checkpoint protocol differs")
        base_model.load_state_dict(resume["model_state_dict"], strict=True)
        ema.load_state_dict(resume["ema_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume["global_step"])
        optimizer_steps = int(resume["optimizer_steps"])
        amp_skipped_steps = int(resume["amp_skipped_steps"])
        history = list(resume["history"])

    steps_per_epoch = math.ceil(len(train_dataset) / 2 / 4)
    accumulation = 4
    for epoch in range(start_epoch, args.stop_after_epoch + 1):
        seed_all(args.seed + epoch)
        if epoch == 3:
            ema.load_state_dict(base_model.state_dict(), strict=True)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate_hr_cbpmil_bags,
        )
        base_model.backbone.set_backbone_trainable(epoch >= 3)
        model.train()
        running: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(
                    batch["image"],
                    batch["candidate_masks"],
                    batch["candidate_valid"],
                    batch["cluster_ids"],
                )
                losses = hr_cbpmil_loss(
                    output,
                    batch["binary_label"],
                    batch["class10_label"],
                    batch["candidate_valid"],
                    epoch_number=epoch,
                )
                scaled_loss = losses["total"] / accumulation
            scaler.scale(scaled_loss).backward()
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())
            final_batch = batch_index == len(train_loader)
            if batch_index % accumulation == 0 or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                global_step += 1
                scale = learning_rate_scale(global_step, steps_per_epoch, args.epochs)
                for group, base_lr in zip(optimizer.param_groups, (1.0e-5, 1.0e-5, 1.0e-4), strict=True):
                    group["lr"] = base_lr * scale
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    amp_skipped_steps += 1
                else:
                    optimizer_steps += 1
                    if epoch >= 3:
                        update_ema(ema, model, 0.999)
                optimizer.zero_grad(set_to_none=True)

        validation = label_safe_validation(ema if epoch >= 3 else model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "amp_skipped_steps": amp_skipped_steps,
            "train": {key: value / len(train_loader) for key, value in running.items()},
            "validation": validation,
        }
        history.append(epoch_row)
        checkpoint = {
            "stage": "hr_cbpmil_ie_plus_training_checkpoint_v1",
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "amp_skipped_steps": amp_skipped_steps,
            "model_state_dict": base_model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "classifier_sha256": args.expected_classifier_sha256,
            "train_candidate_manifest_sha256": args.train_candidate_manifest_sha256,
            "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
            "spatial_ground_truth_read": False,
            "test_images_read": 0,
            "test_evaluated": False,
        }
        torch.save(checkpoint, args.output_dir / "last_checkpoint.pt")
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(epoch_row, sort_keys=True), flush=True)

    final = {
        "stage": "hr_cbpmil_ie_plus_training_receipt_v1",
        "epoch": args.stop_after_epoch,
        "complete": args.stop_after_epoch == 30,
        "global_step": global_step,
        "optimizer_steps": optimizer_steps,
        "amp_skipped_steps": amp_skipped_steps,
        "checkpoint_sha256": sha256_file(args.output_dir / "last_checkpoint.pt"),
        "cluster_cache_sha256": {
            "train": sha256_file(train_cluster_path),
            "val": sha256_file(val_cluster_path),
        },
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "training_receipt.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
