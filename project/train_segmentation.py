from __future__ import annotations

"""Locked ResNet18-U-Net fully-supervised comparison trainer."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.factory import build_segmentation_dataset
from models.losses import bce_dice_loss
from models.unet import architecture_metadata, build_segmentation_model


HISTORY_FIELDS = [
    "epoch", "train_loss", "train_dice", "train_iou", "train_positive_dice",
    "train_empty_specificity", "val_loss", "val_dice", "val_iou",
    "val_positive_dice", "val_empty_specificity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-profile", choices=("btxrd_best",), default="btxrd_best")
    parser.add_argument(
        "--supervision-mode",
        choices=("fully_supervised_comparison",),
        default="fully_supervised_comparison",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--model-architecture", choices=("resnet18_unet",), default="resnet18_unet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--checkpoint-dice-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--pos-weight-mode", choices=("manual",), default="manual")
    parser.add_argument("--pos-weight-value", type=float, default=10.0)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sample_metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    predictions = (torch.sigmoid(logits) >= 0.5).float().flatten(1)
    targets = masks.float().flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    pred_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)
    dice = (2.0 * intersection + 1.0e-6) / (pred_sum + target_sum + 1.0e-6)
    iou = (intersection + 1.0e-6) / (pred_sum + target_sum - intersection + 1.0e-6)
    positive = target_sum > 0
    empty = ~positive
    return {
        "dice_sum": float(dice.sum().item()),
        "iou_sum": float(iou.sum().item()),
        "positive_dice_sum": float(dice[positive].sum().item()),
        "positive_count": int(positive.sum().item()),
        "empty_correct": int((pred_sum[empty] == 0).sum().item()),
        "empty_count": int(empty.sum().item()),
        "count": int(masks.shape[0]),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer,
    pos_weight: float,
) -> tuple[float, dict[str, float]]:
    model.train(train)
    totals = {
        "loss": 0.0, "dice_sum": 0.0, "iou_sum": 0.0,
        "positive_dice_sum": 0.0, "positive_count": 0,
        "empty_correct": 0, "empty_count": 0, "count": 0,
    }
    for images, masks, _image_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images)
                loss = bce_dice_loss(logits, masks, pos_weight=pos_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        metrics = sample_metrics(logits.detach(), masks.detach())
        totals["loss"] += float(loss.item()) * metrics["count"]
        for key in metrics:
            totals[key] += metrics[key]
    count = int(totals["count"])
    if count == 0:
        raise RuntimeError("empty training/evaluation loader")
    return float(totals["loss"] / count), {
        "dice": float(totals["dice_sum"] / count),
        "iou": float(totals["iou_sum"] / count),
        "positive_dice": float(totals["positive_dice_sum"] / max(1, totals["positive_count"])),
        "empty_specificity": float(totals["empty_correct"] / max(1, totals["empty_count"])),
    }


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    source = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}


def save_checkpoint(
    path: Path,
    *,
    state_dict: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_dice: float,
    best_specificity: float,
    split_sha256: str,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset": "btxrd",
        "architecture": architecture_metadata("resnet18_unet"),
        "model_architecture": "resnet18_unet",
        "pretrained_encoder": True,
        "image_size": args.image_size,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "split_manifest_sha256": split_sha256,
        "decision_threshold": 0.5,
        "seed": args.seed,
        "pos_weight": args.pos_weight_value,
        "best_metric": best_dice,
        "best_metric_name": "val_positive_dice",
        "best_tiebreak_metric": best_specificity,
        "best_tiebreak_metric_name": "val_normal_empty_case_specificity",
        "checkpoint_selection_rule": (
            "maximize val_positive_dice at threshold 0.50; within tolerance maximize "
            "val_normal_empty_case_specificity"
        ),
        "supervision_mode": "fully_supervised_comparison",
        "ground_truth_spatial_supervision": True,
        "validation_ground_truth_checkpoint_selection": True,
        "comparison_only": True,
        "wsss_eligible": False,
        "train_pred_mask_root": None,
        "val_pred_mask_root": None,
        "test_evaluated": False,
        "scientific_config": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "maximum_epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "pos_weight": args.pos_weight_value,
        },
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    if args.train_split != "train" or args.val_split != "val":
        raise ValueError("fully-supervised comparison is locked to train/val")
    locked = {
        "image_size": (args.image_size, 448),
        "batch_size": (args.batch_size, 8),
        "lr": (args.lr, 1.0e-4),
        "weight_decay": (args.weight_decay, 1.0e-4),
        "epochs": (args.epochs, 30),
        "seed": (args.seed, 42),
        "early_stop_patience": (args.early_stop_patience, 10),
        "pos_weight_value": (args.pos_weight_value, 10.0),
    }
    changed = {key: values for key, values in locked.items() if values[0] != values[1]}
    if changed:
        raise ValueError(f"fully-supervised configuration differs from locked protocol: {changed}")
    if not torch.cuda.is_available():
        raise RuntimeError("fully-supervised training requires CUDA")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True)
    split_sha = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    train_dataset = build_segmentation_dataset(
        root=args.data_root, split="train", image_size=args.image_size,
        augment=True, split_manifest=args.split_manifest,
    )
    val_dataset = build_segmentation_dataset(
        root=args.data_root, split="val", image_size=args.image_size,
        augment=False, split_manifest=args.split_manifest,
    )
    if len(train_dataset) != 2981 or len(val_dataset) != 371:
        raise ValueError("canonical train/validation counts differ")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    device = torch.device("cuda:0")
    model = build_segmentation_model("resnet18_unet", pretrained=True).to(device)
    device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if args.multi_gpu and len(device_names) > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    history_path = args.output_dir / "training_log.csv"
    best_dice = -1.0
    best_specificity = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_epoch(
                model, train_loader, device, scaler, train=True,
                optimizer=optimizer, pos_weight=args.pos_weight_value,
            )
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, scaler, train=False,
                optimizer=optimizer, pos_weight=args.pos_weight_value,
            )
            global_step += len(train_loader)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_positive_dice": train_metrics["positive_dice"],
                "train_empty_specificity": train_metrics["empty_specificity"],
                "val_loss": val_loss,
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_positive_dice": val_metrics["positive_dice"],
                "val_empty_specificity": val_metrics["empty_specificity"],
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            materially_better = val_metrics["positive_dice"] > best_dice + args.checkpoint_dice_tolerance
            tied_better = (
                abs(val_metrics["positive_dice"] - best_dice) <= args.checkpoint_dice_tolerance
                and val_metrics["empty_specificity"] > best_specificity
            )
            if materially_better or tied_better:
                best_dice = val_metrics["positive_dice"]
                best_specificity = val_metrics["empty_specificity"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    args.output_dir / "best_unet.pt", state_dict=model_state(model),
                    optimizer=optimizer, epoch=epoch, global_step=global_step,
                    best_dice=best_dice, best_specificity=best_specificity,
                    split_sha256=split_sha, args=args,
                )
            else:
                epochs_without_improvement += 1
            save_checkpoint(
                args.output_dir / "last_unet.pt", state_dict=model_state(model),
                optimizer=optimizer, epoch=epoch, global_step=global_step,
                best_dice=best_dice, best_specificity=best_specificity,
                split_sha256=split_sha, args=args,
            )
            if epochs_without_improvement >= args.early_stop_patience:
                break
    best_path = args.output_dir / "best_unet.pt"
    last_path = args.output_dir / "last_unet.pt"
    metadata = {
        "status": "complete",
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "train_split": "train",
        "val_split": "val",
        "image_size": args.image_size,
        "architecture": architecture_metadata("resnet18_unet"),
        "pretrained_encoder": True,
        "supervision_mode": "fully_supervised_comparison",
        "ground_truth_spatial_supervision": True,
        "validation_ground_truth_checkpoint_selection": True,
        "comparison_only": True,
        "wsss_eligible": False,
        "best_epoch": best_epoch,
        "best_val_positive_dice_fixed_0_5": best_dice,
        "best_val_normal_specificity_fixed_0_5": best_specificity,
        "split_manifest_sha256": split_sha,
        "best_checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "last_checkpoint_sha256": hashlib.sha256(last_path.read_bytes()).hexdigest(),
        "training_log_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "cuda_device_names": device_names,
        "data_parallel": isinstance(model, nn.DataParallel),
        "test_evaluated": False,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
