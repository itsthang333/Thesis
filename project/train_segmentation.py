from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SegmentationConfig
from datasets.ramh1200 import RAMH1200SegmentationDataset
from models.losses import bce_dice_loss, dice_coefficient, iou_score
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on RAM-H1200 bone masks")
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--annotation-name", type=str, default="_annotations_bone_rle.coco.json")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=SegmentationConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SegmentationConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=SegmentationConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=SegmentationConfig.epochs)
    parser.add_argument("--seed", type=int, default=SegmentationConfig.seed)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "segmentation")
    parser.add_argument("--use-clahe", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def build_datasets(args: argparse.Namespace) -> tuple[RAMH1200SegmentationDataset, RAMH1200SegmentationDataset]:
    train_dataset = RAMH1200SegmentationDataset(
        root=args.ram_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
    )
    val_dataset = RAMH1200SegmentationDataset(
        root=args.ram_root,
        split=args.val_split,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
    )
    print(
        f"Loaded RAM-H1200: {len(train_dataset)} train images from {args.train_split}, "
        f"{len(val_dataset)} validation images from {args.val_split}."
    )
    return train_dataset, val_dataset


def run_epoch(model, loader, scaler, device, train: bool, optimizer=None) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0
    model.train(train)

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks, _ in progress:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = bce_dice_loss(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice = dice_coefficient(logits.detach(), masks.detach())
        iou = iou_score(logits.detach(), masks.detach())
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        batches += 1
        progress.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    if batches == 0:
        return 0.0, {"dice": 0.0, "iou": 0.0}
    return total_loss / batches, {"dice": total_dice / batches, "iou": total_iou / batches}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "dataset": "RAM-H1200",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_dataset, val_dataset = build_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = args.output_dir / "training_log.csv"
    best_val_dice = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, scaler, device, train=True, optimizer=optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, scaler, device, train=False)

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_metrics["dice"],
                    train_metrics["iou"],
                    val_loss,
                    val_metrics["dice"],
                    val_metrics["iou"],
                ]
            )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        save_checkpoint(args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_dice)
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(args.output_dir / "best_unet.pt", model, optimizer, epoch, best_val_dice)
            print(f"--> Saved new best model with Dice = {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
