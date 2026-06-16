from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SegmentationConfig
from datasets.fracatlas import FracAtlasSegmentationDataset, build_train_val_indices
from models.losses import bce_dice_loss, dice_coefficient, iou_score
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on pseudo masks")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "FracAtlas")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=ROOT / "outputs" / "pseudo_masks" / "masks")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=SegmentationConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SegmentationConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=SegmentationConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=SegmentationConfig.epochs)
    parser.add_argument("--val-fraction", type=float, default=SegmentationConfig.val_fraction)
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

import torch.nn.functional as F

def focal_dice_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    # 1. Tính Focal Loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss) # Xác suất dự đoán đúng
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    focal_loss = focal_loss.mean()
    
    # 2. Tính Dice Loss
    probs = torch.sigmoid(logits)
    smooth = 1e-6
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score.mean()
    
    # Trọng số kết hợp: 1 * Focal + 1 * Dice (có thể tinh chỉnh sau)
    return focal_loss + dice_loss

def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0
    if train:
        model.train()
    else:
        model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks, _ in progress:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = focal_dice_loss(logits, masks)

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
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    image_root = args.image_root or (args.data_root / "images")
    dataset = FracAtlasSegmentationDataset(
        image_roots=image_root,
        mask_root=args.mask_root,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
    )
    train_indices, val_indices = build_train_val_indices(len(dataset), val_fraction=args.val_fraction, seed=args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / "training_log.csv"
    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                epoch,
                train_loss,
                train_metrics["dice"],
                train_metrics["iou"],
                val_loss,
                val_metrics["dice"],
                val_metrics["iou"],
            ])

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_val_loss)


if __name__ == "__main__":
    main()
