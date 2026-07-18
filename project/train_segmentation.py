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

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.factory import build_segmentation_dataset
from models.losses import bce_dice_loss, dice_coefficient, iou_score
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on RAM-H1200 bone masks or BTXRD tumor masks")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=SUPPORTED_DATASETS)
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1",
                        help="Dataset root (RAM-H1200 root or BTXRD root, depending on --dataset)")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--annotation-name", type=str, default="_annotations_bone_rle.coco.json",
                        help="RAM-H1200 only; ignored for --dataset btxrd")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=SegmentationConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SegmentationConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=SegmentationConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=SegmentationConfig.epochs)
    parser.add_argument("--seed", type=int, default=SegmentationConfig.seed)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "segmentation")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training if val_dice does not improve for this many consecutive "
                        "epochs. 0 disables early stopping (always run the full --epochs).")
    parser.add_argument(
        "--pos-weight-mode",
        choices=("auto", "none", "manual"),
        default="auto",
        help=(
            "Foreground weighting for BCE. 'auto' estimates background/foreground pixel ratio "
            "from train masks; 'none' keeps the original unweighted BCE; 'manual' uses "
            "--pos-weight-value."
        ),
    )
    parser.add_argument(
        "--pos-weight-value",
        type=float,
        default=None,
        help="Fixed positive-class weight; required with --pos-weight-mode manual.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def build_datasets(args: argparse.Namespace):
    train_dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
    )
    val_dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.val_split,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
    )
    print(
        f"Loaded {args.dataset}: {len(train_dataset)} train images from {args.train_split}, "
        f"{len(val_dataset)} validation images from {args.val_split}."
    )
    return train_dataset, val_dataset


def compute_pos_weight(train_dataset, num_workers: int = 0, batch_size: int = 32) -> float:
    """Estimate background/foreground pixel ratio from transformed train masks.

    The one-time scan intentionally defaults to ``num_workers=0``. Worker
    subprocesses can hang on Colab/Drive or notebook-backed filesystems, while
    this batched single-process pass is deterministic and runs only once.
    """
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    total_pixels = 0
    foreground_pixels = 0
    for batch_index, (_, masks, _) in enumerate(loader):
        total_pixels += masks.numel()
        foreground_pixels += int((masks > 0.5).sum().item())
        if batch_index % 10 == 0:
            print(f"  pos_weight scan: batch {batch_index}/{len(loader)}", flush=True)

    if foreground_pixels <= 0:
        raise ValueError(
            "No foreground pixels were found in the training masks. "
            "Check BTXRD annotations before training."
        )
    background_pixels = total_pixels - foreground_pixels
    return float(background_pixels / foreground_pixels)


def run_epoch(
    model,
    loader,
    scaler,
    device,
    train: bool,
    optimizer=None,
    pos_weight: float | None = None,
) -> tuple[float, dict[str, float]]:
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
                loss = bce_dice_loss(logits, masks, pos_weight=pos_weight)

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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    dataset: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "dataset": dataset,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_dataset, val_dataset = build_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.pos_weight_mode == "none":
        pos_weight = None
    elif args.pos_weight_mode == "manual":
        if args.pos_weight_value is None:
            raise ValueError("--pos-weight-mode manual requires --pos-weight-value")
        if args.pos_weight_value <= 0:
            raise ValueError("--pos-weight-value must be positive")
        pos_weight = float(args.pos_weight_value)
    else:
        print("Computing pos_weight from train masks (background/foreground)...")
        pos_weight = compute_pos_weight(train_dataset)
    print(f"pos_weight_mode={args.pos_weight_mode} -> pos_weight={pos_weight}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = args.output_dir / "training_log.csv"
    best_val_dice = 0.0
    epochs_without_improvement = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            scaler,
            device,
            train=True,
            optimizer=optimizer,
            pos_weight=pos_weight,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            scaler,
            device,
            train=False,
            pos_weight=pos_weight,
        )

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

        save_checkpoint(args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_dice, args.dataset)
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best_unet.pt", model, optimizer, epoch, best_val_dice, args.dataset)
            print(f"--> Saved new best model with Dice = {best_val_dice:.4f}")
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: val_dice did not improve for {epochs_without_improvement} epochs "
                f"(patience={args.early_stop_patience}). Best val_dice={best_val_dice:.4f}."
            )
            break


if __name__ == "__main__":
    main()
