from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.factory import build_segmentation_dataset
from models.losses import bce_dice_loss, dice_coefficient, iou_score
from models.unet import UNet
from runtime_devices import prepare_data_parallel, resolve_gpu_count, unwrap_model


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
    parser.add_argument(
        "--num-gpus", type=int, choices=(0, 1, 2), default=0,
        help="0=auto-detect up to two GPUs; 1=single GPU; 2=DataParallel on two GPUs.",
    )
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument(
        "--pseudo-boundary-ignore-px", type=int, default=0,
        help="WSSS only: exclude this many pixels around pseudo-mask boundaries from supervised loss.",
    )
    parser.add_argument(
        "--consistency-weight", type=float, default=0.0,
        help="WSSS only: weak-to-strong photometric prediction consistency weight.",
    )
    parser.add_argument(
        "--consistency-confidence", type=float, default=0.8,
        help="Use weak predictions >=c or <=1-c as consistency targets.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "segmentation")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument(
        "--train-pseudo-mask-root", type=Path, default=None,
        help="BTXRD WSSS: train on PNG pseudo masks instead of LabelMe ground truth.",
    )
    parser.add_argument(
        "--val-pseudo-mask-root", type=Path, default=None,
        help="BTXRD WSSS: select checkpoints against pseudo masks, never GT validation masks.",
    )
    parser.add_argument(
        "--allow-partial-train-pseudo-masks", action="store_true",
        help="Pilot only: filter train samples to PNGs present under --train-pseudo-mask-root.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training if val_dice does not improve for this many consecutive "
                        "epochs. 0 disables early stopping (always run the full --epochs).")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                        help="Minimum pseudo-val Dice gain required to reset patience.")
    parser.add_argument("--lr-plateau-patience", type=int, default=0,
                        help="Reduce LR after this many flat pseudo-val epochs; 0 disables it.")
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
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
    if (args.train_pseudo_mask_root is None) != (args.val_pseudo_mask_root is None):
        raise ValueError(
            "Provide both --train-pseudo-mask-root and --val-pseudo-mask-root so a WSSS run "
            "cannot accidentally select checkpoints using GT validation masks."
        )
    if args.dataset != "btxrd" and args.train_pseudo_mask_root is not None:
        raise ValueError("Pseudo-mask-root training is currently implemented for BTXRD only.")
    train_dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
        pseudo_mask_root=args.train_pseudo_mask_root,
        require_all_pseudo_masks=not args.allow_partial_train_pseudo_masks,
    )
    val_dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.val_split,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
        pseudo_mask_root=args.val_pseudo_mask_root,
        require_all_pseudo_masks=True,
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
    pseudo_boundary_ignore_px: int = 0,
    consistency_weight: float = 0.0,
    consistency_confidence: float = 0.8,
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
                if pseudo_boundary_ignore_px > 0:
                    k = 2 * pseudo_boundary_ignore_px + 1
                    dilated = F.max_pool2d(masks, kernel_size=k, stride=1, padding=pseudo_boundary_ignore_px)
                    eroded = -F.max_pool2d(-masks, kernel_size=k, stride=1, padding=pseudo_boundary_ignore_px)
                    valid = (dilated == eroded).float()
                    pixel_bce = F.binary_cross_entropy_with_logits(
                        logits, masks,
                        pos_weight=(torch.as_tensor(pos_weight, device=device) if pos_weight is not None else None),
                        reduction="none",
                    )
                    supervised = (pixel_bce * valid).sum() / valid.sum().clamp_min(1.0)
                    probs = torch.sigmoid(logits)
                    intersection = (probs * masks * valid).sum(dim=(1, 2, 3))
                    denominator = ((probs + masks) * valid).sum(dim=(1, 2, 3))
                    supervised = 0.5 * supervised + 0.5 * (1.0 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean())
                else:
                    supervised = bce_dice_loss(logits, masks, pos_weight=pos_weight)
                loss = supervised
                if train and consistency_weight > 0:
                    # Spatial geometry is unchanged, so no inverse warp is
                    # needed. Contrast/noise make a strong radiographic view.
                    scale = torch.empty((images.shape[0], 1, 1, 1), device=device).uniform_(0.8, 1.2)
                    strong_images = images * scale + 0.05 * torch.randn_like(images)
                    strong_logits = model(strong_images)
                    weak_prob = torch.sigmoid(logits.detach())
                    strong_prob = torch.sigmoid(strong_logits)
                    confident = (
                        (weak_prob >= consistency_confidence)
                        | (weak_prob <= 1.0 - consistency_confidence)
                    ).float()
                    consistency = (
                        ((strong_prob - weak_prob).pow(2) * confident).sum()
                        / confident.sum().clamp_min(1.0)
                    )
                    loss = loss + consistency_weight * consistency

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
    image_size: int,
    supervision: str,
    base_channels: int,
    pseudo_boundary_ignore_px: int,
    consistency_weight: float,
    consistency_confidence: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "dataset": dataset,
            "image_size": image_size,
            "supervision": supervision,
            "base_channels": base_channels,
            "pseudo_boundary_ignore_px": pseudo_boundary_ignore_px,
            "consistency_weight": consistency_weight,
            "consistency_confidence": consistency_confidence,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.early_stop_min_delta < 0 or args.lr_plateau_patience < 0:
        raise ValueError("Early-stop min_delta and LR plateau patience must be >= 0")
    if not 0.0 < args.lr_plateau_factor < 1.0 or args.min_lr < 0:
        raise ValueError("Require 0 < lr_plateau_factor < 1 and min_lr >= 0")
    if args.pseudo_boundary_ignore_px < 0:
        raise ValueError("--pseudo-boundary-ignore-px must be >= 0")
    if args.consistency_weight < 0:
        raise ValueError("--consistency-weight must be >= 0")
    if not 0.5 < args.consistency_confidence < 1.0:
        raise ValueError("--consistency-confidence must be in (0.5, 1.0)")
    if args.train_pseudo_mask_root is None and (
        args.pseudo_boundary_ignore_px > 0 or args.consistency_weight > 0
    ):
        raise ValueError("Pseudo robustness flags require --train-pseudo-mask-root/--val-pseudo-mask-root")
    seed_everything(args.seed)

    train_dataset, val_dataset = build_datasets(args)
    supervision = "pseudo_masks" if args.train_pseudo_mask_root is not None else "ground_truth"
    print(f"segmentation_supervision={supervision}")
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

    num_gpus = resolve_gpu_count(args.num_gpus)
    model = UNet(in_channels=3, out_channels=1, base_channels=args.base_channels)
    model, device = prepare_data_parallel(model, num_gpus)
    print(f"training_device={device} num_gpus={num_gpus} data_parallel={num_gpus > 1}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience, min_lr=args.min_lr,
        )
        if args.lr_plateau_patience > 0 else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = args.output_dir / "training_log.csv"
    # Ensure epoch 1 is checkpointed even if a very noisy pseudo-label run
    # initially produces Dice=0.  Subsequent improvements still obey min_delta.
    best_val_dice = -1.0
    epochs_without_improvement = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps({
            "dataset": args.dataset,
            "supervision": supervision,
            "train_pseudo_mask_root": str(args.train_pseudo_mask_root.resolve()) if args.train_pseudo_mask_root else None,
            "val_pseudo_mask_root": str(args.val_pseudo_mask_root.resolve()) if args.val_pseudo_mask_root else None,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "num_gpus": num_gpus,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "lr_plateau_patience": args.lr_plateau_patience,
            "lr_plateau_factor": args.lr_plateau_factor,
            "min_lr": args.min_lr,
            "pseudo_boundary_ignore_px": args.pseudo_boundary_ignore_px,
            "consistency_weight": args.consistency_weight,
            "consistency_confidence": args.consistency_confidence,
        }, indent=2) + "\n", encoding="utf-8",
    )
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
            pseudo_boundary_ignore_px=(args.pseudo_boundary_ignore_px if supervision == "pseudo_masks" else 0),
            consistency_weight=(args.consistency_weight if supervision == "pseudo_masks" else 0.0),
            consistency_confidence=args.consistency_confidence,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            scaler,
            device,
            train=False,
            pos_weight=pos_weight,
            pseudo_boundary_ignore_px=0,
            consistency_weight=0.0,
            consistency_confidence=args.consistency_confidence,
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

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} lr={current_lr:.3e}"
        )

        save_checkpoint(
            args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_dice,
            args.dataset, args.image_size, supervision, args.base_channels,
            args.pseudo_boundary_ignore_px, args.consistency_weight, args.consistency_confidence,
        )
        if val_metrics["dice"] > best_val_dice + args.early_stop_min_delta:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best_unet.pt", model, optimizer, epoch, best_val_dice,
                args.dataset, args.image_size, supervision, args.base_channels,
                args.pseudo_boundary_ignore_px, args.consistency_weight, args.consistency_confidence,
            )
            print(f"--> Saved new best model with Dice = {best_val_dice:.4f}")
        else:
            epochs_without_improvement += 1

        print(
            f"convergence: best_val_dice={best_val_dice:.4f} "
            f"no_improvement={epochs_without_improvement}/{args.early_stop_patience or 'off'} "
            f"min_delta={args.early_stop_min_delta:.4f}"
        )

        if lr_scheduler is not None:
            previous_lr = float(optimizer.param_groups[0]["lr"])
            lr_scheduler.step(val_metrics["dice"])
            updated_lr = float(optimizer.param_groups[0]["lr"])
            if updated_lr < previous_lr:
                print(f"--> ReduceLROnPlateau: lr {previous_lr:.3e} -> {updated_lr:.3e}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: val_dice did not improve for {epochs_without_improvement} epochs "
                f"(patience={args.early_stop_patience}). Best val_dice={best_val_dice:.4f}."
            )
            break


if __name__ == "__main__":
    main()
