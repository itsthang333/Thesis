from __future__ import annotations

import argparse
import csv
import json
import math
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
from models.losses import (
    bce_dice_loss,
    binary_segmentation_metric_sums,
    dice_coefficient,
    finalize_binary_segmentation_metrics,
    grouped_pseudo_segmentation_loss,
    iou_score,
    soft_boundary_weight_map,
)
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
        help="Legacy WSSS option: hard-ignore this boundary band. Prefer --pseudo-boundary-soft-px.",
    )
    parser.add_argument(
        "--pseudo-boundary-soft-px", type=int, default=0,
        help="WSSS only: down-weight (but never delete) this pseudo-mask boundary band.",
    )
    parser.add_argument(
        "--pseudo-boundary-weight", type=float, default=0.25,
        help="Supervision weight inside --pseudo-boundary-soft-px (0..1).",
    )
    parser.add_argument(
        "--pseudo-bce-weight", type=float, default=0.5,
        help="WSSS loss mix: BCE weight; the remainder is tumor-only soft Dice.",
    )
    parser.add_argument(
        "--consistency-weight", type=float, default=0.0,
        help="WSSS only: weak-to-strong photometric prediction consistency weight.",
    )
    parser.add_argument(
        "--consistency-confidence", type=float, default=0.8,
        help="Use weak predictions >=c or <=1-c as consistency targets.",
    )
    parser.add_argument(
        "--consistency-rampup-epochs", type=int, default=0,
        help="Linearly ramp WSSS consistency from zero over this many epochs.",
    )
    parser.add_argument(
        "--val-thresholds", type=str,
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="WSSS pseudo-validation thresholds. Selection uses tumor-Dice/normal-specificity harmonic mean.",
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
        choices=("auto", "sqrt_auto", "none", "manual"),
        default="auto",
        help=(
            "Foreground weighting for BCE. 'auto' uses the background/foreground ratio, "
            "'sqrt_auto' (recommended for noisy pseudo masks) uses its square root, "
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


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(not 0.0 < item < 1.0 for item in thresholds):
        raise ValueError("--val-thresholds must contain comma-separated values strictly between 0 and 1")
    return tuple(thresholds)


def btxrd_tumor_status_by_name(dataset) -> dict[str, bool]:
    if not hasattr(dataset, "samples"):
        raise ValueError("BTXRD pseudo supervision requires dataset.samples metadata")
    return {
        str(sample["image_id"]): int(sample["tumor_type"]) != 0
        for sample in dataset.samples
    }


def add_metric_sums(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def run_epoch(
    model,
    loader,
    scaler,
    device,
    train: bool,
    optimizer=None,
    pos_weight: float | None = None,
    pseudo_supervision: bool = False,
    group_explicit_metrics: bool = False,
    tumor_status_by_name: dict[str, bool] | None = None,
    pseudo_boundary_ignore_px: int = 0,
    pseudo_boundary_soft_px: int = 0,
    pseudo_boundary_weight: float = 0.25,
    pseudo_bce_weight: float = 0.5,
    consistency_weight: float = 0.0,
    consistency_confidence: float = 0.8,
    metric_thresholds: tuple[float, ...] = (0.5,),
) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0
    metric_totals = {threshold: {} for threshold in metric_thresholds}
    model.train(train)

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks, image_names in progress:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.shape[0]
        status = None
        if pseudo_supervision or group_explicit_metrics:
            if tumor_status_by_name is None:
                raise ValueError("tumor_status_by_name is required for group-explicit BTXRD metrics")
            status = torch.tensor(
                [tumor_status_by_name[str(name)] for name in image_names],
                device=device,
                dtype=torch.bool,
            )

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                if group_explicit_metrics:
                    if pseudo_supervision and pseudo_boundary_ignore_px > 0:
                        k = 2 * pseudo_boundary_ignore_px + 1
                        dilated = F.max_pool2d(masks, kernel_size=k, stride=1, padding=pseudo_boundary_ignore_px)
                        eroded = -F.max_pool2d(-masks, kernel_size=k, stride=1, padding=pseudo_boundary_ignore_px)
                        pixel_weights = dilated.eq(eroded).float()
                    elif pseudo_supervision and pseudo_boundary_soft_px > 0:
                        pixel_weights = soft_boundary_weight_map(
                            masks, pseudo_boundary_soft_px, pseudo_boundary_weight
                        )
                    else:
                        pixel_weights = None
                    supervised, _ = grouped_pseudo_segmentation_loss(
                        logits,
                        masks,
                        status,
                        pos_weight=pos_weight,
                        pixel_weights=pixel_weights,
                        bce_weight=pseudo_bce_weight,
                    )
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

        if group_explicit_metrics:
            probabilities = torch.sigmoid(logits.detach())
            for threshold in metric_thresholds:
                add_metric_sums(
                    metric_totals[threshold],
                    binary_segmentation_metric_sums(
                        probabilities, masks.detach(), status, threshold
                    ),
                )
            display = finalize_binary_segmentation_metrics(metric_totals[metric_thresholds[0]])
            dice_value, iou_value = display["tumor_dice"], display["tumor_iou"]
        else:
            dice_value = float(dice_coefficient(logits.detach(), masks.detach()).item())
            iou_value = float(iou_score(logits.detach(), masks.detach()).item())
            total_dice += dice_value * batch_size
            total_iou += iou_value * batch_size
        total_loss += float(loss.item()) * batch_size
        total_images += batch_size
        progress.set_postfix(loss=loss.item(), dice=dice_value, iou=iou_value)

    if total_images == 0:
        return 0.0, {"dice": 0.0, "iou": 0.0}
    if not group_explicit_metrics:
        return total_loss / total_images, {
            "dice": total_dice / total_images,
            "iou": total_iou / total_images,
            "checkpoint_score": total_dice / total_images,
            "threshold": 0.5,
        }

    threshold_metrics = {
        threshold: finalize_binary_segmentation_metrics(metric_totals[threshold])
        for threshold in metric_thresholds
    }
    best_threshold = max(
        metric_thresholds,
        key=lambda item: (
            threshold_metrics[item]["hmean"],
            threshold_metrics[item]["tumor_dice"],
            -abs(item - 0.5),
        ),
    )
    result = dict(threshold_metrics[best_threshold])
    result.update({
        "dice": result["tumor_dice"],
        "iou": result["tumor_iou"],
        "checkpoint_score": result["hmean"],
        "threshold": float(best_threshold),
        "threshold_metrics": threshold_metrics,
    })
    return total_loss / total_images, result


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
    decision_threshold: float,
    checkpoint_metric: str,
    checkpoint_metrics: dict[str, float],
    training_config: dict[str, object],
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
            "decision_threshold": decision_threshold,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metrics": checkpoint_metrics,
            "training_config": training_config,
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
    if args.pseudo_boundary_soft_px < 0:
        raise ValueError("--pseudo-boundary-soft-px must be >= 0")
    if args.pseudo_boundary_ignore_px > 0 and args.pseudo_boundary_soft_px > 0:
        raise ValueError("Choose either hard boundary ignore or soft boundary weighting, not both")
    if not 0.0 <= args.pseudo_boundary_weight <= 1.0:
        raise ValueError("--pseudo-boundary-weight must be in [0, 1]")
    if not 0.0 <= args.pseudo_bce_weight <= 1.0:
        raise ValueError("--pseudo-bce-weight must be in [0, 1]")
    if args.consistency_weight < 0 or args.consistency_rampup_epochs < 0:
        raise ValueError("Consistency weight and ramp-up epochs must be >= 0")
    if not 0.5 < args.consistency_confidence < 1.0:
        raise ValueError("--consistency-confidence must be in (0.5, 1.0)")
    if args.train_pseudo_mask_root is None and (
        args.pseudo_boundary_ignore_px > 0
        or args.pseudo_boundary_soft_px > 0
        or args.consistency_weight > 0
    ):
        raise ValueError("Pseudo robustness flags require --train-pseudo-mask-root/--val-pseudo-mask-root")
    val_thresholds = parse_thresholds(args.val_thresholds)
    seed_everything(args.seed)

    train_dataset, val_dataset = build_datasets(args)
    supervision = "pseudo_masks" if args.train_pseudo_mask_root is not None else "ground_truth"
    print(f"segmentation_supervision={supervision}")
    group_explicit_metrics = args.dataset == "btxrd"
    train_status = btxrd_tumor_status_by_name(train_dataset) if group_explicit_metrics else None
    val_status = btxrd_tumor_status_by_name(val_dataset) if group_explicit_metrics else None
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
        raw_pos_weight = compute_pos_weight(train_dataset)
        pos_weight = math.sqrt(raw_pos_weight) if args.pos_weight_mode == "sqrt_auto" else raw_pos_weight
        print(f"raw_background_foreground_ratio={raw_pos_weight:.6f}")
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
    best_val_score = -1.0
    epochs_without_improvement = 0

    training_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "pos_weight_mode": args.pos_weight_mode,
        "pos_weight": pos_weight,
        "pseudo_boundary_ignore_px": args.pseudo_boundary_ignore_px,
        "pseudo_boundary_soft_px": args.pseudo_boundary_soft_px,
        "pseudo_boundary_weight": args.pseudo_boundary_weight,
        "pseudo_bce_weight": args.pseudo_bce_weight,
        "consistency_weight": args.consistency_weight,
        "consistency_confidence": args.consistency_confidence,
        "consistency_rampup_epochs": args.consistency_rampup_epochs,
        "val_thresholds": list(val_thresholds),
    }

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
            **training_config,
        }, indent=2) + "\n", encoding="utf-8",
    )
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "train_tumor_dice", "train_tumor_iou",
            "train_normal_specificity", "train_normal_fp_pixel_rate", "train_hmean",
            "val_loss", "val_tumor_dice", "val_tumor_iou", "val_tumor_precision",
            "val_tumor_recall", "val_normal_specificity", "val_normal_fp_pixel_rate",
            "val_hmean", "val_threshold", "blank_tumor_val_images", "lr",
        ])

    for epoch in range(1, args.epochs + 1):
        current_consistency_weight = args.consistency_weight
        if args.consistency_rampup_epochs > 0:
            current_consistency_weight *= min(1.0, epoch / args.consistency_rampup_epochs)
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            scaler,
            device,
            train=True,
            optimizer=optimizer,
            pos_weight=pos_weight,
            pseudo_supervision=supervision == "pseudo_masks",
            group_explicit_metrics=group_explicit_metrics,
            tumor_status_by_name=train_status,
            pseudo_boundary_ignore_px=(args.pseudo_boundary_ignore_px if supervision == "pseudo_masks" else 0),
            pseudo_boundary_soft_px=(args.pseudo_boundary_soft_px if supervision == "pseudo_masks" else 0),
            pseudo_boundary_weight=args.pseudo_boundary_weight,
            pseudo_bce_weight=args.pseudo_bce_weight,
            consistency_weight=(current_consistency_weight if supervision == "pseudo_masks" else 0.0),
            consistency_confidence=args.consistency_confidence,
            metric_thresholds=(0.5,),
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            scaler,
            device,
            train=False,
            pos_weight=pos_weight,
            pseudo_supervision=supervision == "pseudo_masks",
            group_explicit_metrics=group_explicit_metrics,
            tumor_status_by_name=val_status,
            pseudo_boundary_soft_px=(args.pseudo_boundary_soft_px if supervision == "pseudo_masks" else 0),
            pseudo_boundary_weight=args.pseudo_boundary_weight,
            pseudo_bce_weight=args.pseudo_bce_weight,
            consistency_weight=0.0,
            consistency_confidence=args.consistency_confidence,
            metric_thresholds=(val_thresholds if group_explicit_metrics else (0.5,)),
        )

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_metrics["dice"],
                    train_metrics["iou"],
                    train_metrics.get("normal_specificity", ""),
                    train_metrics.get("normal_fp_pixel_rate", ""),
                    train_metrics.get("hmean", train_metrics["dice"]),
                    val_loss,
                    val_metrics["dice"],
                    val_metrics["iou"],
                    val_metrics.get("tumor_precision", ""),
                    val_metrics.get("tumor_recall", ""),
                    val_metrics.get("normal_specificity", ""),
                    val_metrics.get("normal_fp_pixel_rate", ""),
                    val_metrics.get("hmean", val_metrics["dice"]),
                    val_metrics["threshold"],
                    val_metrics.get("blank_tumor_images", ""),
                    float(optimizer.param_groups[0]["lr"]),
                ]
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_tumor_dice={val_metrics['dice']:.4f} "
            f"val_normal_specificity={val_metrics.get('normal_specificity', float('nan')):.4f} "
            f"val_score={val_metrics['checkpoint_score']:.4f} threshold={val_metrics['threshold']:.2f} "
            f"lr={current_lr:.3e}"
        )

        save_checkpoint(
            args.output_dir / "last_unet.pt", model, optimizer, epoch,
            max(best_val_score, val_metrics["checkpoint_score"]),
            args.dataset, args.image_size, supervision, args.base_channels,
            val_metrics["threshold"],
            (f"{supervision}_tumor_dice_normal_specificity_hmean" if group_explicit_metrics else "dice"),
            {key: value for key, value in val_metrics.items() if isinstance(value, (int, float))},
            training_config,
        )
        if val_metrics["checkpoint_score"] > best_val_score + args.early_stop_min_delta:
            best_val_score = val_metrics["checkpoint_score"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best_unet.pt", model, optimizer, epoch, best_val_score,
                args.dataset, args.image_size, supervision, args.base_channels,
                val_metrics["threshold"],
                (f"{supervision}_tumor_dice_normal_specificity_hmean" if group_explicit_metrics else "dice"),
                {key: value for key, value in val_metrics.items() if isinstance(value, (int, float))},
                training_config,
            )
            print(
                f"--> Saved new best model: score={best_val_score:.4f}, "
                f"tumor_dice={val_metrics['dice']:.4f}, threshold={val_metrics['threshold']:.2f}"
            )
        else:
            epochs_without_improvement += 1

        print(
            f"convergence: best_val_score={best_val_score:.4f} "
            f"no_improvement={epochs_without_improvement}/{args.early_stop_patience or 'off'} "
            f"min_delta={args.early_stop_min_delta:.4f}"
        )

        if lr_scheduler is not None:
            previous_lr = float(optimizer.param_groups[0]["lr"])
            lr_scheduler.step(val_metrics["checkpoint_score"])
            updated_lr = float(optimizer.param_groups[0]["lr"])
            if updated_lr < previous_lr:
                print(f"--> ReduceLROnPlateau: lr {previous_lr:.3e} -> {updated_lr:.3e}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: validation checkpoint score did not improve for "
                f"{epochs_without_improvement} epochs (patience={args.early_stop_patience}). "
                f"Best score={best_val_score:.4f}."
            )
            break


if __name__ == "__main__":
    main()
