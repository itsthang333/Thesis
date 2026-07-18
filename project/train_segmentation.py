from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.factory import build_segmentation_dataset
from models.losses import bce_dice_loss, dice_coefficient, iou_score, weighted_bce_dice_loss
from models.unet import UNet
from pseudo.mask_selection import (
    CONFIDENCE_BOUNDARY_UNCERTAIN,
    CONFIDENCE_FOREGROUND_UNCERTAIN,
)


def split_mask_confidence(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """masks is [B, 1, H, W] (no confidence map available/requested) or
    [B, 2, H, W] (BTXRDSegmentationDataset's pred_confidence_dir channel-1
    stacking, see datasets/btxrd.py). Returns (binary_mask, confidence_or_None)
    with binary_mask always [B, 1, H, W], confidence [B, 1, H, W] float labels
    0-3 (see pseudo/mask_selection.py's CONFIDENCE_* constants) or None.
    """
    if masks.shape[1] == 1:
        return masks, None
    if masks.shape[1] == 2:
        return masks[:, 0:1], masks[:, 1:2]
    raise ValueError(f"Expected mask tensor with 1 or 2 channels, got shape {tuple(masks.shape)}")


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
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Wrap the model in nn.DataParallel to use all visible CUDA devices "
                        "(e.g. Kaggle's T4x2) -- splits each batch across GPUs. No-op with 0 or 1 "
                        "GPU visible. Checkpoints are always saved without the DataParallel "
                        "'module.' prefix, so they load identically with or without this flag.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training if val_dice does not improve by at least --min-delta for "
                        "this many consecutive epochs. 0 disables early stopping (always run the full "
                        "--epochs). This project's own anatomy-aware design doc fixes this at 6 when "
                        "training a WSSS pipeline off pseudo-mask val Dice -- pass "
                        "--early-stop-patience 6 --min-delta 0.002 explicitly to match it (there is no "
                        "--pipeline-profile mechanism in this script yet).")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum val_dice improvement over the current best to count as progress, "
                        "for both --early-stop-patience and --lr-scheduler-plateau. 0 (default) means "
                        "any improvement, however tiny, resets the patience counter -- matching this "
                        "script's original behavior before --min-delta existed.")
    parser.add_argument("--lr-scheduler-plateau", action="store_true",
                        help="Enable torch.optim.lr_scheduler.ReduceLROnPlateau on val_dice (mode="
                        "'max'), reducing LR by --lr-scheduler-factor after --lr-scheduler-patience "
                        "epochs without a >= --min-delta improvement. Off by default (matching prior "
                        "behavior: a constant LR for the whole run) -- opt in for the anatomy-aware "
                        "design doc's config (patience=2, factor=0.5, min_delta=0.002).")
    parser.add_argument("--lr-scheduler-patience", type=int, default=2,
                        help="Epochs without improvement before --lr-scheduler-plateau reduces LR.")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5,
                        help="Multiplicative LR reduction factor for --lr-scheduler-plateau.")
    parser.add_argument("--train-pred-mask-root", type=Path, default=None,
                        help="btxrd only. Directory of pseudo-mask PNGs (generate_pseudo_masks.py's "
                        "masks/ output) for --train-split, one file per image named "
                        "<image_stem>.png. When set, U-Net trains on these WSSS-generated masks "
                        "instead of ground-truth polygons -- this is how to measure the pipeline's "
                        "actual end-to-end segmentation quality rather than a supervised oracle.")
    parser.add_argument("--val-pred-mask-root", type=Path, default=None,
                        help="Same as --train-pred-mask-root but for --val-split. Ground-truth "
                        "polygons are still used for the printed val_dice/val_iou metrics unless "
                        "this is also set; set both consistently unless deliberately measuring "
                        "against GT while training on pseudo-masks.")
    parser.add_argument("--pos-weight-mode", type=str, default="auto", choices=["auto", "none", "manual"],
                        help="How to weight foreground (lesion) pixels in BCE, countering the "
                        "collapse-to-all-background failure found empirically on BTXRD (lesions "
                        "average ~2.6%% of image area, so plain BCE can drive loss low by predicting "
                        "'no lesion anywhere' before learning anything -- observed as val_dice frozen "
                        "at exactly the dataset's normal-image fraction for several epochs). "
                        "'auto' (default) computes background/foreground pixel ratio from the actual "
                        "train-set masks. 'none' disables weighting (original behavior). 'manual' uses "
                        "--pos-weight-value directly.")
    parser.add_argument("--pos-weight-value", type=float, default=None,
                        help="Fixed pos_weight to use when --pos-weight-mode=manual.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Path to a checkpoint (e.g. last_unet.pt) to resume training from -- "
                        "restores model/optimizer state and continues epoch numbering and the "
                        "training log instead of starting over. Needed since this script has no "
                        "built-in resume: without it, re-running after an interrupted session (e.g. "
                        "a disconnected Colab runtime) silently restarts from a fresh model AND "
                        "overwrites training_log.csv, discarding all prior epochs.")
    parser.add_argument("--train-pred-confidence-root", type=Path, default=None,
                        help="btxrd only. Directory of confidence-map PNGs (generate_pseudo_masks.py's "
                        "--save-confidence-map confidence/ output, raw uint8 labels 0-3, see "
                        "pseudo/mask_selection.py's CONFIDENCE_* constants) for --train-split, one "
                        "file per image, matching --train-pred-mask-root's naming. Requires "
                        "--train-pred-mask-root to also be set. Enables --boundary-ignore-loss and/or "
                        "--confidence-weighted-loss.")
    parser.add_argument("--val-pred-confidence-root", type=Path, default=None,
                        help="Same as --train-pred-confidence-root but for --val-split.")
    parser.add_argument("--boundary-ignore-loss", action="store_true",
                        help="Exclude pixels labeled boundary-uncertain (CONFIDENCE_BOUNDARY_UNCERTAIN, "
                        "see pseudo/mask_selection.py) from the BCE+Dice loss entirely -- SAM's own "
                        "mask boundary is typically its least reliable region (this project's own "
                        "oracle diagnostics), so training against it as if it were as trustworthy as "
                        "the mask interior can teach the wrong edge location. Requires "
                        "--train-pred-confidence-root.")
    parser.add_argument("--confidence-weighted-loss", action="store_true",
                        help="Down-weight foreground-uncertain pixels (CONFIDENCE_FOREGROUND_UNCERTAIN) "
                        "in the BCE term by --confidence-uncertain-weight, instead of trusting every "
                        "non-boundary pseudo-mask pixel equally. Requires --train-pred-confidence-root.")
    parser.add_argument("--confidence-uncertain-weight", type=float, default=0.5,
                        help="BCE weight multiplier for foreground-uncertain pixels when "
                        "--confidence-weighted-loss is set (0=ignore them like boundary pixels, "
                        "1=treat them as fully trusted, same as foreground-confident pixels).")
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="Weight for a weak/strong-augmentation consistency loss: two views of "
                        "the same training image (weak=--augment's existing flip, strong=added "
                        "photometric jitter) must agree on high-confidence pixels (see "
                        "--consistency-confidence-threshold). 0 (default) disables it entirely -- "
                        "no extra forward pass, unchanged behavior. Only meaningful during training "
                        "(never applied to validation). This project's own anatomy-aware design doc "
                        "(config.py's BTXRD_ANATOMY_PIPELINE) fixes this at 0.1 when training on "
                        "anatomy-conditioned pseudo-masks -- pass --consistency-weight 0.1 explicitly "
                        "to match it (there is no --pipeline-profile mechanism in this script yet, "
                        "unlike train_classifier.py/generate_pseudo_masks.py).")
    parser.add_argument("--consistency-confidence-threshold", type=float, default=0.80,
                        help="Only penalize prediction disagreement between the weak/strong views on "
                        "pixels where the weak view's predicted probability is >= this threshold or "
                        "<= 1-this threshold (i.e. the model itself is already confident) -- "
                        "penalizing disagreement on genuinely ambiguous pixels would inject noise "
                        "rather than a useful regularizer.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def build_datasets(args: argparse.Namespace):
    if (args.train_pred_mask_root is not None or args.val_pred_mask_root is not None) and args.dataset != "btxrd":
        raise ValueError("--train-pred-mask-root/--val-pred-mask-root are only supported for --dataset btxrd")
    if args.train_pred_confidence_root is not None and args.train_pred_mask_root is None:
        raise ValueError("--train-pred-confidence-root requires --train-pred-mask-root")
    if args.val_pred_confidence_root is not None and args.val_pred_mask_root is None:
        raise ValueError("--val-pred-confidence-root requires --val-pred-mask-root")
    if (args.boundary_ignore_loss or args.confidence_weighted_loss) and args.train_pred_confidence_root is None:
        raise ValueError(
            "--boundary-ignore-loss/--confidence-weighted-loss require --train-pred-confidence-root"
        )
    if args.train_pred_mask_root is not None and args.val_pred_mask_root is None:
        # This project's own anatomy-aware WSSS design doc requires
        # checkpoint selection/early-stopping/LR-plateau to be driven ONLY
        # by pseudo-mask val Dice, never ground-truth polygon Dice -- but
        # without --val-pred-mask-root, val_dataset falls back to rasterizing
        # the GT polygon (see BTXRDSegmentationDataset's docstring), and
        # best_val_dice/early-stopping/--lr-scheduler-plateau below are ALL
        # keyed on that GT Dice. This is not rejected outright (GT Dice
        # during training is still a legitimate thing to WATCH, e.g. for a
        # supervised-oracle run), but silently doing this in a WSSS run
        # would violate the design doc's separation between pipeline-
        # internal validation and the final GT evaluation stage -- so it's
        # surfaced loudly here instead.
        print(
            "WARNING: --train-pred-mask-root is set (training on pseudo-masks) but "
            "--val-pred-mask-root is NOT set -- best_unet.pt/early-stopping/--lr-scheduler-plateau "
            "will be selected using GROUND-TRUTH POLYGON val Dice, not pseudo-mask val Dice. If this "
            "is a WSSS run, pass --val-pred-mask-root pointing at the same generate_pseudo_masks.py "
            "output used for --val-split, or checkpoint selection is silently peeking at GT."
        )
    train_dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
        pred_mask_dir=args.train_pred_mask_root,
        pred_confidence_dir=args.train_pred_confidence_root,
    )
    val_dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.val_split,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        annotation_name=args.annotation_name,
        pred_mask_dir=args.val_pred_mask_root,
        pred_confidence_dir=args.val_pred_confidence_root,
    )
    print(
        f"Loaded {args.dataset}: {len(train_dataset)} train images from {args.train_split}, "
        f"{len(val_dataset)} validation images from {args.val_split}."
    )
    return train_dataset, val_dataset


def compute_pos_weight(train_dataset, num_workers: int = 0, batch_size: int = 32) -> float:
    """background_pixels / foreground_pixels across the actual train-set
    masks -- used to weight BCE so missing a lesion pixel costs as much as
    a false positive on background, countering the collapse-to-empty-mask
    failure mode found empirically on BTXRD (see bce_dice_loss's docstring).
    Iterates the raw mask tensors already produced by the dataset's own
    transform, so this matches exactly what the model is trained against --
    including pseudo-masks when --train-pred-mask-root is set, since the
    ratio must reflect whatever target the model actually sees, not the
    ground-truth polygon distribution.

    num_workers defaults to 0 (single-process, no DataLoader worker
    subprocesses) rather than the main training loop's --num-workers value.
    An earlier version of this function defaulted to num_workers>0 here and
    was observed to hang indefinitely (30+ minutes, no progress) on Colab --
    a known failure mode of PyTorch DataLoader worker subprocesses combined
    with a Google-Drive-backed FUSE mount (fork() inside a notebook kernel
    plus network-filesystem I/O in each worker is a common deadlock
    trigger). This one-time startup pass is small enough (~3000 images) that
    single-process iteration is an acceptable, safe default; only raise
    num_workers here if you've confirmed it doesn't hang in your environment.
    """
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    total_pixels = 0
    foreground_pixels = 0
    for batch_idx, (_, masks, _) in enumerate(loader):
        binary_mask, _ = split_mask_confidence(masks)
        total_pixels += binary_mask.numel()
        foreground_pixels += float((binary_mask > 0.5).sum().item())
        if batch_idx % 10 == 0:
            print(f"  pos_weight scan: batch {batch_idx}/{len(loader)}", flush=True)
    background_pixels = total_pixels - foreground_pixels
    if foreground_pixels <= 0:
        raise ValueError(
            "No foreground (lesion) pixels found in the entire train set -- pos_weight is "
            "undefined. Check that annotations/pseudo-masks are loading correctly before training."
        )
    return background_pixels / foreground_pixels


_STRONG_PHOTOMETRIC_AUGMENT = None


def _strong_photometric_augment():
    """Lazily built torchvision ColorJitter+blur for --consistency-weight's
    strong view -- kept photometric-only (no geometric change) so the SAME
    pixel location in the weak and strong views should get the SAME
    prediction; a geometric augmentation here would require warping one
    prediction back before comparing them.
    """
    global _STRONG_PHOTOMETRIC_AUGMENT
    if _STRONG_PHOTOMETRIC_AUGMENT is None:
        from torchvision import transforms as tv_transforms
        _STRONG_PHOTOMETRIC_AUGMENT = tv_transforms.Compose([
            tv_transforms.ColorJitter(brightness=0.3, contrast=0.3),
            tv_transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ])
    return _STRONG_PHOTOMETRIC_AUGMENT


def consistency_loss(
    logits_weak: torch.Tensor, logits_strong: torch.Tensor, confidence_threshold: float = 0.80
) -> torch.Tensor:
    """MSE between the strong view's predicted probabilities and the weak
    view's (detached) predicted probabilities, restricted to pixels where the
    weak view is already confident (prob >= threshold or <= 1-threshold).
    Detaching the weak-view target means only the strong-view forward pass
    receives a gradient from this term -- the weak view acts as a pseudo-
    label source, not something this loss pulls toward the strong view.
    Returns a differentiable 0.0 (safe to add into a total loss) if no pixel
    in the batch meets the confidence threshold, rather than NaN from an
    empty-tensor mean.
    """
    probs_weak = torch.sigmoid(logits_weak).detach()
    probs_strong = torch.sigmoid(logits_strong)
    confident_mask = (probs_weak >= confidence_threshold) | (probs_weak <= 1.0 - confidence_threshold)
    if not confident_mask.any():
        return logits_strong.sum() * 0.0
    return F.mse_loss(probs_strong[confident_mask], probs_weak[confident_mask])


def run_epoch(
    model, loader, scaler, device, train: bool, optimizer=None, pos_weight: float | None = None,
    boundary_ignore_loss: bool = False,
    confidence_weighted_loss: bool = False,
    confidence_uncertain_weight: float = 0.5,
    consistency_weight: float = 0.0,
    consistency_confidence_threshold: float = 0.80,
) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_consistency = 0.0
    batches = 0
    model.train(train)
    use_confidence_pixel_weight = boundary_ignore_loss or confidence_weighted_loss
    use_consistency = train and consistency_weight > 0

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks, _ in progress:
        images = images.to(device)
        masks = masks.to(device)
        binary_mask, confidence = split_mask_confidence(masks)

        pixel_weight = None
        if use_confidence_pixel_weight:
            if confidence is None:
                raise ValueError(
                    "--boundary-ignore-loss/--confidence-weighted-loss require a confidence map "
                    "(pass --train-pred-confidence-root/--val-pred-confidence-root); this batch's "
                    "mask tensor has no second channel."
                )
            pixel_weight = torch.ones_like(binary_mask)
            if boundary_ignore_loss:
                pixel_weight[confidence == CONFIDENCE_BOUNDARY_UNCERTAIN] = 0.0
            if confidence_weighted_loss:
                # A boundary pixel that also happens to be foreground-uncertain
                # keeps weight 0 from the line above -- boundary_ignore_loss's
                # exclusion always wins over confidence_weighted_loss's partial
                # down-weighting, since 0 is already the strictest possible weight.
                uncertain_mask = confidence == CONFIDENCE_FOREGROUND_UNCERTAIN
                pixel_weight[uncertain_mask] = torch.minimum(
                    pixel_weight[uncertain_mask],
                    torch.full_like(pixel_weight[uncertain_mask], confidence_uncertain_weight),
                )

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                if pixel_weight is not None:
                    loss = weighted_bce_dice_loss(logits, binary_mask, pixel_weight, pos_weight=pos_weight)
                else:
                    loss = bce_dice_loss(logits, binary_mask, pos_weight=pos_weight)

                consistency_value = 0.0
                if use_consistency:
                    with torch.no_grad():
                        strong_images = _strong_photometric_augment()(images)
                    # A SECOND, fully independent forward pass through model
                    # (no activation sharing with the `logits = model(images)`
                    # forward above) -- --consistency-weight > 0 roughly
                    # doubles this batch's compute/memory, not just adds a
                    # cheap extra loss term. consistency_loss() detaches the
                    # weak view's (logits, from the first pass) probabilities
                    # internally, so only THIS second pass receives a
                    # gradient from the consistency term.
                    logits_strong = model(strong_images)
                    consistency = consistency_loss(
                        logits, logits_strong, confidence_threshold=consistency_confidence_threshold
                    )
                    loss = loss + consistency_weight * consistency
                    consistency_value = consistency.item()

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice = dice_coefficient(logits.detach(), binary_mask.detach())
        iou = iou_score(logits.detach(), binary_mask.detach())
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        total_consistency += consistency_value
        batches += 1
        progress.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item(), consistency=consistency_value)

    if batches == 0:
        return 0.0, {"dice": 0.0, "iou": 0.0, "consistency": 0.0}
    return total_loss / batches, {
        "dice": total_dice / batches,
        "iou": total_iou / batches,
        "consistency": total_consistency / batches,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    dataset: str,
    pos_weight: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap nn.DataParallel before saving: DataParallel prefixes every state_dict
    # key with "module." (model.module.inc.block.0.weight instead of
    # model.inc.block.0.weight), so a checkpoint saved from a DataParallel-wrapped
    # model would fail to load into a plain UNet() with strict=True (used by
    # evaluate_unet.py and --resume-from) -- saving the unwrapped state_dict here
    # keeps checkpoints identical whether or not multi-GPU was used to train.
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "dataset": dataset,
            "pos_weight": pos_weight,
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
    # Not saved/restored across --resume-from: its internal "best"/
    # "num_bad_epochs" state always restarts fresh on resume, so a resumed
    # run may reduce LR earlier/later than an uninterrupted one would have.
    # Only the trained weights/optimizer momentum matter for correctness;
    # this only affects the LR schedule's timing, not correctness.
    lr_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience, threshold=max(args.min_delta, 1e-8),
            threshold_mode="abs",
        )
        if args.lr_scheduler_plateau else None
    )

    num_gpus = torch.cuda.device_count()
    use_multi_gpu = args.multi_gpu and device.type == "cuda" and num_gpus > 1
    if args.multi_gpu and not use_multi_gpu:
        print(f"--multi-gpu requested but only {num_gpus} CUDA device(s) visible; running single-device.")

    if args.pos_weight_mode == "none":
        pos_weight = None
    elif args.pos_weight_mode == "manual":
        if args.pos_weight_value is None:
            raise ValueError("--pos-weight-mode=manual requires --pos-weight-value.")
        pos_weight = args.pos_weight_value
    else:  # "auto"
        print("Computing pos_weight from train-set masks (background/foreground pixel ratio)...")
        pos_weight = compute_pos_weight(train_dataset)
    print(f"pos_weight_mode={args.pos_weight_mode} -> pos_weight={pos_weight}")

    history_path = args.output_dir / "training_log.csv"
    best_val_dice = 0.0
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_dice = checkpoint.get("best_metric", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed at epoch {start_epoch} (best_val_dice so far: {best_val_dice:.4f})")
        # pos_weight is recomputed above from the CURRENT train_dataset (so it
        # always reflects --train-pred-mask-root as currently set), not
        # restored from the checkpoint. If that changed since the run being
        # resumed (e.g. --train-pred-mask-root now points at a different
        # pseudo-mask directory), the loss weighting shifts mid-training --
        # warn loudly rather than silently changing the optimization target.
        checkpoint_pos_weight = checkpoint.get("pos_weight")
        if (
            checkpoint_pos_weight is not None
            and pos_weight is not None
            and abs(checkpoint_pos_weight - pos_weight) > 0.05 * max(checkpoint_pos_weight, 1e-6)
        ):
            print(
                f"WARNING: resumed checkpoint was trained with pos_weight={checkpoint_pos_weight:.4f}, "
                f"but the current run computed pos_weight={pos_weight:.4f} (>5% difference) -- "
                "this usually means --train-pred-mask-root points at different masks than the "
                "resumed run used. Loss weighting will change starting this epoch."
            )

    # Wrap in DataParallel AFTER loading any --resume-from checkpoint (which
    # was saved from a plain, unwrapped model -- see save_checkpoint) so
    # load_state_dict above always sees keys without the "module." prefix
    # DataParallel would otherwise require.
    if use_multi_gpu:
        print(f"Using nn.DataParallel across {num_gpus} GPUs (effective batch size = "
              f"{args.batch_size} x {num_gpus} = {args.batch_size * num_gpus} per step).")
        model = nn.DataParallel(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Only (re)write the header when starting fresh -- resuming must append to
    # the existing training_log.csv, not overwrite it (the file was opened
    # with mode "w" unconditionally before this fix, which silently discarded
    # every prior epoch's row on any re-run, resume or not).
    if args.resume_from is None or not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "epoch", "train_loss", "train_dice", "train_iou", "train_consistency",
                "val_loss", "val_dice", "val_iou",
            ])

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, scaler, device, train=True, optimizer=optimizer, pos_weight=pos_weight,
            boundary_ignore_loss=args.boundary_ignore_loss,
            confidence_weighted_loss=args.confidence_weighted_loss,
            confidence_uncertain_weight=args.confidence_uncertain_weight,
            consistency_weight=args.consistency_weight,
            consistency_confidence_threshold=args.consistency_confidence_threshold,
        )
        # Validation always uses the plain, unweighted loss/Dice/IoU against
        # the FULL pseudo-mask -- --boundary-ignore-loss/--confidence-
        # weighted-loss only change what the optimizer is trained against,
        # not how "good" a checkpoint looks. Consistency is also train-only
        # (see use_consistency's `train and ...` gate in run_epoch): applying
        # a stochastic strong-augmentation pass to validation would make
        # val_loss non-deterministic across epochs for no benefit, since
        # early-stopping/best-checkpoint selection here is keyed on val_dice,
        # not val_loss.
        val_loss, val_metrics = run_epoch(model, val_loader, scaler, device, train=False, pos_weight=pos_weight)

        if lr_scheduler is not None:
            previous_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step(val_metrics["dice"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < previous_lr:
                print(f"  --> LR reduced: {previous_lr:.2e} -> {new_lr:.2e} (val_dice plateaued)")

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_metrics["dice"],
                    train_metrics["iou"],
                    train_metrics["consistency"],
                    val_loss,
                    val_metrics["dice"],
                    val_metrics["iou"],
                ]
            )

        consistency_suffix = (
            f" train_consistency={train_metrics['consistency']:.4f}" if args.consistency_weight > 0 else ""
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f}{consistency_suffix}"
        )

        # Update best_val_dice for THIS epoch before saving last_unet.pt, so its
        # best_metric field reflects the true best-so-far (including this epoch)
        # rather than lagging one epoch behind -- otherwise resuming from
        # last_unet.pt reads a stale best_metric and can let a worse later
        # epoch overwrite best_unet.pt.
        #
        # min_delta gates whether this epoch resets the early-stopping
        # patience counter (a tiny, noise-level uptick no longer counts as
        # "progress"), but best_unet.pt/best_val_dice still track the true
        # best-ever value regardless of min_delta -- min_delta changes when
        # to STOP, never which checkpoint is "best" so far.
        is_new_best = val_metrics["dice"] > best_val_dice
        is_meaningful_improvement = val_metrics["dice"] > best_val_dice + args.min_delta
        if is_new_best:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_dice, args.dataset, pos_weight)
            save_checkpoint(args.output_dir / "best_unet.pt", model, optimizer, epoch, best_val_dice, args.dataset, pos_weight)
            print(f"--> Saved new best model with Dice = {best_val_dice:.4f}")
        else:
            save_checkpoint(args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_dice, args.dataset, pos_weight)
        if is_meaningful_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: val_dice did not improve by >= min_delta={args.min_delta} for "
                f"{epochs_without_improvement} epochs (patience={args.early_stop_patience}). "
                f"Best val_dice={best_val_dice:.4f}."
            )
            break


if __name__ == "__main__":
    main()
