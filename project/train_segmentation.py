from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Train U-Net on BTXRD tumor masks")
    parser.set_defaults(dataset="btxrd")
    parser.add_argument(
        "--pipeline-profile",
        choices=["btxrd_best", "btxrd_hybrid"],
        default="btxrd_best",
        help="Provenance label shared with classifier/CAM/SAM commands; U-Net recipe is common to both.",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="BTXRD dataset root")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Immutable derived split manifest. Its assignments are authoritative for BTXRD.",
    )
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
                        help="Stop training if val_positive_dice does not improve for this many consecutive "
                        "epochs. 0 disables early stopping (always run the full --epochs).")
    parser.add_argument(
        "--checkpoint-dice-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Validation positive-Dice values within this absolute tolerance are treated as tied; "
            "the checkpoint with higher normal empty-case specificity wins. This keeps lesion "
            "Dice primary while controlling false-positive masks on normal radiographs."
        ),
    )
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
    parser.add_argument(
        "--pos-weight-mode",
        type=str,
        default="auto-clamped",
        choices=["auto-clamped", "auto-raw", "none", "manual", "auto"],
                        help="How to weight foreground (lesion) pixels in BCE, countering the "
                        "collapse-to-all-background failure found empirically on BTXRD (lesions "
                        "average ~2.6%% of image area, so plain BCE can drive loss low by predicting "
                        "'no lesion anywhere' before learning anything -- observed as val_dice frozen "
                        "at exactly the dataset's normal-image fraction for several epochs). "
                        "'auto-clamped' computes background/foreground ratio and clamps it; "
                        "'auto-raw' uses the raw ratio; deprecated 'auto' aliases auto-clamped; "
                        "'none' disables weighting; 'manual' uses --pos-weight-value.")
    parser.add_argument("--pos-weight-value", type=float, default=None,
                        help="Fixed pos_weight to use when --pos-weight-mode=manual.")
    parser.add_argument("--pos-weight-clamp-min", type=float, default=1.0)
    parser.add_argument("--pos-weight-clamp-max", type=float, default=20.0)
    parser.add_argument(
        "--pos-weight-fixed-reference",
        type=float,
        default=10.0,
        help="Fixed reference recorded beside raw/clamped weights for ablation planning.",
    )
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Path to a checkpoint (e.g. last_unet.pt) to resume training from -- "
                        "restores model/optimizer state and continues epoch numbering and the "
                        "training log instead of starting over. Compatibility checks reject changes "
                        "to the split, resolved training configuration, pseudo-mask manifest, or "
                        "foreground weighting.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_provenance() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT.parent, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except Exception:
        return "unknown", None


def resolved_training_config(args: argparse.Namespace) -> dict[str, object]:
    excluded = {"output_dir", "resume_from", "epochs", "num_workers", "multi_gpu"}
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        if key.startswith("_") or key in excluded:
            continue
        config[key] = str(value.resolve()) if isinstance(value, Path) else value
    config.update({
        "architecture": {"name": "UNet", "in_channels": 3, "out_channels": 1, "base_channels": 64},
        "loss": "0.5 * BCEWithLogits(pos_weight) + 0.5 * soft Dice loss",
        "decision_threshold": 0.5,
        "scheduler": None,
        "gradient_accumulation_steps": 1,
        "gradient_clip_max_norm": None,
    })
    return config


def build_datasets(args: argparse.Namespace):
    if (args.train_pred_mask_root is not None or args.val_pred_mask_root is not None) and args.dataset != "btxrd":
        raise ValueError("Predicted-mask roots are supported only for BTXRD")
    train_dataset = build_segmentation_dataset(
        root=args.data_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
        pred_mask_dir=args.train_pred_mask_root,
        split_manifest=args.split_manifest,
    )
    val_dataset = build_segmentation_dataset(
        root=args.data_root,
        split=args.val_split,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        pred_mask_dir=args.val_pred_mask_root,
        split_manifest=args.split_manifest,
    )
    print(
        f"Loaded {args.dataset}: {len(train_dataset)} train images from {args.train_split}, "
        f"{len(val_dataset)} validation images from {args.val_split}."
    )
    return train_dataset, val_dataset


def scan_mask_statistics(train_dataset, num_workers: int = 0, batch_size: int = 32) -> dict[str, float | int]:
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
    image_count = 0
    empty_mask_count = 0
    for batch_idx, (_, masks, _) in enumerate(loader):
        total_pixels += masks.numel()
        foreground_pixels += float((masks > 0.5).sum().item())
        foreground_by_image = (masks > 0.5).flatten(1).sum(dim=1)
        image_count += int(masks.shape[0])
        empty_mask_count += int((foreground_by_image == 0).sum().item())
        if batch_idx % 10 == 0:
            print(f"  pos_weight scan: batch {batch_idx}/{len(loader)}", flush=True)
    background_pixels = total_pixels - foreground_pixels
    if foreground_pixels <= 0:
        raise ValueError(
            "No foreground (lesion) pixels found in the entire train set -- pos_weight is "
            "undefined. Check that annotations/pseudo-masks are loading correctly before training."
        )
    if image_count <= 0 or total_pixels <= 0:
        raise ValueError("Cannot compute mask statistics from an empty train dataset")
    return {
        "images": image_count,
        "total_pixels": int(total_pixels),
        "foreground_pixels": int(foreground_pixels),
        "background_pixels": int(background_pixels),
        "foreground_ratio": foreground_pixels / total_pixels,
        "empty_mask_count": empty_mask_count,
        "empty_mask_rate": empty_mask_count / image_count,
        "raw_pos_weight": background_pixels / foreground_pixels,
    }


def run_epoch(
    model, loader, scaler, device, train: bool, optimizer=None, pos_weight: float | None = None
) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    positive_target_dice = 0.0
    positive_target_count = 0
    empty_target_correct = 0
    empty_target_count = 0
    samples = 0
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
        batch_size = int(images.shape[0])
        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_iou += iou.item() * batch_size
        samples += batch_size

        predictions = (torch.sigmoid(logits.detach()) >= 0.5).float()
        pred_flat = predictions.flatten(1)
        target_flat = masks.detach().flatten(1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        sample_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        positive_targets = target_flat.sum(dim=1) > 0
        empty_targets = ~positive_targets
        positive_target_dice += float(sample_dice[positive_targets].sum().item())
        positive_target_count += int(positive_targets.sum().item())
        empty_target_correct += int((pred_flat[empty_targets].sum(dim=1) == 0).sum().item())
        empty_target_count += int(empty_targets.sum().item())
        progress.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    if samples == 0:
        return 0.0, {"dice": 0.0, "iou": 0.0, "positive_dice": 0.0, "empty_specificity": 0.0}
    return total_loss / samples, {
        "dice": total_dice / samples,
        "iou": total_iou / samples,
        "positive_dice": positive_target_dice / positive_target_count if positive_target_count else 0.0,
        "empty_specificity": empty_target_correct / empty_target_count if empty_target_count else 0.0,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    dataset: str,
    pos_weight: float | None = None,
    scaler=None,
    run_config: argparse.Namespace | None = None,
    epochs_without_improvement: int = 0,
    best_model_state_dict: dict[str, torch.Tensor] | None = None,
    best_epoch: int | None = None,
    global_step: int = 0,
    best_tiebreak_metric: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap nn.DataParallel before saving: DataParallel prefixes every state_dict
    # key with "module." (model.module.inc.block.0.weight instead of
    # model.inc.block.0.weight), so a checkpoint saved from a DataParallel-wrapped
    # model would fail to load into a plain UNet() with strict=True (used by
    # evaluate_unet.py and --resume-from) -- saving the unwrapped state_dict here
    # keeps checkpoints identical whether or not multi-GPU was used to train.
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    state = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scheduler_name": None,
        "global_step": int(global_step),
        "best_metric": best_metric,
        "best_metric_name": "val_positive_dice",
        "best_tiebreak_metric": best_tiebreak_metric,
        "best_tiebreak_metric_name": "val_normal_empty_case_specificity",
        "checkpoint_selection_rule": (
            "maximize val_positive_dice; within checkpoint_dice_tolerance maximize "
            "val_normal_empty_case_specificity"
        ),
        "dataset": dataset,
        "pos_weight": pos_weight,
        "pos_weight_audit": getattr(run_config, "_pos_weight_audit", None) if run_config else None,
        "epochs_without_improvement": epochs_without_improvement,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if run_config is not None:
        split_manifest = Path(run_config.split_manifest).resolve() if run_config.split_manifest else None
        resolved_config = resolved_training_config(run_config)
        resolved_config_json = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
        git_commit, git_dirty = git_provenance()
        train_pseudo_info = getattr(run_config, "_train_pseudo_manifest_info", None)
        val_pseudo_info = getattr(run_config, "_val_pseudo_manifest_info", None)
        state.update(
            {
                "image_size": run_config.image_size,
                "architecture": {"name": "UNet", "in_channels": 3, "out_channels": 1, "base_channels": 64},
                "train_split": run_config.train_split,
                "val_split": run_config.val_split,
                "train_pred_mask_root": str(run_config.train_pred_mask_root) if run_config.train_pred_mask_root else None,
                "val_pred_mask_root": str(run_config.val_pred_mask_root) if run_config.val_pred_mask_root else None,
                "decision_threshold": 0.5,
                "seed": run_config.seed,
                "use_clahe": run_config.use_clahe,
                "split_manifest": str(split_manifest) if split_manifest else None,
                "split_manifest_sha256": (
                    hashlib.sha256(split_manifest.read_bytes()).hexdigest()
                    if split_manifest is not None and split_manifest.is_file()
                    else None
                ),
                "resolved_config": resolved_config,
                "resolved_config_sha256": hashlib.sha256(resolved_config_json.encode("utf-8")).hexdigest(),
                "train_pseudo_mask_manifest": train_pseudo_info,
                "train_pseudo_mask_manifest_sha256": (
                    train_pseudo_info.get("manifest_sha256") if train_pseudo_info else None
                ),
                "val_pseudo_mask_manifest": val_pseudo_info,
                "val_pseudo_mask_manifest_sha256": (
                    val_pseudo_info.get("manifest_sha256") if val_pseudo_info else None
                ),
                "dataset_identifier": (
                    f"{run_config.dataset}:split_manifest_sha256="
                    f"{hashlib.sha256(split_manifest.read_bytes()).hexdigest()}"
                    if split_manifest is not None and split_manifest.is_file()
                    else f"{run_config.dataset}:unmanifested"
                ),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            }
        )
    if best_model_state_dict is not None:
        state["best_model_state_dict"] = {
            key: value.detach().cpu().clone() for key, value in best_model_state_dict.items()
        }
    if best_epoch is not None:
        state["best_epoch"] = best_epoch
    torch.save(state, path)


HISTORY_FIELDS = [
    "epoch", "train_loss", "train_dice", "train_iou", "train_positive_dice",
    "train_empty_specificity", "val_loss", "val_dice", "val_iou",
    "val_positive_dice", "val_empty_specificity",
]


def ensure_history_schema(path: Path, start_fresh: bool) -> None:
    """Create or upgrade training_log.csv without losing resumed epochs."""
    if start_fresh or not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_FIELDS).writeheader()
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        old_fields = reader.fieldnames or []
        rows = list(reader)
    if old_fields == HISTORY_FIELDS:
        return
    unknown_fields = [field for field in old_fields if field not in HISTORY_FIELDS]
    if unknown_fields:
        raise ValueError(f"Cannot resume from training log with unknown columns: {unknown_fields}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu().clone() for key, value in model_to_save.state_dict().items()}


def is_better_checkpoint(
    candidate_dice: float,
    candidate_normal_specificity: float,
    best_dice: float,
    best_normal_specificity: float,
    dice_tolerance: float,
) -> bool:
    """Lexicographic selection with a clinically relevant normal-case tie-break.

    Positive-target Dice remains the primary endpoint. Only values that are
    statistically indistinguishable at the configured numerical tolerance use
    the normal empty-case rate, preventing an arbitrary earlier checkpoint from
    winning while producing more false-positive normal masks.
    """
    if dice_tolerance < 0:
        raise ValueError("--checkpoint-dice-tolerance must be non-negative")
    if candidate_dice > best_dice + dice_tolerance:
        return True
    return (
        candidate_dice >= best_dice
        and candidate_dice - best_dice <= dice_tolerance
        and candidate_normal_specificity > best_normal_specificity
    )


def restore_best_checkpoint_from_resume(
    output_path: Path,
    resume_checkpoint: dict[str, object],
    best_model_state_dict: dict[str, torch.Tensor],
) -> None:
    """Recreate best_unet.pt when resuming into a fresh output directory."""
    best_state = dict(resume_checkpoint)
    best_state["model_state_dict"] = best_model_state_dict
    best_state["best_model_state_dict"] = best_model_state_dict
    best_state["epoch"] = int(resume_checkpoint.get("best_epoch", resume_checkpoint.get("epoch", 0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)


def validate_resume_compatibility(checkpoint: dict[str, object], args: argparse.Namespace) -> None:
    expected_architecture = {"name": "UNet", "in_channels": 3, "out_channels": 1, "base_channels": 64}
    if checkpoint.get("architecture") not in (None, expected_architecture):
        raise ValueError(f"Resume checkpoint architecture mismatch: {checkpoint.get('architecture')!r}")
    expected = {
        "dataset": args.dataset,
        "image_size": args.image_size,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "seed": args.seed,
        "use_clahe": args.use_clahe,
    }
    for key, value in expected.items():
        if key in checkpoint and checkpoint[key] != value:
            raise ValueError(
                f"Resume checkpoint {key}={checkpoint[key]!r} does not match current run {value!r}"
            )
    current_config = resolved_training_config(args)
    current_hash = hashlib.sha256(
        json.dumps(current_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checkpoint_config_hash = checkpoint.get("resolved_config_sha256")
    if checkpoint_config_hash is None:
        raise ValueError("Resume checkpoint has no resolved_config_sha256")
    if checkpoint_config_hash != current_hash:
        raise ValueError("Resume checkpoint resolved training configuration does not match current run")
    current_train_pseudo = getattr(args, "_train_pseudo_manifest_info", None)
    expected_pseudo_hash = current_train_pseudo.get("manifest_sha256") if current_train_pseudo else None
    if checkpoint.get("train_pseudo_mask_manifest_sha256") != expected_pseudo_hash:
        raise ValueError("Resume checkpoint pseudo-mask manifest hash does not match current training masks")
    if args.split_manifest:
        manifest_path = args.split_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Current split manifest does not exist: {manifest_path}")
        expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        checkpoint_hash = checkpoint.get("split_manifest_sha256")
        if checkpoint_hash is None:
            raise ValueError("Resume checkpoint has no split_manifest_sha256; refusing an unproven split resume")
        if checkpoint_hash != expected_hash:
            raise ValueError("Resume checkpoint split manifest hash does not match current manifest")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_dataset, val_dataset = build_datasets(args)
    args._train_pseudo_manifest_info = getattr(train_dataset, "pseudo_manifest_info", None)
    args._val_pseudo_manifest_info = getattr(val_dataset, "pseudo_manifest_info", None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    num_gpus = torch.cuda.device_count()
    use_multi_gpu = args.multi_gpu and device.type == "cuda" and num_gpus > 1
    if args.multi_gpu and not use_multi_gpu:
        print(f"--multi-gpu requested but only {num_gpus} CUDA device(s) visible; running single-device.")

    print("Scanning deterministic, non-augmented train masks for foreground ratio and empty-mask rate...")
    mask_audit_dataset = build_segmentation_dataset(
        root=args.data_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        pred_mask_dir=args.train_pred_mask_root,
        split_manifest=args.split_manifest,
    )
    mask_stats = scan_mask_statistics(mask_audit_dataset)
    raw_pos_weight = float(mask_stats["raw_pos_weight"])
    if args.pos_weight_clamp_min <= 0 or args.pos_weight_clamp_max < args.pos_weight_clamp_min:
        raise ValueError("Invalid pos_weight clamp bounds")
    clamped_pos_weight = float(np.clip(raw_pos_weight, args.pos_weight_clamp_min, args.pos_weight_clamp_max))
    resolved_mode = "auto-clamped" if args.pos_weight_mode == "auto" else args.pos_weight_mode
    if resolved_mode == "none":
        pos_weight = None
    elif resolved_mode == "manual":
        if args.pos_weight_value is None:
            raise ValueError("--pos-weight-mode=manual requires --pos-weight-value.")
        if args.pos_weight_value <= 0:
            raise ValueError("--pos-weight-value must be positive")
        pos_weight = args.pos_weight_value
    elif resolved_mode == "auto-raw":
        pos_weight = raw_pos_weight
    else:
        pos_weight = clamped_pos_weight
    pos_weight_audit = {
        **mask_stats,
        "requested_mode": args.pos_weight_mode,
        "resolved_mode": resolved_mode,
        "candidate_weights": {
            "raw": raw_pos_weight,
            "clamped": clamped_pos_weight,
            "fixed_reference": args.pos_weight_fixed_reference,
            "none": None,
        },
        "selected_pos_weight": pos_weight,
        "effective_pos_weight": pos_weight,
        "clamp_min": args.pos_weight_clamp_min,
        "clamp_max": args.pos_weight_clamp_max,
        "ablation_note": (
            "Compare separate otherwise-identical runs using auto-raw, auto-clamped, "
            "and manual --pos-weight-value equal to fixed_reference."
        ),
    }
    args._pos_weight_audit = pos_weight_audit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pos_weight_audit.json").write_text(
        json.dumps(pos_weight_audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(pos_weight_audit, indent=2))

    history_path = args.output_dir / "training_log.csv"
    best_val_positive_dice = -1.0
    best_val_normal_specificity = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    global_step = 0

    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        validate_resume_compatibility(checkpoint, args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("best_metric_name") == "val_positive_dice":
            best_val_positive_dice = checkpoint.get("best_metric", -1.0)
            best_val_normal_specificity = float(checkpoint.get("best_tiebreak_metric", -1.0) or -1.0)
            epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
            best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
            raw_best_state = checkpoint.get("best_model_state_dict")
            if not isinstance(raw_best_state, dict):
                legacy_best_path = args.resume_from.parent / "best_unet.pt"
                if legacy_best_path.is_file():
                    legacy_best = torch.load(legacy_best_path, map_location="cpu")
                    raw_best_state = legacy_best.get("model_state_dict")
            if not isinstance(raw_best_state, dict):
                raise RuntimeError(
                    "Resume checkpoint does not contain best_model_state_dict and no adjacent "
                    "best_unet.pt is available; refusing to continue with an unknown best model."
                )
            best_model_state_dict = {
                key: value.detach().cpu().clone() for key, value in raw_best_state.items()
            }
            restore_best_checkpoint_from_resume(
                args.output_dir / "best_unet.pt", checkpoint, best_model_state_dict
            )
        else:
            print(
                "WARNING: resumed checkpoint used the legacy all-image val_dice criterion; "
                "resetting best/early-stop state for the new val_positive_dice criterion."
            )
            raise RuntimeError(
                "Cannot safely resume a checkpoint selected with a different metric; "
                "start a fresh run or provide a compatible val_positive_dice checkpoint."
            )
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        print(
            f"Resumed at epoch {start_epoch} "
            f"(best_val_positive_dice={best_val_positive_dice:.4f}, "
            f"best_val_normal_specificity={best_val_normal_specificity:.4f})"
        )
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
            raise ValueError(
                f"Resume pos_weight mismatch: checkpoint={checkpoint_pos_weight:.4f}, "
                f"current={pos_weight:.4f}. Refusing to change the optimization objective mid-run."
            )

    # Wrap in DataParallel AFTER loading any --resume-from checkpoint (which
    # was saved from a plain, unwrapped model -- see save_checkpoint) so
    # load_state_dict above always sees keys without the "module." prefix
    # DataParallel would otherwise require.
    if use_multi_gpu:
        print(
            f"Using nn.DataParallel across {num_gpus} GPUs "
            f"(global batch size = {args.batch_size}; split across devices per step)."
        )
        model = nn.DataParallel(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Only (re)write the header when starting fresh -- resuming must append to
    # the existing training_log.csv, not overwrite it (the file was opened
    # with mode "w" unconditionally before this fix, which silently discarded
    # every prior epoch's row on any re-run, resume or not).
    ensure_history_schema(history_path, start_fresh=args.resume_from is None)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, scaler, device, train=True, optimizer=optimizer, pos_weight=pos_weight
        )
        val_loss, val_metrics = run_epoch(model, val_loader, scaler, device, train=False, pos_weight=pos_weight)
        global_step += len(train_loader)

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writerow({
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
            })

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_positive_dice={val_metrics['positive_dice']:.4f} "
            f"val_normal_empty_case_specificity={val_metrics['empty_specificity']:.4f}"
        )

        # Update best metric for THIS epoch before saving last_unet.pt, so its
        # best_metric field reflects the true best-so-far (including this epoch)
        # rather than lagging one epoch behind -- otherwise resuming from
        # last_unet.pt reads a stale best_metric and can let a worse later
        # epoch overwrite best_unet.pt.
        if is_better_checkpoint(
            val_metrics["positive_dice"],
            val_metrics["empty_specificity"],
            best_val_positive_dice,
            best_val_normal_specificity,
            args.checkpoint_dice_tolerance,
        ):
            best_val_positive_dice = val_metrics["positive_dice"]
            best_val_normal_specificity = val_metrics["empty_specificity"]
            best_epoch = epoch
            best_model_state_dict = clone_model_state(model)
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_positive_dice,
                args.dataset, pos_weight, scaler, args, epochs_without_improvement,
                best_model_state_dict, best_epoch, global_step, best_val_normal_specificity,
            )
            save_checkpoint(
                args.output_dir / "best_unet.pt", model, optimizer, epoch, best_val_positive_dice,
                args.dataset, pos_weight, scaler, args, epochs_without_improvement,
                best_model_state_dict, best_epoch, global_step, best_val_normal_specificity,
            )
            print(
                "--> Saved new best model: "
                f"positive-target Dice={best_val_positive_dice:.4f}, "
                f"normal empty-case specificity={best_val_normal_specificity:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if best_model_state_dict is None:
                raise RuntimeError("No best model state is available while saving last_unet.pt")
            save_checkpoint(
                args.output_dir / "last_unet.pt", model, optimizer, epoch, best_val_positive_dice,
                args.dataset, pos_weight, scaler, args, epochs_without_improvement,
                best_model_state_dict, best_epoch, global_step, best_val_normal_specificity,
            )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: val_positive_dice did not improve for {epochs_without_improvement} epochs "
                f"(patience={args.early_stop_patience}). Best val_positive_dice={best_val_positive_dice:.4f}."
            )
            break


if __name__ == "__main__":
    main()
