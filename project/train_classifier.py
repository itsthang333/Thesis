from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    BTXRD_BEST_PIPELINE,
    BTXRD_HYBRID_PIPELINE,
    ClassifierConfig,
    DATASET_TARGET_COLUMNS,
    DEFAULT_DATASET,
    SUPPORTED_DATASETS,
)
from datasets.factory import build_classification_dataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from models.puzzle_cam import puzzle_alpha as puzzle_alpha_schedule, puzzle_cam_consistency_loss
from models.teacher_student import EMATeacher, attention_distillation_loss
from pseudo.generate_layercam import generate_fused_cam
from pseudo.visualization import save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand/tumor classifier for LayerCAM feature extraction")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=SUPPORTED_DATASETS,
                        help="Which dataset to train on: ramh1200 (hand-only) or btxrd (tumor-vs-normal)")
    parser.add_argument(
        "--pipeline-profile",
        type=str,
        default="default",
        choices=["default", "btxrd_best", "btxrd_hybrid"],
        help=(
            "btxrd_best freezes the classifier setup paired with the selected "
            "WSSS pipeline: 10-class tumor_type CE at 320 px, batch 4, 6 epochs, "
            "and PuzzleCAM/attention losses disabled. btxrd_hybrid keeps that "
            "same 320px/tumor_type recipe but trains for 25 epochs with early "
            "stopping and PuzzleCAM + Teacher-Student attention distillation "
            "enabled, combining btxrd_best's downstream CAM/SAM/selection "
            "recipe (higher oracle_dice) with the other pipeline's classifier "
            "training recipe (higher val_f1)."
        ),
    )
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1",
                        help="Dataset root (RAM-H1200 root or BTXRD root, depending on --dataset)")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--target-columns", type=str, default=None,
                        help="Defaults to 'hand' for ramh1200 or 'tumor' for btxrd")
    parser.add_argument("--image-size", type=int, default=ClassifierConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=ClassifierConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ClassifierConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ClassifierConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=ClassifierConfig.epochs)
    parser.add_argument("--seed", type=int, default=ClassifierConfig.seed)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true",
                        help="Apply the dataset's training-only horizontal flip augmentation; validation remains deterministic.")
    parser.add_argument("--random-erasing", action="store_true",
                        help="After standard normalization, randomly erase small image patches during training "
                             "to reduce shortcuts from radiographic letters/markers; validation is unchanged.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "classifier")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--radimagenet-checkpoint", type=Path, default=None,
                        help="Path to a RadImageNet DenseNet121.pt checkpoint to use as the "
                        "backbone's pretrained weights instead of ImageNet. Overrides "
                        "--no-pretrained when set.")
    parser.add_argument("--puzzle-alpha-max", type=float, default=0.0,
                        help="Max weight for PuzzleCAM's consistency loss (see models/puzzle_cam.py). "
                        "0 (default) disables it entirely -- pure CrossEntropy, unchanged behavior. "
                        "Only applies to the single-label ('tumor_type') task. Linearly ramps from 0 "
                        "to this value over the first half of --epochs, per the paper's warmup schedule.")
    parser.add_argument("--attention-alpha-max", type=float, default=0.0,
                        help="Max weight for the Teacher-Student attention distillation loss (see "
                        "models/teacher_student.py). 0 (default) disables it entirely. Only applies "
                        "to the single-label ('tumor_type') task. Requires --teacher-warmup-epochs. "
                        "Ramps linearly from 0 to this value over the epochs following warmup (same "
                        "schedule shape as --puzzle-alpha-max), since this loss's per-pixel BCE "
                        "gradient was found to be ~100-150x larger in magnitude than the per-sample "
                        "CrossEntropy loss on real DenseNet121 -- an unscaled alpha=1.0 would let "
                        "attention distillation dominate and destabilize classification learning "
                        "from the moment the teacher activates. A small max (e.g. 0.01-0.05) plus "
                        "warmup keeps CE in control while still letting the signal grow in.")
    parser.add_argument("--teacher-warmup-epochs", type=int, default=3,
                        help="Number of initial epochs to train the student on CE (+ PuzzleCAM if "
                        "enabled) alone, with the teacher frozen at its starting checkpoint, before "
                        "EMA updates and the attention loss begin. Needed because at epoch 0 the "
                        "teacher is an exact copy of the student -- there's nothing yet for it to "
                        "teach that the student doesn't already know.")
    parser.add_argument("--teacher-ema-decay", type=float, default=0.999,
                        help="EMA decay rate for the teacher's weights once warmup ends.")
    parser.add_argument("--teacher-cam-percentile", type=float, default=96.0,
                        help="Percentile threshold applied to the teacher's LayerCAM before Torch "
                        "morphology + Gaussian blur, when building the student's soft attention "
                        "target. This project's own measurements found LayerCAM's thresholded area "
                        "pinned at ~15%% of the image (percentile 85) regardless of content, while "
                        "actual lesions average ~2.6%% -- a much higher percentile here (keeping only "
                        "the CAM's brightest ~4%%) forces the target to start from the CAM's most "
                        "confident region rather than refining the full diffuse blob.")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--preprocessing-mode", type=str, default="none",
                        choices=["none", "clahe", "contrast", "gamma", "foreground_crop"],
                        help="Optional X-ray preprocessing before resize/normalization")
    parser.add_argument("--save-cam-epochs", type=str, default="",
                        help="Comma-separated epoch numbers (e.g. '2,4,5') to save LayerCAM "
                        "overlays for a fixed set of validation images, for a visual sanity "
                        "check of CAM quality across training. Empty disables this.")
    parser.add_argument("--cam-preview-count", type=int, default=4,
                        help="Number of fixed positive-class validation images to snapshot "
                        "CAM for when --save-cam-epochs is set")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training if val_f1 does not improve for this many consecutive "
                        "epochs. 0 disables early stopping (always run the full --epochs).")
    args = parser.parse_args()
    args._explicit_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return args


def apply_pipeline_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Freeze the classifier recipe paired with ``btxrd_best``/``btxrd_hybrid``.

    This profile contains no segmentation supervision.  It only selects the
    image-level classification head and training hyperparameters that feed
    LayerCAM.  Explicit changes to recipe-critical options are rejected rather
    than silently producing a checkpoint that no longer matches the frozen
    downstream pipeline.
    """
    if args.pipeline_profile == "default":
        return args
    if args.dataset != "btxrd":
        raise ValueError(f"--pipeline-profile {args.pipeline_profile} requires --dataset btxrd")

    profile = BTXRD_HYBRID_PIPELINE if args.pipeline_profile == "btxrd_hybrid" else BTXRD_BEST_PIPELINE
    name = args.pipeline_profile
    explicit = getattr(args, "_explicit_options", set())

    def require_or_set(option: str, attribute: str, expected: object) -> None:
        if option in explicit and getattr(args, attribute) != expected:
            raise ValueError(
                f"--pipeline-profile {name} fixes {option}={expected!r}; "
                f"received {getattr(args, attribute)!r}"
            )
        setattr(args, attribute, expected)

    require_or_set("--target-columns", "target_columns", ",".join(profile.target_columns))
    require_or_set("--train-split", "train_split", "train")
    require_or_set("--val-split", "val_split", "val")
    require_or_set("--image-size", "image_size", profile.classifier_image_size)
    require_or_set("--batch-size", "batch_size", profile.classifier_batch_size)
    require_or_set("--epochs", "epochs", profile.classifier_epochs)
    require_or_set("--lr", "lr", profile.classifier_lr)
    require_or_set("--weight-decay", "weight_decay", profile.classifier_weight_decay)
    require_or_set("--seed", "seed", profile.classifier_seed)
    require_or_set("--puzzle-alpha-max", "puzzle_alpha_max", profile.classifier_puzzle_alpha_max)
    require_or_set("--attention-alpha-max", "attention_alpha_max", profile.classifier_attention_alpha_max)
    require_or_set("--preprocessing-mode", "preprocessing_mode", "none")
    if "--augment" in explicit and args.augment:
        raise ValueError(f"--pipeline-profile {name} fixes training augmentation off")
    if "--random-erasing" in explicit and args.random_erasing:
        raise ValueError(f"--pipeline-profile {name} fixes random erasing off")
    if "--no-pretrained" in explicit and args.no_pretrained:
        raise ValueError(f"--pipeline-profile {name} fixes ImageNet-pretrained initialization")
    if "--use-clahe" in explicit and args.use_clahe:
        raise ValueError(f"--pipeline-profile {name} fixes CLAHE off")
    if "--radimagenet-checkpoint" in explicit and args.radimagenet_checkpoint is not None:
        raise ValueError(f"--pipeline-profile {name} fixes ImageNet normalization/pretraining")
    if name == "btxrd_hybrid":
        # btxrd_hybrid deliberately trains longer with early stopping enabled
        # (see config.py's BtxrdHybridPipelineConfig) instead of btxrd_best's
        # fixed 6-epoch / no-early-stop recipe.
        require_or_set("--early-stop-patience", "early_stop_patience", profile.classifier_early_stop_patience)
        require_or_set("--teacher-warmup-epochs", "teacher_warmup_epochs", profile.teacher_warmup_epochs)
        require_or_set("--teacher-ema-decay", "teacher_ema_decay", profile.teacher_ema_decay)
        require_or_set("--teacher-cam-percentile", "teacher_cam_percentile", profile.teacher_cam_percentile)
    elif "--early-stop-patience" in explicit and args.early_stop_patience != 0:
        raise ValueError(f"--pipeline-profile {name} fixes early stopping off")
    args.augment = False
    args.random_erasing = False
    if "--output-dir" not in explicit:
        args.output_dir = ROOT / "outputs" / f"btxrd_classifier_{name}"
    return args


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def confusion_counts(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    """Raw TP/FP/FN/TN counts for the positive (index 0) class, accumulated over a full epoch.

    Computed from per-sample counts (not averaged per-batch metrics) so precision/recall/F1
    over an epoch reflect the true epoch-level confusion matrix instead of a batch-size-biased
    average of noisy small-batch ratios.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    targets = targets.float()
    if preds.ndim == 1:
        preds = preds.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    preds = preds[:, 0].reshape(-1)
    targets = targets[:, 0].reshape(-1)
    return {
        "tp": int(((preds == 1) & (targets == 1)).sum().item()),
        "fp": int(((preds == 1) & (targets == 0)).sum().item()),
        "fn": int(((preds == 0) & (targets == 1)).sum().item()),
        "tn": int(((preds == 0) & (targets == 0)).sum().item()),
    }


def metrics_from_confusion(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"acc": accuracy, "precision": precision, "recall": recall, "f1": f1}


def multiclass_confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[num_classes, num_classes] confusion matrix, rows=true class, cols=predicted class."""
    preds = logits.argmax(dim=1)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1).tolist(), preds.view(-1).tolist()):
        matrix[t, p] += 1
    return matrix


def metrics_from_multiclass_confusion(matrix: torch.Tensor) -> dict[str, float]:
    """Accuracy + macro-averaged precision/recall/F1 across all classes.

    Macro averaging (not micro/weighted) matters here specifically because
    BTXRD's tumor_type classes are extremely imbalanced (44-1879 images per
    class) -- a micro/weighted average would be dominated by the majority
    classes (normal, osteochondroma) and could look good even if the rarest
    classes (osteofibroma, other_mt, synovial_osteochondroma) are never
    predicted correctly at all.
    """
    num_classes = matrix.shape[0]
    total = matrix.sum().item()
    accuracy = matrix.diag().sum().item() / max(1, total)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = matrix[c, c].item()
        fp = matrix[:, c].sum().item() - tp
        fn = matrix[c, :].sum().item() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": accuracy,
        "precision": sum(precisions) / num_classes,
        "recall": sum(recalls) / num_classes,
        "f1": sum(f1s) / num_classes,
    }


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> tuple[float, dict[str, float], dict[str, int]]:
    total_loss = 0.0
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batches = 0
    model.train(train)

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, targets, _ in progress:
        images = images.to(device)
        targets = targets.to(device)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                # See run_epoch_multiclass's matching comment: a rare
                # pathological input can still produce a finite-but-extreme
                # logit even with forward_features forced to fp32; clamp so
                # it can't distort the displayed loss average or BCE's
                # internal sigmoid/log terms.
                logits = torch.clamp(logits, -30.0, 30.0)
                loss = criterion(logits, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] Skipping batch with non-finite loss (pathological input)")
                continue

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        batch_counts = confusion_counts(logits.detach(), targets.detach())
        for key in counts:
            counts[key] += batch_counts[key]
        total_loss += loss.item()
        batches += 1
        batch_metrics = metrics_from_confusion(batch_counts)
        progress.set_postfix(loss=loss.item(), f1=batch_metrics["f1"])

    if batches == 0:
        return 0.0, metrics_from_confusion(counts), counts
    return total_loss / batches, metrics_from_confusion(counts), counts


def run_epoch_multiclass(
    model, loader, criterion, optimizer, scaler, device, num_classes: int, train: bool,
    puzzle_alpha: float = 0.0,
    teacher=None,
    attention_alpha: float = 0.0,
    teacher_percentile: float = 96.0,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Single-label multi-class variant of run_epoch (targets are class indices, not multi-hot).

    puzzle_alpha > 0 adds PuzzleCAM's consistency loss (Jo & Yu, ICIP 2021,
    adapted for single-label CrossEntropy -- see models/puzzle_cam.py): an
    extra forward pass on the image split into a 2x2 tile grid, comparing
    the CAM reconstructed from the 4 tiles against the full-image CAM for
    each sample's target class, and penalizing their L1 difference. This
    directly targets the diffuse/non-discriminative CAM problem this
    project's own debug_cam_by_tumor_type.py measured (cam_area_ratio pinned
    at ~15% of the image regardless of content) by forcing the model's
    spatial evidence to agree whether it sees the whole image or 4 separate
    quadrants -- something a model that just "looks at everything" can't do
    consistently, but one that's actually localizing the lesion can. Only
    applied during training (train=True); val loss stays pure CrossEntropy
    for a clean, comparable val_f1/early-stopping signal.
    """
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    batches = 0
    model.train(train)

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, targets, _ in progress:
        images = images.to(device)
        targets = targets.to(device)  # [B], long class indices -- do NOT unsqueeze

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                need_features = train and teacher is not None and attention_alpha > 0
                if need_features:
                    logits, student_features = model(images, return_features=True)
                else:
                    logits = model(images)
                # forward_features now forces fp32 through the backbone, so
                # logits are always finite -- but a pathological input (see
                # the skip-batch warning below) can still produce a finite
                # but extreme logit (hundreds+), which blows up val_loss's
                # displayed average without ever triggering the isnan/isinf
                # skip. Clamp to a range that can't meaningfully change which
                # class wins (correct-class logits for real predictions stay
                # under ~1 early in training, per this project's own debug
                # runs) while keeping CrossEntropyLoss's internal exp() finite.
                logits = torch.clamp(logits, -30.0, 30.0)
                # cls_loss is logged separately from the combined `loss` used
                # for backward(), so train_loss/val_loss in the CSV stay
                # directly comparable to runs without PuzzleCAM (pure
                # CrossEntropy) -- `loss` (with the puzzle term folded in)
                # only drives the actual gradient step.
                cls_loss = criterion(logits, targets)
                loss = cls_loss

                re_loss_value = 0.0
                p_cls_loss_value = 0.0
                if train and puzzle_alpha > 0:
                    # Full PuzzleCAM objective: L_cls + L_p-cls + alpha*L_re
                    # (Jo & Yu, ICIP 2021, Eq. 7). L_p-cls has weight 1.0 (not
                    # scaled by alpha/warmup) since it's an ordinary
                    # classification loss on the tiled reconstruction, not a
                    # consistency regularizer -- only L_re gets the warmup
                    # treatment, matching the paper.
                    _, _, re_loss, p_cls_loss = puzzle_cam_consistency_loss(model, images, targets)
                    loss = cls_loss + p_cls_loss + puzzle_alpha * re_loss
                    re_loss_value = re_loss.item()
                    p_cls_loss_value = p_cls_loss.item()

                att_loss_value = 0.0
                teacher_conf_value = 0.0
                if need_features:
                    # L_attention: student's own CAM vs. the teacher's refined
                    # soft target (frozen EMA teacher -> LayerCAM -> percentile
                    # threshold -> Torch morphology -> Gaussian blur), weighted
                    # by the teacher's own confidence in the ground-truth
                    # class (squared) to guard against confirmation bias --
                    # an unsure teacher shouldn't teach as if it were certain.
                    # Unlike PuzzleCAM's L_re (CAM(full) ~= CAM(tiles), which
                    # a model can satisfy by being uniformly diffuse in both
                    # views -- confirmed empirically: PuzzleCAM raised val_f1
                    # sharply but left cam_area_ratio pinned at ~0.15,
                    # unchanged from the plain baseline), this directly
                    # supervises the student's CAM shape against a sharpened
                    # target that's already been pushed toward the lesion's
                    # actual (~2.6% of image area) footprint by the
                    # percentile+morphology pipeline -- "CAM ~= lesion-shaped-
                    # region", not just "CAM ~= CAM".
                    att_loss, teacher_conf = attention_distillation_loss(
                        teacher, model, student_features, images, targets, percentile=teacher_percentile
                    )
                    loss = loss + attention_alpha * att_loss
                    teacher_conf_value = teacher_conf.item()
                    att_loss_value = att_loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] Skipping batch with non-finite loss (a rare pathological "
                      f"input -- found empirically: a converted-from-grayscale X-ray with all "
                      f"3 channels identical pushes this RadImageNet-pretrained backbone's "
                      f"internal features to ~2.6e5, overflowing fp16's 65504 max mid-forward-"
                      f"pass, before logits/loss are ever computed -- clamping logits afterward "
                      f"can't fix this since inf/nan has already propagated by then)")
                continue

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Unscale before clipping so the norm is computed on true
                # (not loss-scaled) gradients -- without this, a handful of
                # early batches with unusually large gradients (e.g. a
                # freshly-initialized classifier head paired with a
                # differently-scaled pretrained backbone) can push weights
                # into a regime where logits blow up to the tens of
                # thousands and never recover, since nothing bounds the
                # update step size.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        batch_confusion = multiclass_confusion_matrix(logits.detach().cpu(), targets.detach().cpu(), num_classes)
        confusion += batch_confusion
        total_loss += cls_loss.item()
        batches += 1
        batch_metrics = metrics_from_multiclass_confusion(batch_confusion)
        progress.set_postfix(loss=cls_loss.item(), macro_f1=batch_metrics["f1"],
                              re_loss=re_loss_value, p_cls_loss=p_cls_loss_value, att_loss=att_loss_value,
                              teacher_conf=teacher_conf_value)

        if train and teacher is not None:
            teacher.update(model)

    if batches == 0:
        return 0.0, metrics_from_multiclass_confusion(confusion), confusion
    return total_loss / batches, metrics_from_multiclass_confusion(confusion), confusion


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    target_columns: list[str],
    dataset: str,
    task: str = "multi-label",
    num_classes: int | None = None,
    normalization: str = "imagenet",
    train_augment: bool = False,
    pipeline_profile: str = "default",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "target_columns": target_columns,
            "task": task,
            "dataset": dataset,
            # Consumers (generate_pseudo_masks.py/inference.py/
            # visualize_pipeline.py) must read num_classes from here, not
            # infer it from len(target_columns) -- that breaks for
            # target_columns=["tumor_type"] (1 element) mapping to a
            # 10-class model. Falls back to len(target_columns) for old
            # checkpoints saved before this field existed.
            "num_classes": num_classes if num_classes is not None else len(target_columns),
            # Which input preprocessing this checkpoint's backbone expects --
            # "imagenet" (RGB, ImageNet mean/std) or "radimagenet" (BGR,
            # (x-127.5)*2/255, no mean/std). Must match at inference/CAM time.
            "normalization": normalization,
            "train_augment": bool(train_augment),
            "pipeline_profile": pipeline_profile,
        },
        path,
    )


def select_cam_preview_indices(val_dataset, count: int) -> list[int]:
    """Pick a fixed set of positive-class (or, for tumor_type, any non-normal-class)
    validation samples for CAM snapshots.

    Fixed indices (not re-sampled per epoch) so the same images are compared
    across epochs, isolating changes in CAM quality from changes in which
    image is shown.
    """
    indices: list[int] = []
    for index in range(len(val_dataset)):
        _, target, _ = val_dataset[index]
        # tumor_type: target is a scalar long class index (0=normal); binary
        # tumor: target is a 1-element float multi-hot vector.
        is_positive = int(target.item()) != 0 if target.ndim == 0 else float(target[0]) == 1.0
        if is_positive:
            indices.append(index)
        if len(indices) >= count:
            break
    return indices


def save_cam_preview(
    model: nn.Module,
    val_dataset,
    indices: list[int],
    epoch: int,
    output_dir: Path,
    device: torch.device,
    is_multiclass: bool = False,
    normalization: str = "imagenet",
) -> None:
    """Save a LayerCAM overlay for each fixed preview image at this epoch.

    Lets you flip through cam_epoch{N}_sample{i}.png across epochs to judge
    whether CAM localization is actually improving — in WSSS this matters more
    than a few points of classifier F1, since CAM quality directly drives the
    downstream SAM prompts.
    """
    was_training = model.training
    model.eval()
    layercam = LayerCAM(model, device=device)
    try:
        for sample_index in indices:
            image_tensor, target, image_name = val_dataset[sample_index]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image_tensor)
                if is_multiclass:
                    # Softmax (mutually exclusive classes), not sigmoid --
                    # class_weights here just needs to pick out the single
                    # predicted class for generate_fused_cam's confidence
                    # gate, same as classifier_class_weights() in
                    # generate_pseudo_masks.py does for a "single-label" task.
                    class_weights = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                else:
                    class_weights = torch.sigmoid(logits)[0].detach().cpu().numpy()

            fused_cam, _, _ = generate_fused_cam(
                layercam, image_tensor, class_weights=class_weights, confidence_threshold=0.0,
            )
            image_pil = tensor_to_pil(image_tensor[0].detach().cpu(), normalization=normalization)
            stem = Path(str(image_name)).stem
            save_overlay(
                image_pil,
                fused_cam,
                output_dir / f"cam_epoch{epoch:03d}_{stem}.png",
            )
    finally:
        layercam.close()
        model.train(was_training)


def main() -> None:
    args = apply_pipeline_profile(parse_args())
    seed_everything(args.seed)

    default_columns = DATASET_TARGET_COLUMNS[args.dataset]
    if args.target_columns is None:
        target_columns = list(default_columns)
    else:
        target_columns = [column.strip() for column in args.target_columns.split(",") if column.strip()]
    is_canonical_btxrd_type = args.dataset == "btxrd" and target_columns == ["tumor_type"]
    if tuple(target_columns) != default_columns and not is_canonical_btxrd_type:
        print(
            f"[WARNING] '{args.dataset}' expects target-columns={list(default_columns)}. "
            "Only change this if you intentionally prepared extra labels for this dataset."
        )

    # RadImageNet's BatchNorm statistics were empirically confirmed (see
    # datasets/common.py's RadImageNetNormalize docstring) to match the
    # official BGR+[-1,1] preprocessing, not ImageNet mean/std -- tied
    # automatically to --radimagenet-checkpoint so the two can't drift out of sync.
    normalization = "radimagenet" if args.radimagenet_checkpoint else "imagenet"

    train_dataset = build_classification_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.train_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        augment=args.augment,
        preprocessing_mode=args.preprocessing_mode,
        normalization=normalization,
    )
    if args.random_erasing:
        # RandomErasing operates on the tensor after resize/normalization and
        # is attached only to the training dataset.  It uses no segmentation
        # annotations and therefore remains a weak image-level augmentation.
        train_dataset.image_transform.transforms.append(
            tv_transforms.RandomErasing(
                p=0.50, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0.0
            )
        )
    val_dataset = build_classification_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.val_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        augment=False,
        preprocessing_mode=args.preprocessing_mode,
        normalization=normalization,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target_columns=["tumor_type"] is single-label multi-class (10 mutually
    # exclusive BTXRD classes: normal + 9 tumor types), not the usual
    # multi-label setup where num_classes == len(target_columns). Detect it
    # explicitly rather than inferring from len(target_columns), since that
    # would otherwise (wrongly) build a 1-output model for a 10-class problem.
    is_multiclass = target_columns == ["tumor_type"]
    if is_multiclass:
        from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
        num_classes = len(TUMOR_TYPE_CLASS_NAMES)
    else:
        num_classes = len(target_columns)

    model = DenseNet121AnatomyClassifier(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        radimagenet_checkpoint=args.radimagenet_checkpoint,
    ).to(device)

    if is_multiclass:
        # Inverse-frequency class weights: BTXRD's tumor_type classes range
        # from 44 to 1879 images (>40x imbalance) -- without weighting, the
        # loss is dominated by the majority classes and the rarest tumor
        # types (osteofibroma, other_mt, synovial_osteochondroma) are likely
        # to never be predicted at all.
        class_counts = torch.zeros(num_classes)
        for sample in train_dataset.samples:
            class_counts[int(sample["tumor_type"])] += 1
        class_weights_tensor = (class_counts.sum() / (num_classes * class_counts.clamp(min=1))).to(device)
        print(f"tumor_type class counts (train split): {class_counts.tolist()}")
        print(f"tumor_type class weights (inverse frequency): {class_weights_tensor.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    cam_epochs = {int(value.strip()) for value in args.save_cam_epochs.split(",") if value.strip()}
    cam_preview_indices: list[int] = []
    cam_output_dir = args.output_dir / "cam_preview"
    if cam_epochs:
        cam_preview_indices = select_cam_preview_indices(val_dataset, args.cam_preview_count)
        cam_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"CAM preview: {len(cam_preview_indices)} fixed validation images, epochs {sorted(cam_epochs)}")

    history_path = args.output_dir / "training_log.csv"
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    # "single-label" must match exactly what generate_pseudo_masks.py/
    # inference.py/visualize_pipeline.py's classifier_class_weights() checks
    # for (it applies softmax instead of sigmoid for this task string).
    checkpoint_task = "single-label" if is_multiclass else "multi-label"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "pipeline_profile": args.pipeline_profile,
                "dataset": args.dataset,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "target_columns": target_columns,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augment": args.augment,
                "random_erasing": args.random_erasing,
                "puzzle_alpha_max": args.puzzle_alpha_max,
                "attention_alpha_max": args.attention_alpha_max,
                "preprocessing_mode": args.preprocessing_mode,
                "normalization": normalization,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_multiclass:
            # No fixed TP/FP/FN/TN here -- with 10 classes the full confusion
            # matrix (10x10) is printed to stdout each epoch instead of being
            # flattened into the CSV; macro precision/recall/f1 summarize it.
            writer.writerow([
                "epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
            ])
        else:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
                "train_tp", "train_fp", "train_fn", "train_tn",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
                "val_tp", "val_fp", "val_fn", "val_tn",
            ])

    teacher = None  # created once warmup ends; stays None (attention loss disabled) until then

    for epoch in range(1, args.epochs + 1):
        if is_multiclass:
            current_puzzle_alpha = (
                puzzle_alpha_schedule(epoch, args.epochs, alpha_max=args.puzzle_alpha_max)
                if args.puzzle_alpha_max > 0 else 0.0
            )

            current_attention_alpha = 0.0
            if args.attention_alpha_max > 0 and epoch > args.teacher_warmup_epochs:
                if teacher is None:
                    # Snapshot the student as it is right after warmup -- this
                    # is the "phá đối xứng" (symmetry-breaking) point: the
                    # student has now trained on CE (+PuzzleCAM) alone for
                    # teacher_warmup_epochs, so it's no longer identical to a
                    # freshly-loaded checkpoint, giving EMA updates something
                    # real to average over from here on.
                    teacher = EMATeacher(model, decay=args.teacher_ema_decay)
                    print(f"  --> Teacher initialized at epoch {epoch} (post-warmup snapshot)")
                # Ramp over the epochs remaining after warmup, same linear
                # shape as puzzle_alpha_schedule -- needed because this loss's
                # per-pixel BCE gradient was measured at ~100-150x a
                # per-sample CrossEntropy step's magnitude on real
                # DenseNet121; an unscaled alpha from the moment the teacher
                # activates would let it dominate and destabilize
                # classification learning before CE has a chance to adapt.
                epochs_since_warmup = epoch - args.teacher_warmup_epochs
                remaining_epochs = args.epochs - args.teacher_warmup_epochs
                current_attention_alpha = puzzle_alpha_schedule(
                    epochs_since_warmup, remaining_epochs, alpha_max=args.attention_alpha_max
                )

            train_loss, train_metrics, _train_confusion = run_epoch_multiclass(
                model, train_loader, criterion, optimizer, scaler, device, num_classes, train=True,
                puzzle_alpha=current_puzzle_alpha,
                teacher=teacher, attention_alpha=current_attention_alpha,
                teacher_percentile=args.teacher_cam_percentile,
            )
            val_loss, val_metrics, val_confusion = run_epoch_multiclass(
                model, val_loader, criterion, optimizer, scaler, device, num_classes, train=False
            )
        else:
            train_loss, train_metrics, train_counts = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
            val_loss, val_metrics, val_counts = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_multiclass:
                writer.writerow([
                    epoch, train_loss, train_metrics["acc"], train_metrics["precision"],
                    train_metrics["recall"], train_metrics["f1"],
                    val_loss, val_metrics["acc"], val_metrics["precision"],
                    val_metrics["recall"], val_metrics["f1"],
                ])
            else:
                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        train_metrics["acc"],
                        train_metrics["precision"],
                        train_metrics["recall"],
                        train_metrics["f1"],
                        train_counts["tp"],
                        train_counts["fp"],
                        train_counts["fn"],
                        train_counts["tn"],
                        val_loss,
                        val_metrics["acc"],
                        val_metrics["precision"],
                        val_metrics["recall"],
                        val_metrics["f1"],
                        val_counts["tp"],
                        val_counts["fp"],
                        val_counts["fn"],
                        val_counts["tn"],
                    ]
                )

        puzzle_suffix = f" puzzle_alpha={current_puzzle_alpha:.3f}" if is_multiclass and args.puzzle_alpha_max > 0 else ""
        teacher_suffix = ""
        if is_multiclass and args.attention_alpha_max > 0:
            teacher_suffix = (
                f" attention_alpha={current_attention_alpha:.4f}"
                f" teacher={'active' if teacher is not None else 'warmup'}"
            )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_{'macro_f1' if is_multiclass else 'f1'}={train_metrics['f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_{'macro_f1' if is_multiclass else 'f1'}={val_metrics['f1']:.4f}{puzzle_suffix}{teacher_suffix}"
        )
        if is_multiclass:
            from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
            print(f"  val macro precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}")
            print("  val confusion matrix (rows=true, cols=predicted):")
            print("   " + " ".join(f"{name[:6]:>7}" for name in TUMOR_TYPE_CLASS_NAMES))
            for i, name in enumerate(TUMOR_TYPE_CLASS_NAMES):
                print(f"  {name[:10]:<10}" + " ".join(f"{int(val_confusion[i, j]):>7}" for j in range(num_classes)))
        else:
            positive_label = target_columns[0]
            print(
                f"  val confusion matrix (positive={positive_label}): "
                f"TP={val_counts['tp']} FP={val_counts['fp']} FN={val_counts['fn']} TN={val_counts['tn']} "
                f"| precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}"
            )

        save_checkpoint(
            args.output_dir / "last_classifier.pt", model, optimizer, epoch, best_val_f1,
            target_columns, args.dataset, task=checkpoint_task, num_classes=num_classes,
            normalization=normalization,
            train_augment=args.augment,
            pipeline_profile=args.pipeline_profile,
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best_classifier.pt", model, optimizer, epoch, best_val_f1,
                target_columns, args.dataset, task=checkpoint_task, num_classes=num_classes,
                normalization=normalization,
                train_augment=args.augment,
                pipeline_profile=args.pipeline_profile,
            )
            print(f"  --> Saved new best checkpoint (val_f1={best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        if epoch in cam_epochs and cam_preview_indices:
            save_cam_preview(model, val_dataset, cam_preview_indices, epoch, cam_output_dir, device, is_multiclass=is_multiclass, normalization=normalization)
            print(f"  --> Saved CAM preview for epoch {epoch} to {cam_output_dir}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: val_f1 did not improve for {epochs_without_improvement} epochs "
                f"(patience={args.early_stop_patience}). Best val_f1={best_val_f1:.4f}."
            )
            break


if __name__ == "__main__":
    main()
