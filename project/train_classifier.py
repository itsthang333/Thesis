from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
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
from models.sam_segment_contrastive import (
    SamSegmentMapStore,
    sam_segment_contrastive_loss,
)
from progress import should_disable_tqdm
from models.puzzle_cam import puzzle_alpha as puzzle_alpha_schedule, puzzle_cam_consistency_loss
from models.teacher_student import EMATeacher, attention_distillation_loss
from pseudo.generate_layercam import generate_fused_cam
from pseudo.visualization import save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand/tumor classifier for LayerCAM feature extraction")
    parser.set_defaults(dataset="btxrd")
    parser.add_argument(
        "--pipeline-profile",
        type=str,
        default="default",
        choices=["default", "btxrd_best", "btxrd_hybrid"],
        help=(
            "btxrd_best freezes the classifier setup paired with the selected "
            "WSSS pipeline: 10-class tumor_type CE at 320 px, batch 4, up to 30 epochs, "
            "and PuzzleCAM/attention losses disabled. btxrd_hybrid keeps that "
            "same 320px/tumor_type recipe but trains for up to 30 epochs with early "
            "stopping and PuzzleCAM + Teacher-Student attention distillation "
            "enabled, combining btxrd_best's downstream CAM/SAM/selection "
            "recipe (higher oracle_dice) with the other pipeline's classifier "
            "training recipe (higher val_f1)."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True, help="BTXRD dataset root")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Immutable derived split manifest. Required for BTXRD group-aware evaluation; "
        "when supplied, its train/val/test assignments are authoritative.",
    )
    parser.add_argument("--target-columns", type=str, default=None,
                        help="BTXRD target column; defaults to 'tumor_type'")
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
    parser.add_argument(
        "--sam-segment-map-root",
        type=Path,
        default=None,
        help=(
            "Root of a validated train-only S2C Segment-Everything artifact containing "
            "region_map_manifest.csv and region_maps/. Required when "
            "--sam-segment-contrastive-weight is positive."
        ),
    )
    parser.add_argument(
        "--sam-segment-map-manifest-sha256",
        type=str,
        default="",
        help="Frozen SHA-256 of region_map_manifest.csv; required for S2C SSC.",
    )
    parser.add_argument(
        "--sam-segment-contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for S2C SAM-Segment Contrasting loss. 0 disables SSC.",
    )
    parser.add_argument(
        "--sam-segment-temperature",
        type=float,
        default=1.0,
        help=(
            "Cosine-logit multiplier for S2C SSC. The official S2C default is 1.0."
        ),
    )
    args = parser.parse_args()
    args._explicit_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return args


def apply_pipeline_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.pipeline_profile == "default":
        return args
    if args.dataset != "btxrd":
        raise ValueError(f"--pipeline-profile {args.pipeline_profile} requires BTXRD")

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
    require_or_set(
        "--early-stop-patience",
        "early_stop_patience",
        profile.classifier_early_stop_patience,
    )
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
        require_or_set("--teacher-warmup-epochs", "teacher_warmup_epochs", profile.teacher_warmup_epochs)
        require_or_set("--teacher-ema-decay", "teacher_ema_decay", profile.teacher_ema_decay)
        require_or_set("--teacher-cam-percentile", "teacher_cam_percentile", profile.teacher_cam_percentile)
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
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def confusion_counts(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
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
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
    }


def multiclass_confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[num_classes, num_classes] confusion matrix, rows=true class, cols=predicted class."""
    preds = logits.argmax(dim=1)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1).tolist(), preds.view(-1).tolist()):
        matrix[t, p] += 1
    return matrix


def metrics_from_multiclass_confusion(matrix: torch.Tensor) -> dict[str, float]:
    num_classes = matrix.shape[0]
    total = matrix.sum().item()
    accuracy = matrix.diag().sum().item() / max(1, total)

    precisions, recalls, f1s, supports = [], [], [], []
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
        supports.append(matrix[c, :].sum().item())

    return {
        "acc": accuracy,
        "precision": sum(precisions) / num_classes,
        "recall": sum(recalls) / num_classes,
        "f1": sum(f1s) / num_classes,
        "weighted_f1": (
            sum(score * support for score, support in zip(f1s, supports)) / max(1, sum(supports))
        ),
    }


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    train: bool,
    sam_segment_store: SamSegmentMapStore | None = None,
    sam_segment_contrastive_weight: float = 0.0,
    sam_segment_temperature: float = 1.0,
) -> tuple[float, dict[str, float], dict[str, int], dict[str, float]]:
    total_classification_loss = 0.0
    total_ssc_loss = 0.0
    total_optimization_loss = 0.0
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batches = 0
    model.train(train)

    progress = tqdm(
        loader,
        desc="train" if train else "val",
        leave=False,
        disable=should_disable_tqdm(),
    )
    for images, targets, image_ids in progress:
        images = images.to(device)
        targets = targets.to(device)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                use_ssc = (
                    train
                    and sam_segment_store is not None
                    and sam_segment_contrastive_weight > 0
                )
                if use_ssc:
                    logits, features = model(images, return_features=True)
                else:
                    logits = model(images)
                classification_loss = criterion(logits, targets)
            if use_ssc:
                region_maps = sam_segment_store.load_batch(image_ids, device=device)
                with torch.cuda.amp.autocast(enabled=False):
                    ssc_loss = sam_segment_contrastive_loss(
                        features,
                        region_maps,
                        temperature=sam_segment_temperature,
                    )
                loss = (
                    classification_loss
                    + sam_segment_contrastive_weight * ssc_loss
                )
            else:
                ssc_loss = classification_loss.new_zeros(())
                loss = classification_loss

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
        total_classification_loss += classification_loss.item()
        total_ssc_loss += ssc_loss.item()
        total_optimization_loss += loss.item()
        batches += 1
        batch_metrics = metrics_from_confusion(batch_counts)
        progress.set_postfix(
            cls=classification_loss.item(),
            ssc=ssc_loss.item(),
            loss=loss.item(),
            f1=batch_metrics["f1"],
        )

    if batches == 0:
        diagnostics = {
            "classification_loss": 0.0,
            "sam_segment_contrastive_loss": 0.0,
            "optimization_loss": 0.0,
        }
        return 0.0, metrics_from_confusion(counts), counts, diagnostics
    diagnostics = {
        "classification_loss": total_classification_loss / batches,
        "sam_segment_contrastive_loss": total_ssc_loss / batches,
        "optimization_loss": total_optimization_loss / batches,
    }
    return (
        diagnostics["classification_loss"],
        metrics_from_confusion(counts),
        counts,
        diagnostics,
    )


def run_epoch_multiclass(
    model, loader, criterion, optimizer, scaler, device, num_classes: int, train: bool,
    puzzle_alpha: float = 0.0,
    teacher=None,
    attention_alpha: float = 0.0,
    teacher_percentile: float = 96.0,
) -> tuple[float, dict[str, float], torch.Tensor, dict[str, float]]:
    total_cls_loss = 0.0
    total_optimization_loss = 0.0
    total_puzzle_cls_loss = 0.0
    total_reconstruction_loss = 0.0
    total_attention_loss = 0.0
    total_teacher_confidence = 0.0
    total_valid_teacher_fraction = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    samples = 0
    model.train(train)

    progress = tqdm(
        loader,
        desc="train" if train else "val",
        leave=False,
        disable=should_disable_tqdm(),
    )
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
                cls_loss = criterion(logits, targets)
                loss = cls_loss

                re_loss_value = 0.0
                p_cls_loss_value = 0.0
                if train and puzzle_alpha > 0:
                    _, _, re_loss, p_cls_loss = puzzle_cam_consistency_loss(model, images, targets)
                    loss = cls_loss + p_cls_loss + puzzle_alpha * re_loss
                    re_loss_value = re_loss.item()
                    p_cls_loss_value = p_cls_loss.item()

                att_loss_value = 0.0
                teacher_conf_value = 0.0
                valid_teacher_fraction_value = 0.0
                if need_features:
                    att_loss, teacher_conf, valid_teacher_fraction = attention_distillation_loss(
                        teacher, model, student_features, images, targets, percentile=teacher_percentile
                    )
                    loss = loss + attention_alpha * att_loss
                    teacher_conf_value = teacher_conf.item()
                    att_loss_value = att_loss.item()
                    valid_teacher_fraction_value = valid_teacher_fraction.item()

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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        batch_confusion = multiclass_confusion_matrix(logits.detach().cpu(), targets.detach().cpu(), num_classes)
        confusion += batch_confusion
        batch_size = int(images.shape[0])
        total_cls_loss += cls_loss.item() * batch_size
        total_optimization_loss += loss.item() * batch_size
        total_puzzle_cls_loss += p_cls_loss_value * batch_size
        total_reconstruction_loss += re_loss_value * batch_size
        total_attention_loss += att_loss_value * batch_size
        total_teacher_confidence += teacher_conf_value * batch_size
        total_valid_teacher_fraction += valid_teacher_fraction_value * batch_size
        samples += batch_size
        batch_metrics = metrics_from_multiclass_confusion(batch_confusion)
        progress.set_postfix(loss=cls_loss.item(), macro_f1=batch_metrics["f1"],
                              re_loss=re_loss_value, p_cls_loss=p_cls_loss_value, att_loss=att_loss_value,
                              teacher_conf=teacher_conf_value)

        if train and teacher is not None:
            teacher.update(model)

    diagnostics = {
        "classification_loss": total_cls_loss / samples if samples else 0.0,
        "optimization_loss": total_optimization_loss / samples if samples else 0.0,
        "puzzle_classification_loss": total_puzzle_cls_loss / samples if samples else 0.0,
        "reconstruction_loss": total_reconstruction_loss / samples if samples else 0.0,
        "attention_loss": total_attention_loss / samples if samples else 0.0,
        "teacher_confidence": total_teacher_confidence / samples if samples else 0.0,
        "valid_teacher_fraction": total_valid_teacher_fraction / samples if samples else 0.0,
    }
    return diagnostics["classification_loss"], metrics_from_multiclass_confusion(confusion), confusion, diagnostics


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
    split_manifest: Path | None = None,
    image_size: int | None = None,
    seed: int | None = None,
    sam_segment_contrastive: dict[str, object] | None = None,
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
            "num_classes": num_classes if num_classes is not None else len(target_columns),
            "normalization": normalization,
            "train_augment": bool(train_augment),
            "pipeline_profile": pipeline_profile,
            "image_size": image_size,
            "seed": seed,
            "split_manifest": str(split_manifest.resolve()) if split_manifest else None,
            "split_manifest_sha256": (
                hashlib.sha256(split_manifest.resolve().read_bytes()).hexdigest()
                if split_manifest is not None and split_manifest.is_file()
                else None
            ),
            "sam_segment_contrastive": sam_segment_contrastive,
        },
        path,
    )


def select_cam_preview_indices(val_dataset, count: int) -> list[int]:
    indices: list[int] = []
    for index in range(len(val_dataset)):
        _, target, _ = val_dataset[index]
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


def classifier_epoch_budget_audit(
    records: list[dict[str, float | int]],
    requested_epochs: int,
    *,
    stopped_early: bool = False,
    early_stop_patience: int = 0,
) -> dict[str, object]:
    """Diagnose whether the audited validation curve supports the epoch budget."""
    if not records:
        raise ValueError("Cannot audit an empty classifier training history")
    best = max(records, key=lambda row: float(row["val_f1"]))
    tail = records[-min(3, len(records)):]
    if len(tail) >= 2:
        x = np.asarray([float(row["epoch"]) for row in tail], dtype=np.float64)
        y = np.asarray([float(row["val_f1"]) for row in tail], dtype=np.float64)
        tail_slope = float(np.polyfit(x, y, 1)[0])
    else:
        tail_slope = None
    last_epoch = int(records[-1]["epoch"])
    best_epoch = int(best["epoch"])
    epochs_since_best = last_epoch - best_epoch
    best_at_boundary = best_epoch == last_epoch and last_epoch >= requested_epochs
    valid_early_stop = (
        stopped_early
        and early_stop_patience > 0
        and epochs_since_best >= early_stop_patience
    )
    if best_at_boundary:
        assessment = "budget_boundary_best_requires_longer_ablation"
        assessment_basis = "best validation F1 occurred at the requested epoch boundary"
    elif valid_early_stop:
        assessment = "plateau_or_decline_observed"
        assessment_basis = (
            f"early stopping fired after {epochs_since_best} epochs without a new best "
            f"(patience={early_stop_patience})"
        )
    elif epochs_since_best >= 2 and tail_slope is not None and tail_slope <= 0:
        assessment = "plateau_or_decline_observed"
        assessment_basis = "best epoch precedes a non-positive trailing validation-F1 trend"
    else:
        assessment = "inconclusive"
        assessment_basis = "neither a valid early stop nor a non-positive trailing trend was observed"
    return {
        "metric": "audited-split validation F1",
        "requested_epochs": int(requested_epochs),
        "completed_epochs": last_epoch,
        "best_epoch": best_epoch,
        "best_val_f1": float(best["val_f1"]),
        "final_val_f1": float(records[-1]["val_f1"]),
        "epochs_since_best": epochs_since_best,
        "stopped_early": bool(stopped_early),
        "early_stop_patience": int(early_stop_patience),
        "valid_early_stop": valid_early_stop,
        "best_at_budget_boundary": best_at_boundary,
        "tail_window_epochs": [int(row["epoch"]) for row in tail],
        "tail_val_f1_slope_per_epoch": tail_slope,
        "assessment": assessment,
        "assessment_basis": assessment_basis,
        "decision_rule": (
            "If the best audited-split validation F1 is at the requested budget boundary, "
            "run a longer otherwise-identical validation-only budget ablation before test freeze."
        ),
    }


def main() -> None:
    args = apply_pipeline_profile(parse_args())
    print("Resolved classifier configuration:")
    print(f"  Profile: {args.pipeline_profile}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.early_stop_patience}")
    print("  Early stopping minimum delta: 0.0")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Puzzle alpha: {args.puzzle_alpha_max}")
    print(f"  Attention alpha: {args.attention_alpha_max}")
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

    normalization = "radimagenet" if args.radimagenet_checkpoint else "imagenet"
    if not np.isfinite(args.sam_segment_contrastive_weight) or args.sam_segment_contrastive_weight < 0:
        raise ValueError("--sam-segment-contrastive-weight must be finite and non-negative")
    if not np.isfinite(args.sam_segment_temperature) or args.sam_segment_temperature <= 0:
        raise ValueError("--sam-segment-temperature must be finite and positive")
    if args.sam_segment_contrastive_weight > 0:
        if args.sam_segment_map_root is None:
            raise ValueError(
                "--sam-segment-map-root is required when SSC weight is positive"
            )
        if not args.sam_segment_map_manifest_sha256:
            raise ValueError(
                "--sam-segment-map-manifest-sha256 is required when SSC is enabled"
            )
        if args.augment or args.random_erasing:
            raise ValueError(
                "S2C SSC requires augmentation and random erasing to remain off so "
                "the precomputed SAM regions stay spatially aligned"
            )
        if args.preprocessing_mode != "none":
            raise ValueError(
                "S2C SSC requires preprocessing-mode=none to match the region maps"
            )
        if args.image_size != 320:
            raise ValueError(
                "The frozen S2C region maps are aligned to classifier image-size 320"
            )
    elif args.sam_segment_map_root is not None:
        raise ValueError(
            "--sam-segment-map-root was supplied but SSC weight is zero"
        )

    train_dataset = build_classification_dataset(
        root=args.data_root,
        split=args.train_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        augment=args.augment,
        preprocessing_mode=args.preprocessing_mode,
        normalization=normalization,
        split_manifest=args.split_manifest,
    )
    if args.random_erasing:
        train_dataset.image_transform.transforms.append(
            tv_transforms.RandomErasing(
                p=0.50, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0.0
            )
        )
    val_dataset = build_classification_dataset(
        root=args.data_root,
        split=args.val_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        augment=False,
        preprocessing_mode=args.preprocessing_mode,
        normalization=normalization,
        split_manifest=args.split_manifest,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    sam_segment_store = None
    sam_segment_config: dict[str, object] | None = None
    if args.sam_segment_contrastive_weight > 0:
        sam_segment_store = SamSegmentMapStore(
            args.sam_segment_map_root,
            train_dataset.samples,
            expected_manifest_sha256=args.sam_segment_map_manifest_sha256,
        )
        sam_segment_config = {
            "method": "S2C SAM-Segment Contrasting",
            "weight": float(args.sam_segment_contrastive_weight),
            "temperature": float(args.sam_segment_temperature),
            "map_root": str(args.sam_segment_map_root.resolve()),
            "manifest_sha256": sam_segment_store.manifest_sha256,
            "map_shape": list(sam_segment_store.map_shape or ()),
            "train_maps": len(sam_segment_store),
            "prototype_gradient": "detached",
            "ignore_region_id": 0,
        }
        print(f"Validated S2C region maps: {json.dumps(sam_segment_config)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_multiclass = target_columns == ["tumor_type"]
    if is_multiclass and args.sam_segment_contrastive_weight > 0:
        raise ValueError(
            "The controlled S2C ablation is defined for the binary tumor classifier only"
        )
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
    checkpoint_task = "single-label" if is_multiclass else "multi-label"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "pipeline_profile": args.pipeline_profile,
                "dataset": args.dataset,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "split_manifest": str(args.split_manifest) if args.split_manifest else None,
                "split_manifest_sha256": (
                    hashlib.sha256(args.split_manifest.resolve().read_bytes()).hexdigest()
                    if args.split_manifest is not None and args.split_manifest.is_file()
                    else None
                ),
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
                "sam_segment_contrastive": sam_segment_config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_multiclass:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "train_weighted_f1",
                "train_puzzle_cls_loss", "train_reconstruction_loss", "train_attention_loss",
                "puzzle_alpha", "attention_alpha", "teacher_confidence", "valid_teacher_fraction",
                "total_optimization_loss",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "val_weighted_f1",
            ])
        else:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
                "train_sam_segment_contrastive_loss", "train_optimization_loss",
                "train_tp", "train_fp", "train_fn", "train_tn",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
                "val_tp", "val_fp", "val_fn", "val_tn",
            ])

    teacher = None
    epoch_budget_records: list[dict[str, float | int]] = []
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        if is_multiclass:
            current_puzzle_alpha = (
                puzzle_alpha_schedule(epoch, args.epochs, alpha_max=args.puzzle_alpha_max)
                if args.puzzle_alpha_max > 0 else 0.0
            )

            current_attention_alpha = 0.0
            if args.attention_alpha_max > 0 and epoch > args.teacher_warmup_epochs:
                if teacher is None:
                    teacher = EMATeacher(model, decay=args.teacher_ema_decay)
                    print(f"  --> Teacher initialized at epoch {epoch} (post-warmup snapshot)")
                epochs_since_warmup = epoch - args.teacher_warmup_epochs
                remaining_epochs = args.epochs - args.teacher_warmup_epochs
                current_attention_alpha = puzzle_alpha_schedule(
                    epochs_since_warmup, remaining_epochs, alpha_max=args.attention_alpha_max
                )

            train_loss, train_metrics, _train_confusion, train_diagnostics = run_epoch_multiclass(
                model, train_loader, criterion, optimizer, scaler, device, num_classes, train=True,
                puzzle_alpha=current_puzzle_alpha,
                teacher=teacher, attention_alpha=current_attention_alpha,
                teacher_percentile=args.teacher_cam_percentile,
            )
            val_loss, val_metrics, val_confusion, _val_diagnostics = run_epoch_multiclass(
                model, val_loader, criterion, optimizer, scaler, device, num_classes, train=False
            )
        else:
            train_loss, train_metrics, train_counts, train_diagnostics = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                train=True,
                sam_segment_store=sam_segment_store,
                sam_segment_contrastive_weight=args.sam_segment_contrastive_weight,
                sam_segment_temperature=args.sam_segment_temperature,
            )
            val_loss, val_metrics, val_counts, _val_diagnostics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                scaler,
                device,
                train=False,
            )

        epoch_budget_records.append({
            "epoch": epoch,
            "val_f1": float(val_metrics["f1"]),
            "val_weighted_f1": float(val_metrics.get("weighted_f1", val_metrics["f1"])),
            "val_loss": float(val_loss),
        })

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_multiclass:
                writer.writerow([
                    epoch, train_loss, train_metrics["acc"], train_metrics["precision"],
                    train_metrics["recall"], train_metrics["f1"], train_metrics["weighted_f1"],
                    train_diagnostics["puzzle_classification_loss"],
                    train_diagnostics["reconstruction_loss"],
                    train_diagnostics["attention_loss"],
                    current_puzzle_alpha,
                    current_attention_alpha,
                    train_diagnostics["teacher_confidence"],
                    train_diagnostics["valid_teacher_fraction"],
                    train_diagnostics["optimization_loss"],
                    val_loss, val_metrics["acc"], val_metrics["precision"],
                    val_metrics["recall"], val_metrics["f1"], val_metrics["weighted_f1"],
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
                        train_diagnostics["sam_segment_contrastive_loss"],
                        train_diagnostics["optimization_loss"],
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
        ssc_suffix = ""
        if not is_multiclass and args.sam_segment_contrastive_weight > 0:
            ssc_suffix = (
                f" ssc={train_diagnostics['sam_segment_contrastive_loss']:.4f}"
                f" optimization_loss={train_diagnostics['optimization_loss']:.4f}"
            )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_{'macro_f1' if is_multiclass else 'f1'}={train_metrics['f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_{'macro_f1' if is_multiclass else 'f1'}={val_metrics['f1']:.4f}"
            f"{puzzle_suffix}{teacher_suffix}{ssc_suffix}"
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

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best_classifier.pt", model, optimizer, epoch, best_val_f1,
                target_columns, args.dataset, task=checkpoint_task, num_classes=num_classes,
                normalization=normalization,
                train_augment=args.augment,
                pipeline_profile=args.pipeline_profile,
                split_manifest=args.split_manifest,
                image_size=args.image_size,
                seed=args.seed,
                sam_segment_contrastive=sam_segment_config,
            )
            print(f"  --> Saved new best checkpoint (val_f1={best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            args.output_dir / "last_classifier.pt", model, optimizer, epoch, best_val_f1,
            target_columns, args.dataset, task=checkpoint_task, num_classes=num_classes,
            normalization=normalization,
            train_augment=args.augment,
            pipeline_profile=args.pipeline_profile,
            split_manifest=args.split_manifest,
            image_size=args.image_size,
            seed=args.seed,
            sam_segment_contrastive=sam_segment_config,
        )

        if epoch in cam_epochs and cam_preview_indices:
            save_cam_preview(model, val_dataset, cam_preview_indices, epoch, cam_output_dir, device, is_multiclass=is_multiclass, normalization=normalization)
            print(f"  --> Saved CAM preview for epoch {epoch} to {cam_output_dir}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            stopped_early = True
            print(
                f"Early stopping: val_f1 did not improve for {epochs_without_improvement} epochs "
                f"(patience={args.early_stop_patience}). Best val_f1={best_val_f1:.4f}."
            )
            break

    budget_audit = classifier_epoch_budget_audit(
        epoch_budget_records,
        args.epochs,
        stopped_early=stopped_early,
        early_stop_patience=args.early_stop_patience,
    )
    budget_audit.update({
        "split": args.val_split,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "split_manifest_sha256": (
            hashlib.sha256(args.split_manifest.resolve().read_bytes()).hexdigest()
            if args.split_manifest is not None and args.split_manifest.is_file()
            else None
        ),
        "pipeline_profile": args.pipeline_profile,
    })
    budget_audit_path = args.output_dir / "classifier_epoch_budget_audit.json"
    budget_audit_path.write_text(json.dumps(budget_audit, indent=2) + "\n", encoding="utf-8")
    print(f"Classifier epoch-budget audit: {json.dumps(budget_audit, indent=2)}")


if __name__ == "__main__":
    main()
