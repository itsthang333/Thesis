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

from config import ClassifierConfig, DATASET_TARGET_COLUMNS, DEFAULT_DATASET, SUPPORTED_DATASETS
from datasets.factory import build_classification_dataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from pseudo.generate_layercam import generate_fused_cam
from pseudo.visualization import save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand/tumor classifier for LayerCAM feature extraction")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=SUPPORTED_DATASETS,
                        help="Which dataset to train on: ramh1200 (hand-only) or btxrd (tumor-vs-normal)")
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "classifier")
    parser.add_argument("--no-pretrained", action="store_true")
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
    return parser.parse_args()


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
                loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    target_columns: list[str],
    dataset: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "target_columns": target_columns,
            "task": "multi-label",
            "dataset": dataset,
        },
        path,
    )


def select_cam_preview_indices(val_dataset, count: int) -> list[int]:
    """Pick a fixed set of positive-class validation samples for CAM snapshots.

    Fixed indices (not re-sampled per epoch) so the same images are compared
    across epochs, isolating changes in CAM quality from changes in which
    image is shown.
    """
    indices: list[int] = []
    for index in range(len(val_dataset)):
        _, target, _ = val_dataset[index]
        if float(target[0]) == 1.0:
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
                class_weights = torch.sigmoid(logits)[0].detach().cpu().numpy()

            fused_cam, _, _ = generate_fused_cam(
                layercam, image_tensor, class_weights=class_weights, confidence_threshold=0.0,
            )
            image_pil = tensor_to_pil(image_tensor[0].detach().cpu())
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
    args = parse_args()
    seed_everything(args.seed)

    default_columns = DATASET_TARGET_COLUMNS[args.dataset]
    if args.target_columns is None:
        target_columns = list(default_columns)
    else:
        target_columns = [column.strip() for column in args.target_columns.split(",") if column.strip()]
    if tuple(target_columns) != default_columns:
        print(
            f"[WARNING] '{args.dataset}' expects target-columns={list(default_columns)}. "
            "Only change this if you intentionally prepared extra labels for this dataset."
        )

    train_dataset = build_classification_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.train_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        preprocessing_mode=args.preprocessing_mode,
    )
    val_dataset = build_classification_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.val_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        preprocessing_mode=args.preprocessing_mode,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121AnatomyClassifier(num_classes=len(target_columns), pretrained=not args.no_pretrained).to(device)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
            "train_tp", "train_fp", "train_fn", "train_tn",
            "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
            "val_tp", "val_fp", "val_fn", "val_tn",
        ])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, train_counts = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_metrics, val_counts = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
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

        positive_label = target_columns[0]
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} | val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        print(
            f"  val confusion matrix (positive={positive_label}): "
            f"TP={val_counts['tp']} FP={val_counts['fp']} FN={val_counts['fn']} TN={val_counts['tn']} "
            f"| precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}"
        )

        save_checkpoint(args.output_dir / "last_classifier.pt", model, optimizer, epoch, best_val_f1, target_columns, args.dataset)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best_classifier.pt", model, optimizer, epoch, best_val_f1, target_columns, args.dataset)
            print(f"  --> Saved new best checkpoint (val_f1={best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        if epoch in cam_epochs and cam_preview_indices:
            save_cam_preview(model, val_dataset, cam_preview_indices, epoch, cam_output_dir, device)
            print(f"  --> Saved CAM preview for epoch {epoch} to {cam_output_dir}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping: val_f1 did not improve for {epochs_without_improvement} epochs "
                f"(patience={args.early_stop_patience}). Best val_f1={best_val_f1:.4f}."
            )
            break


if __name__ == "__main__":
    main()
