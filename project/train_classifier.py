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

from config import ClassifierConfig, DEFAULT_ANATOMY_COLUMNS
from datasets.ramh1200 import RAMH1200ClassificationDataset
from models.classifier import DenseNet121AnatomyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAM-H1200 hand classifier for LayerCAM feature extraction")
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--target-columns", type=str, default=",".join(DEFAULT_ANATOMY_COLUMNS))
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
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    targets = targets.float()
    if preds.ndim == 1:
        preds = preds.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"acc": accuracy, "precision": precision, "recall": recall, "f1": f1}


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    aggregate = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
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

        metrics = classification_metrics(logits.detach(), targets.detach())
        total_loss += loss.item()
        for key in aggregate:
            aggregate[key] += metrics[key]
        batches += 1
        progress.set_postfix(loss=loss.item(), f1=metrics["f1"])

    if batches == 0:
        return 0.0, aggregate
    return total_loss / batches, {key: value / batches for key, value in aggregate.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    target_columns: list[str],
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
            "dataset": "RAM-H1200",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    target_columns = [column.strip() for column in args.target_columns.split(",") if column.strip()]
    if target_columns != ["hand"]:
        print(
            "[WARNING] RAM-H1200 segmentation images are hand radiographs. "
            "Use '--target-columns hand' unless you intentionally prepared extra RAM-H1200 labels."
        )

    train_dataset = RAMH1200ClassificationDataset(
        root=args.ram_root,
        split=args.train_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
    )
    val_dataset = RAMH1200ClassificationDataset(
        root=args.ram_root,
        split=args.val_split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121AnatomyClassifier(num_classes=len(target_columns), pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = args.output_dir / "training_log.csv"
    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

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
                    val_loss,
                    val_metrics["acc"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                    val_metrics["f1"],
                ]
            )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output_dir / "best_classifier.pt", model, optimizer, epoch, best_val_loss, target_columns)
        # Save last checkpoint AFTER updating best_val_loss so the stored metric is current.
        save_checkpoint(args.output_dir / "last_classifier.pt", model, optimizer, epoch, best_val_loss, target_columns)


if __name__ == "__main__":
    main()
