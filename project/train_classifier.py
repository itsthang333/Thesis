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

from config import ClassifierConfig
from datasets.fracatlas import FracAtlasClassificationDataset, build_train_val_indices
from models.classifier import DenseNet121AnatomyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet121 anatomy classifier on FracAtlas")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "FracAtlas")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--target-columns", type=str, default="hand,leg,hip,shoulder")
    parser.add_argument("--image-size", type=int, default=ClassifierConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=ClassifierConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ClassifierConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ClassifierConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=ClassifierConfig.epochs)
    parser.add_argument("--val-fraction", type=float, default=ClassifierConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=ClassifierConfig.seed)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "classifier")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument(
        "--task",
        type=str,
        default="single-label",
        choices=["single-label", "multi-label"],
        help="single-label filters to rows with exactly one anatomy label and trains with CrossEntropyLoss",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def _single_label_class_index(row: dict[str, str], target_columns: list[str]) -> int:
    positive_indices = [
        i for i, column in enumerate(target_columns)
        if float(row.get(column, 0.0) or 0.0) > 0.0
    ]
    if len(positive_indices) != 1:
        raise ValueError("Expected exactly one positive anatomy label.")
    return positive_indices[0]


def build_stratified_train_val_indices(
    dataset: FracAtlasClassificationDataset,
    target_columns: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    import random

    by_class: dict[int, list[int]] = {i: [] for i in range(len(target_columns))}
    for index, sample in enumerate(dataset.samples):
        row = sample["row"]
        assert isinstance(row, dict)
        by_class[_single_label_class_index(row, target_columns)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        val_size = max(1, int(len(indices) * val_fraction)) if len(indices) > 1 else 0
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
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


def single_label_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    targets = targets.long().view(-1)
    accuracy = (preds == targets).float().mean().item()

    precisions = []
    recalls = []
    f1s = []
    for class_index in range(num_classes):
        pred_pos = preds == class_index
        true_pos = targets == class_index
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": accuracy,
        "precision": sum(precisions) / max(1, num_classes),
        "recall": sum(recalls) / max(1, num_classes),
        "f1": sum(f1s) / max(1, num_classes),
    }


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, task: str, num_classes: int) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    aggregate = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    batches = 0
    if train:
        model.train()
    else:
        model.eval()

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, targets, _ in progress:
        images = images.to(device)
        targets = targets.to(device)
        if task == "multi-label" and targets.ndim == 1:
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

        if task == "single-label":
            metrics = single_label_metrics(logits.detach(), targets.detach(), num_classes=num_classes)
        else:
            metrics = multilabel_metrics(logits.detach(), targets.detach())
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
    task: str,
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
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    csv_path = args.csv_path or (args.data_root / "dataset.csv")
    image_root = args.image_root or (args.data_root / "images")
    target_columns = [column.strip() for column in args.target_columns.split(",") if column.strip()]

    single_label = args.task == "single-label"
    train_base_dataset = FracAtlasClassificationDataset(
        csv_path=csv_path,
        image_roots=image_root,
        target_columns=target_columns,
        image_size=args.image_size,
        augment=True,
        use_clahe=args.use_clahe,
        single_label_only=single_label,
        return_class_index=single_label,
    )
    val_base_dataset = FracAtlasClassificationDataset(
        csv_path=csv_path,
        image_roots=image_root,
        target_columns=target_columns,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
        single_label_only=single_label,
        return_class_index=single_label,
    )
    if single_label:
        train_indices, val_indices = build_stratified_train_val_indices(
            train_base_dataset,
            target_columns=target_columns,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    else:
        train_indices, val_indices = build_train_val_indices(len(train_base_dataset), val_fraction=args.val_fraction, seed=args.seed)
    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)
    print(
        f"Task={args.task} | samples={len(train_base_dataset)} "
        f"train={len(train_dataset)} val={len(val_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121AnatomyClassifier(num_classes=len(target_columns), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss() if single_label else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = args.output_dir / "training_log.csv"
    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, task=args.task, num_classes=len(target_columns),
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, task=args.task, num_classes=len(target_columns),
        )

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
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
            ])

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        save_checkpoint(args.output_dir / "last_classifier.pt", model, optimizer, epoch, best_val_loss, target_columns, args.task)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output_dir / "best_classifier.pt", model, optimizer, epoch, best_val_loss, target_columns, args.task)


if __name__ == "__main__":
    main()
