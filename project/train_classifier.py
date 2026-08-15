from __future__ import annotations

"""Train the binary image-level classifier used by LayerCAM."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.factory import build_classification_dataset
from models.classifier import DenseNet121AnatomyClassifier
from progress import should_disable_tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-profile", choices=("default",), default="default")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-columns", default="tumor")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def binary_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int]:
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    tp = int(torch.logical_and(predictions, truth).sum().item())
    fp = int(torch.logical_and(predictions, ~truth).sum().item())
    fn = int(torch.logical_and(~predictions, truth).sum().item())
    tn = int(torch.logical_and(~predictions, ~truth).sum().item())
    return tp, fp, fn, tn


def summarize(loss_sum: float, samples: int, counts: tuple[int, int, int, int]) -> dict[str, float]:
    tp, fp, fn, tn = counts
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "loss": loss_sum / max(1, samples),
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1.0e-12, precision + recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    samples = 0
    totals = np.zeros(4, dtype=np.int64)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets, _ in tqdm(
            loader,
            desc=description,
            disable=should_disable_tqdm(),
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True).reshape(-1, 1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * batch
            samples += batch
            totals += np.asarray(binary_counts(logits.detach(), targets), dtype=np.int64)
    return summarize(loss_sum, samples, tuple(int(value) for value in totals))


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_f1: float,
    args: argparse.Namespace,
    split_sha256: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "dataset": "btxrd",
            "task": "multi-label",
            "target_columns": ["tumor"],
            "num_classes": 1,
            "normalization": "imagenet",
            "image_size": args.image_size,
            "pipeline_profile": args.pipeline_profile,
            "split_manifest_sha256": split_sha256,
            "seed": args.seed,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    target_columns = [item.strip() for item in args.target_columns.split(",") if item.strip()]
    if target_columns != ["tumor"]:
        raise ValueError("the final classifier accepts only the binary tumor target")
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("epochs, batch size, and image size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("classifier output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    split_sha256 = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    seed_everything(args.seed)

    train_dataset = build_classification_dataset(
        root=args.data_root,
        split=args.train_split,
        target_columns=target_columns,
        image_size=args.image_size,
        augment=args.augment,
        normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    val_dataset = build_classification_dataset(
        root=args.data_root,
        split=args.val_split,
        target_columns=target_columns,
        image_size=args.image_size,
        augment=False,
        normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121AnatomyClassifier(
        num_classes=1,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    metadata = {
        "dataset": "btxrd",
        "supervision": "binary_image_labels",
        "train_split": args.train_split,
        "val_split": args.val_split,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split_sha256,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "normalization": "imagenet",
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    history_path = args.output_dir / "training_log.csv"
    fields = [
        "epoch",
        *[f"train_{key}" for key in ("loss", "accuracy", "precision", "recall", "f1")],
        *[f"val_{key}" for key in ("loss", "accuracy", "precision", "recall", "f1")],
    ]
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                description=f"train-{epoch}",
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=scaler,
                description=f"val-{epoch}",
            )
            row = {"epoch": epoch}
            row.update({f"train_{key}": train_metrics[key] for key in ("loss", "accuracy", "precision", "recall", "f1")})
            row.update({f"val_{key}": val_metrics[key] for key in ("loss", "accuracy", "precision", "recall", "f1")})
            writer.writerow(row)
            handle.flush()

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                epochs_without_improvement = 0
                save_checkpoint(
                    args.output_dir / "best_classifier.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_f1=best_val_f1,
                    args=args,
                    split_sha256=split_sha256,
                )
            else:
                epochs_without_improvement += 1
            if args.early_stop_patience and epochs_without_improvement >= args.early_stop_patience:
                break

    save_checkpoint(
        args.output_dir / "last_classifier.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_val_f1=best_val_f1,
        args=args,
        split_sha256=split_sha256,
    )


if __name__ == "__main__":
    main()
