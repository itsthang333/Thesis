from __future__ import annotations

"""Evaluate a trained U-Net against polygon ground truth on a locked split."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.factory import build_segmentation_dataset
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U-Net checkpoint against segmentation ground truth")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--ram-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Defaults to the checkpoint image_size, then SegmentationConfig.image_size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def sample_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_dataset = checkpoint.get("dataset")
    if checkpoint_dataset and checkpoint_dataset != args.dataset:
        raise ValueError(f"Checkpoint dataset={checkpoint_dataset!r}, requested dataset={args.dataset!r}")
    image_size = args.image_size or int(checkpoint.get("image_size", SegmentationConfig.image_size))

    dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.split,
        image_size=image_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=int(checkpoint.get("base_channels", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    rows: list[dict[str, object]] = []
    tumor_dice: list[float] = []
    tumor_iou: list[float] = []
    tumor_detected: list[float] = []
    normal_empty: list[float] = []

    with torch.no_grad():
        for images, targets, image_names in tqdm(loader, desc=f"evaluate-unet-{args.split}"):
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits).cpu() >= args.threshold).float()
            for pred, target, image_name in zip(predictions, targets, image_names):
                is_tumor = bool(target.sum().item() > 0)
                predicted_positive = bool(pred.sum().item() > 0)
                dice, iou = sample_metrics(pred, target)
                group = "tumor" if is_tumor else "normal"
                rows.append(
                    {
                        "image_name": image_name,
                        "group": group,
                        "predicted_positive": predicted_positive,
                        "dice": dice,
                        "iou": iou,
                    }
                )
                if is_tumor:
                    tumor_dice.append(dice)
                    tumor_iou.append(iou)
                    tumor_detected.append(float(predicted_positive))
                else:
                    normal_empty.append(float(not predicted_positive))

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "image_size": image_size,
        "threshold": args.threshold,
        "images": len(rows),
        "tumor_images": len(tumor_dice),
        "normal_images": len(normal_empty),
        "mean_tumor_dice": mean(tumor_dice),
        "mean_tumor_iou": mean(tumor_iou),
        "tumor_detection_sensitivity": mean(tumor_detected),
        "normal_specificity": mean(normal_empty),
        "normal_false_positive_rate": 1.0 - mean(normal_empty),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "group", "predicted_positive", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {args.output_csv}")
    print(f"Saved summary to {output_json}")


if __name__ == "__main__":
    main()
