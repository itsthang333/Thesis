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
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the no-GT pseudo-validation threshold stored in the checkpoint.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def sample_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = 2 * intersection / (pred_sum + target_sum).clamp_min(eps)
    union = pred_sum + target_sum - intersection
    iou = intersection / union.clamp_min(eps)
    precision = intersection / pred_sum.clamp_min(eps)
    recall = intersection / target_sum.clamp_min(eps)
    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "intersection": float(intersection.item()),
        "predicted_pixels": float(pred_sum.item()),
        "target_pixels": float(target_sum.item()),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_dataset = checkpoint.get("dataset")
    if checkpoint_dataset and checkpoint_dataset != args.dataset:
        raise ValueError(f"Checkpoint dataset={checkpoint_dataset!r}, requested dataset={args.dataset!r}")
    image_size = args.image_size or int(checkpoint.get("image_size", SegmentationConfig.image_size))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("decision_threshold", 0.5))
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("Evaluation threshold must be strictly between 0 and 1")

    dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.split,
        image_size=image_size,
        augment=False,
    )
    image_level_tumor_status = None
    if args.dataset == "btxrd":
        image_level_tumor_status = {
            str(sample["image_id"]): int(sample["tumor_type"]) != 0
            for sample in dataset.samples
        }
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
    tumor_precision: list[float] = []
    tumor_recall: list[float] = []
    tumor_overlap_hit: list[float] = []
    tumor_dice_ge_01: list[float] = []
    tumor_dice_ge_04: list[float] = []
    normal_empty: list[float] = []
    normal_fp_fraction: list[float] = []
    tumor_intersection = 0.0
    tumor_predicted_pixels = 0.0
    tumor_target_pixels = 0.0
    missing_tumor_gt = 0

    with torch.no_grad():
        for images, targets, image_names in tqdm(loader, desc=f"evaluate-unet-{args.split}"):
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits).cpu() >= threshold).float()
            for pred, target, image_name in zip(predictions, targets, image_names):
                has_foreground_gt = bool(target.sum().item() > 0)
                is_tumor = (
                    image_level_tumor_status[str(image_name)]
                    if image_level_tumor_status is not None
                    else has_foreground_gt
                )
                predicted_positive = bool(pred.sum().item() > 0)
                metrics = sample_metrics(pred, target)
                group = "tumor" if is_tumor else "normal"
                if is_tumor and not has_foreground_gt:
                    group = "tumor_missing_segmentation_gt"
                rows.append(
                    {
                        "image_name": image_name,
                        "group": group,
                        "predicted_positive": predicted_positive,
                        **metrics,
                        "predicted_fraction": metrics["predicted_pixels"] / pred.numel(),
                        "target_fraction": metrics["target_pixels"] / target.numel(),
                    }
                )
                if is_tumor:
                    if not has_foreground_gt:
                        missing_tumor_gt += 1
                        continue
                    tumor_dice.append(metrics["dice"])
                    tumor_iou.append(metrics["iou"])
                    tumor_precision.append(metrics["precision"])
                    tumor_recall.append(metrics["recall"])
                    tumor_overlap_hit.append(float(metrics["intersection"] > 0))
                    tumor_dice_ge_01.append(float(metrics["dice"] >= 0.1))
                    tumor_dice_ge_04.append(float(metrics["dice"] >= 0.4))
                    tumor_intersection += metrics["intersection"]
                    tumor_predicted_pixels += metrics["predicted_pixels"]
                    tumor_target_pixels += metrics["target_pixels"]
                else:
                    normal_empty.append(float(not predicted_positive))
                    normal_fp_fraction.append(metrics["predicted_pixels"] / pred.numel())

    micro_denominator = tumor_predicted_pixels + tumor_target_pixels

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "image_size": image_size,
        "threshold": threshold,
        "threshold_source": "cli" if args.threshold is not None else (
            "checkpoint_pseudo_validation" if "decision_threshold" in checkpoint else "legacy_default_0.5"
        ),
        "checkpoint_metric": checkpoint.get("checkpoint_metric"),
        "images": len(rows),
        "tumor_images": len(tumor_dice),
        "tumor_images_missing_segmentation_gt": missing_tumor_gt,
        "normal_images": len(normal_empty),
        "mean_tumor_dice": mean(tumor_dice),
        "mean_tumor_iou": mean(tumor_iou),
        "mean_tumor_precision": mean(tumor_precision),
        "mean_tumor_recall": mean(tumor_recall),
        "micro_tumor_dice": (
            2.0 * tumor_intersection / micro_denominator if micro_denominator > 0 else 0.0
        ),
        "tumor_overlap_hit_rate": mean(tumor_overlap_hit),
        "tumor_dice_ge_0.1_rate": mean(tumor_dice_ge_01),
        "tumor_dice_ge_0.4_rate": mean(tumor_dice_ge_04),
        "normal_specificity": mean(normal_empty),
        "normal_false_positive_rate": 1.0 - mean(normal_empty),
        "mean_normal_false_positive_pixel_fraction": mean(normal_fp_fraction),
        "metric_policy": {
            "overlap_metrics": "macro mean over non-empty tumor GT masks only",
            "empty_normal_masks": "reported separately; never mixed into Dice/IoU",
            "checkpoint_selection": "threshold and checkpoint selected without segmentation GT",
        },
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_name", "group", "predicted_positive", "dice", "iou", "precision", "recall",
            "intersection", "predicted_pixels", "target_pixels", "predicted_fraction", "target_fraction",
        ])
        writer.writeheader()
        writer.writerows(rows)
    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {args.output_csv}")
    print(f"Saved summary to {output_json}")


if __name__ == "__main__":
    main()
