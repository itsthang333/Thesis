from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SegmentationConfig
from datasets.ramh1200 import RAMH1200SegmentationDataset
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against RAM-H1200 bone GT masks")
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--annotation-name", type=str, default="_annotations_bone_rle.coco.json")
    parser.add_argument("--pred-mask-root", type=Path, default=ROOT / "outputs" / "pseudo_masks" / "masks")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "ramh1200_eval.csv")
    return parser.parse_args()


def resolve_pred_mask(mask_root: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = mask_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_pred_mask(mask_path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    mask = Image.open(mask_path).convert("L")
    return (transform(mask) > 0.5).float()


def binary_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


def main() -> None:
    args = parse_args()
    dataset = RAMH1200SegmentationDataset(
        root=args.ram_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        annotation_name=args.annotation_name,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[object]] = []
    dice_values: list[float] = []
    iou_values: list[float] = []
    missing = 0

    for _, gt_masks, image_names in tqdm(loader, desc="evaluate"):
        pred_masks = []
        valid_names = []
        valid_gt = []
        for index, image_name in enumerate(image_names):
            pred_path = resolve_pred_mask(args.pred_mask_root, image_name)
            if pred_path is None:
                missing += 1
                rows.append([image_name, "missing", "", ""])
                continue
            pred_masks.append(load_pred_mask(pred_path, args.image_size))
            valid_gt.append(gt_masks[index])
            valid_names.append(image_name)

        if not pred_masks:
            continue

        for image_name, pred_mask, gt_mask in zip(valid_names, pred_masks, valid_gt):
            dice, iou = binary_metrics(pred_mask, gt_mask)
            dice_values.append(dice)
            iou_values.append(iou)
            rows.append([image_name, "ok", dice, iou])

    mean_dice = sum(dice_values) / max(1, len(dice_values))
    mean_iou = sum(iou_values) / max(1, len(iou_values))

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "status", "dice", "iou"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["mean", "ok", mean_dice, mean_iou])
        writer.writerow(["missing", missing, "", ""])

    print(f"RAM-H1200 {args.split}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, missing={missing}")
    print(f"Saved per-image results to {args.output_csv}")


if __name__ == "__main__":
    main()
