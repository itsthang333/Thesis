from __future__ import annotations

import argparse
import csv
import json
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

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.factory import build_segmentation_dataset
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against RAM-H1200/BTXRD GT masks")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=SUPPORTED_DATASETS)
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1",
                        help="Dataset root (RAM-H1200 root or BTXRD root, depending on --dataset)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--annotation-name", type=str, default="_annotations_bone_rle.coco.json",
                        help="RAM-H1200 only; ignored for --dataset btxrd")
    parser.add_argument("--pred-mask-root", type=Path, default=ROOT / "outputs" / "pseudo_masks" / "masks")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "eval.csv")
    parser.add_argument("--image-list", type=Path, default=None,
                        help="Optional text file of image names to evaluate (one per line).")
    parser.add_argument("--skipped-list", type=Path, default=None,
                        help="Path to skipped_low_confidence.txt from generate_pseudo_masks.py "
                        "(defaults to <pred-mask-root>/../skipped_low_confidence.txt if present). "
                        "Tumor images the classifier skipped for low confidence get an all-zero "
                        "pseudo-mask by construction; reporting their Dice alongside images that "
                        "actually ran CAM/SAM would blame the wrong pipeline stage for the failure.")
    return parser.parse_args()


def load_skipped_names(path: Path | None, pred_mask_root: Path) -> set[str]:
    candidate = path if path is not None else pred_mask_root.parent / "skipped_low_confidence.txt"
    if not candidate.exists():
        return set()
    return {Path(line.strip()).stem for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_run_protocol(pred_mask_root: Path) -> str:
    metadata_path = pred_mask_root.parent / "run_metadata.json"
    if not metadata_path.exists():
        return "unknown"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(metadata.get("cam_target_class", "unknown"))


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
    dataset = build_segmentation_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        annotation_name=args.annotation_name,
    )
    if args.image_list is not None:
        requested_names = {
            line.strip() for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        original_count = len(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples if str(sample["image_id"]) in requested_names
        ]
        if not dataset.samples:
            raise ValueError(f"--image-list {args.image_list} matched no images in split '{args.split}'")
        print(f"Image-list filter: {len(dataset.samples)}/{original_count} samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_names = load_skipped_names(args.skipped_list, args.pred_mask_root)
    run_protocol = load_run_protocol(args.pred_mask_root)
    print(f"Pseudo-mask protocol: {run_protocol}")
    if skipped_names:
        print(f"Loaded {len(skipped_names)} low-confidence-skipped image names")

    rows: list[list[object]] = []
    # Split metrics by whether the GT mask actually has a lesion/bone region,
    # and further split tumor images by whether the classifier skipped them
    # for low confidence. A pooled mean over all three conflates three
    # different failure modes: normal images trivially score Dice=1 when both
    # prediction and GT are empty (a correct *detection*, not evidence of good
    # *segmentation*); classifier-skipped tumor images always get Dice≈0 by
    # construction (an all-zero mask was saved before CAM/SAM ever ran) and
    # blame the classifier threshold, not CAM/SAM/morphology; only tumor
    # images that actually ran the full pipeline reflect CAM/SAM quality.
    tumor_dice: list[float] = []
    tumor_iou: list[float] = []
    end_to_end_tumor_dice: list[float] = []
    end_to_end_tumor_iou: list[float] = []
    tumor_skipped_count = 0
    normal_specificity: list[float] = []  # 1.0 if predicted mask is also empty, else 0.0
    normal_false_positive_pixels = 0
    missing = 0

    for _, gt_masks, image_names in tqdm(loader, desc="evaluate"):
        pred_masks = []
        valid_names = []
        valid_gt = []
        for index, image_name in enumerate(image_names):
            pred_path = resolve_pred_mask(args.pred_mask_root, image_name)
            if pred_path is None:
                missing += 1
                rows.append([image_name, "missing", "", "", "", ""])
                continue
            pred_masks.append(load_pred_mask(pred_path, args.image_size))
            valid_gt.append(gt_masks[index])
            valid_names.append(image_name)

        if not pred_masks:
            continue

        for image_name, pred_mask, gt_mask in zip(valid_names, pred_masks, valid_gt):
            is_tumor = bool(gt_mask.sum().item() > 0)
            was_skipped = Path(image_name).stem in skipped_names
            dice, iou = binary_metrics(pred_mask, gt_mask)
            group = "tumor" if is_tumor else "normal"
            rows.append([image_name, "ok", group, was_skipped, dice, iou])
            if is_tumor:
                # End-to-end metrics include detection/classification
                # failures. If predicted protocol maps a tumor image to the
                # normal class, generation deliberately writes an empty mask;
                # that Dice/IoU belongs in the end-to-end result even though
                # it is excluded from the conditional CAM/SAM-only metric.
                end_to_end_tumor_dice.append(dice)
                end_to_end_tumor_iou.append(iou)
                if was_skipped:
                    tumor_skipped_count += 1
                else:
                    tumor_dice.append(dice)
                    tumor_iou.append(iou)
            else:
                predicted_empty = bool(pred_mask.sum().item() == 0)
                normal_specificity.append(1.0 if predicted_empty else 0.0)
                if not predicted_empty:
                    normal_false_positive_pixels += int(pred_mask.sum().item())

    mean_tumor_dice = sum(tumor_dice) / max(1, len(tumor_dice))
    mean_tumor_iou = sum(tumor_iou) / max(1, len(tumor_iou))
    mean_end_to_end_tumor_dice = sum(end_to_end_tumor_dice) / max(1, len(end_to_end_tumor_dice))
    mean_end_to_end_tumor_iou = sum(end_to_end_tumor_iou) / max(1, len(end_to_end_tumor_iou))
    specificity = sum(normal_specificity) / max(1, len(normal_specificity))
    false_positive_rate = 1.0 - specificity

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "status", "group", "skipped_low_confidence", "dice", "iou"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["tumor_images_evaluated", len(tumor_dice), "", "", "", ""])
        writer.writerow(["tumor_images_skipped_low_confidence", tumor_skipped_count, "", "", "", ""])
        writer.writerow(["cam_sam_conditional_mean_tumor_dice", "", "", "", mean_tumor_dice, ""])
        writer.writerow(["cam_sam_conditional_mean_tumor_iou", "", "", "", "", mean_tumor_iou])
        writer.writerow(["end_to_end_tumor_images", len(end_to_end_tumor_dice), "", "", "", ""])
        writer.writerow(["end_to_end_mean_tumor_dice", "", "", "", mean_end_to_end_tumor_dice, ""])
        writer.writerow(["end_to_end_mean_tumor_iou", "", "", "", "", mean_end_to_end_tumor_iou])
        writer.writerow(["cam_target_class_protocol", run_protocol, "", "", "", ""])
        writer.writerow(["normal_images", len(normal_specificity), "", "", "", ""])
        writer.writerow(["specificity_empty_mask_rate", "", "", "", specificity, ""])
        writer.writerow(["false_positive_rate", "", "", "", false_positive_rate, ""])
        writer.writerow(["normal_false_positive_pixels_total", normal_false_positive_pixels, "", "", "", ""])
        writer.writerow(["missing", missing, "", "", "", ""])

    print(
        f"{args.dataset} {args.split}: "
        f"tumor images evaluated={len(tumor_dice)} (skipped for low confidence={tumor_skipped_count}) "
        f"conditional Dice={mean_tumor_dice:.4f} IoU={mean_tumor_iou:.4f} | "
        f"end-to-end Dice={mean_end_to_end_tumor_dice:.4f} IoU={mean_end_to_end_tumor_iou:.4f} | "
        f"normal images={len(normal_specificity)} specificity={specificity:.4f} "
        f"false_positive_rate={false_positive_rate:.4f} | missing={missing}"
    )
    print(f"Saved per-image results to {args.output_csv}")


if __name__ == "__main__":
    main()
