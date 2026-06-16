from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.fracatlas import FracAtlasSegmentationDataset
from models.losses import dice_coefficient, iou_score
from models.unet import UNet

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net Segmentation Model")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "FracAtlas")
    parser.add_argument("--image-root", type=Path, default=None)
    # Trỏ thẳng vào thư mục chứa mask CHUẨN (Ground Truth) của dataset gốc
    parser.add_argument("--gt-mask-root", type=Path, required=True, help="Path to ground truth masks")
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--use-clahe", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_root = args.image_root or (args.data_root / "images")
    
    # Load dataset với mask chuẩn (không dùng augment khi evaluate)
    dataset = FracAtlasSegmentationDataset(
        image_roots=image_root,
        mask_root=args.gt_mask_root,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=SequentialSampler(dataset), 
        num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo và load model
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    checkpoint = torch.load(args.segmentation_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    print("Bắt đầu đánh giá mô hình trên tập Ground Truth...")
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            
            # Tính metrics
            dice = dice_coefficient(logits, masks)
            iou = iou_score(logits, masks)
            
            total_dice += dice.item()
            total_iou += iou.item()
            batches += 1

    final_dice = total_dice / batches
    final_iou = total_iou / batches

    print("\n" + "="*40)
    print("KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)")
    print("="*40)
    print(f"Mean Dice Score : {final_dice:.4f}")
    print(f"Mean IoU Score  : {final_iou:.4f}")
    print("="*40)

    # Lưu kết quả ra file txt
    with open(args.output_dir / "metrics_report.txt", "w") as f:
        f.write(f"Mean Dice Score: {final_dice:.4f}\n")
        f.write(f"Mean IoU Score: {final_iou:.4f}\n")

if __name__ == "__main__":
    main()