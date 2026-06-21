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

from config import SegmentationConfig
from datasets.fracatlas import FracAtlasSegmentationDataset, build_train_val_indices
from models.losses import bce_dice_loss, dice_coefficient, iou_score
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on pseudo masks with GT validation")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "FracAtlas")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=ROOT / "outputs" / "pseudo_masks" / "masks", help="Pseudo masks cho Train")
    parser.add_argument("--gt-mask-root", type=Path, default=None, help="Ground Truth masks cho Validation")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=SegmentationConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SegmentationConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=SegmentationConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=SegmentationConfig.epochs)
    parser.add_argument("--val-fraction", type=float, default=SegmentationConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=SegmentationConfig.seed)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "segmentation")
    parser.add_argument("--use-clahe", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def build_aligned_indices(dataset: FracAtlasSegmentationDataset, data_root: Path, image_root: Path, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    csv_path = data_root / "dataset.csv"
    global_stems = []
    
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col_name = "image_id" if "image_id" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                global_stems.append(Path(row[col_name]).stem)
    else:
        global_files = sorted([f for f in image_root.iterdir() if f.is_file()])
        global_stems = [f.stem for f in global_files if f.suffix.lower() in (".jpg", ".png", ".jpeg")]

    if not global_stems:
        raise RuntimeError("Không tìm thấy dữ liệu gốc để tái tạo tập split của Stage 1.")

    global_train_idx, global_val_idx = build_train_val_indices(len(global_stems), val_fraction=val_fraction, seed=seed)
    
    stage1_train_stems = {global_stems[i] for i in global_train_idx}
    stage1_val_stems = {global_stems[i] for i in global_val_idx}

    train_indices = []
    val_indices = []
    
    dataset_paths = getattr(dataset, "image_paths", getattr(dataset, "images", None))
    
    print("Đang đồng bộ hóa Train/Val split với Stage 1...")
    for i in range(len(dataset)):
        if dataset_paths is not None:
            stem = Path(dataset_paths[i]).stem
        else:
            _, _, name = dataset[i]
            stem = Path(name).stem

        if stem in stage1_train_stems:
            train_indices.append(i)
        elif stem in stage1_val_stems:
            val_indices.append(i)
            
    print(f"Đã ánh xạ: {len(train_indices)} ảnh cho Train, {len(val_indices)} ảnh cho Validation.")
    return train_indices, val_indices


def run_epoch(model, loader, optimizer, scaler, device, train: bool) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0
    if train:
        model.train()
    else:
        model.eval()

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks, _ in progress:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = bce_dice_loss(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice = dice_coefficient(logits.detach(), masks.detach())
        iou = iou_score(logits.detach(), masks.detach())
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        batches += 1
        progress.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    if batches == 0:
        return 0.0, {"dice": 0.0, "iou": 0.0}
    return total_loss / batches, {"dice": total_dice / batches, "iou": total_iou / batches}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    image_root = args.image_root or (args.data_root / "images")
    
    train_base_dataset = FracAtlasSegmentationDataset(
        image_roots=image_root,
        mask_root=args.mask_root,
        image_size=args.image_size,
        augment=True, # Train CẦN augment
        use_clahe=args.use_clahe,
    )
    
    val_mask_root = args.gt_mask_root if args.gt_mask_root else args.mask_root
    if not args.gt_mask_root:
        print("[CẢNH BÁO] Không có GT masks. Model đang đánh giá validation trên Pseudo Masks!")
        
    val_base_dataset = FracAtlasSegmentationDataset(
        image_roots=image_root,
        mask_root=val_mask_root,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
    )
    
    train_indices, val_indices = build_aligned_indices(
        dataset=train_base_dataset, 
        data_root=args.data_root, 
        image_root=image_root, 
        val_fraction=args.val_fraction, 
        seed=args.seed
    )
    
    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = args.output_dir / "training_log.csv"
    
    best_val_dice = 0.0 
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, train=False)

        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                epoch,
                train_loss,
                train_metrics["dice"],
                train_metrics["iou"],
                val_loss,
                val_metrics["dice"],
                val_metrics["iou"],
            ])

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, best_val_dice)
        
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_val_dice)
            print(f"--> Đã lưu Best Model mới với Dice = {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
