from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if __package__:
    from .mae_reconstruction_io import (
        load_split_rows_without_annotations,
        locate_verified_image,
        sha256_file,
        verify_model_snapshot,
    )
    from .models.mae_reconstruction import pad_to_square
else:
    from mae_reconstruction_io import (
        load_split_rows_without_annotations,
        locate_verified_image,
        sha256_file,
        verify_model_snapshot,
    )
    from models.mae_reconstruction import pad_to_square


EXPECTED_TRANSFORMERS_VERSION = "4.50.2"


class NormalRadiographMAEDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict[str, str]],
        dataset_root: Path,
        image_size: int,
    ) -> None:
        self.rows = [row for row in rows if row["tumor"] == "0"]
        if not self.rows:
            raise ValueError("Normal-only MAE training set is empty")
        self.dataset_root = dataset_root
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.20, 1.00),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        image_path = locate_verified_image(self.dataset_root, row)
        with Image.open(image_path) as source:
            padded, _projection = pad_to_square(source.convert("RGB"))
        return self.transform(padded), row["image_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a frozen ImageNet ViT-MAE to clean-train normal BTXRD "
            "radiographs using only the binary image-level label."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.effective_batch_size <= 0:
        raise ValueError("Epoch and batch sizes must be positive")
    if args.effective_batch_size % args.batch_size:
        raise ValueError("effective-batch-size must be divisible by batch-size")
    if args.image_size != 224:
        raise ValueError("The predeclared adaptation grid is fixed at 224")
    if not math.isclose(args.mask_ratio, 0.75, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The predeclared MAE mask ratio is fixed at 0.75")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must lie in [0,1)")
    if len(args.source_commit) != 40:
        raise ValueError("source-commit must be a full Git commit")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("MAE training output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="train",
    )
    dataset = NormalRadiographMAEDataset(
        rows=rows,
        dataset_root=args.dataset_root.resolve(),
        image_size=args.image_size,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    import transformers
    from transformers import ViTMAEForPreTraining

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError(
            f"Transformers version drift: {transformers.__version__} "
            f"!= {EXPECTED_TRANSFORMERS_VERSION}"
        )
    model = ViTMAEForPreTraining.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    model.config.mask_ratio = args.mask_ratio
    model.config.norm_pix_loss = False
    model.gradient_checkpointing_enable()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Heavy MAE adaptation must run on a Kaggle CUDA GPU")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    accumulation_steps = args.effective_batch_size // args.batch_size
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history: list[dict[str, object]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        samples = 0
        pending = 0
        for batch_index, (pixel_values, _image_ids) in enumerate(loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                output = model(pixel_values=pixel_values)
                loss = output.loss / accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite MAE training loss")
            scaler.scale(loss).backward()
            pending += 1
            batch_samples = int(pixel_values.shape[0])
            loss_sum += float(output.loss.detach().cpu()) * batch_samples
            samples += batch_samples
            if pending == accumulation_steps or batch_index == len(loader):
                scaler.unscale_(optimizer)
                if pending < accumulation_steps:
                    correction = float(accumulation_steps) / float(pending)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                pending = 0
        epoch_row = {
            "epoch": epoch,
            "mean_reconstruction_loss": loss_sum / max(1, samples),
            "optimizer_steps": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "samples": samples,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row))

    model_dir = args.output_dir / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    shutil.copy2(
        args.model_dir / "preprocessor_config.json",
        model_dir / "preprocessor_config.json",
    )
    weight_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    preprocessor_path = model_dir / "preprocessor_config.json"
    if not weight_path.is_file() or not config_path.is_file() or not preprocessor_path.is_file():
        raise RuntimeError("Final MAE checkpoint was not saved completely")

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "stage": "normal-only BTXRD ViT-MAE adaptation",
        "scientific_role": "SKELEX-inspired mechanism feasibility, not SKELEX reproduction",
        "supervision": "clean-train images and binary image-level normal/tumor labels only",
        "source_commit": args.source_commit,
        "source_files": {
            "train_mae_reconstruction.py": sha256_file(Path(__file__).resolve()),
            "mae_reconstruction_io.py": sha256_file(
                Path(__file__).resolve().parent / "mae_reconstruction_io.py"
            ),
            "models/mae_reconstruction.py": sha256_file(
                Path(__file__).resolve().parent / "models" / "mae_reconstruction.py"
            ),
        },
        "split": "train",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "population": {
            "all_train_images": len(rows),
            "normal_training_images": len(dataset),
            "tumor_training_images_used": 0,
        },
        "initial_snapshot": snapshot,
        "training": {
            "epochs": args.epochs,
            "selected_checkpoint": "fixed final epoch; no validation selection",
            "batch_size": args.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999],
            "schedule": "linear warmup then cosine decay",
            "warmup_ratio": args.warmup_ratio,
            "image_size": args.image_size,
            "mask_ratio": args.mask_ratio,
            "normalized_pixel_loss": False,
            "augmentation": {
                "square_black_pad": True,
                "random_resized_crop_scale": [0.20, 1.00],
                "horizontal_flip_probability": 0.5,
            },
            "seed": args.seed,
        },
        "final_checkpoint": {
            "weight_file": "model/model.safetensors",
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
            "config_sha256": sha256_file(config_path),
            "preprocessor_sha256": sha256_file(preprocessor_path),
        },
        "training_history_sha256": sha256_file(history_path),
        "annotation_contract": "segmentation annotation paths were never enumerated or opened",
        "validation_images_read": False,
        "validation_gt_read": False,
        "test_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
