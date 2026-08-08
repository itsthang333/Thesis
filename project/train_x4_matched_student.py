from __future__ import annotations

"""Train one matched X4 ResNet18-U-Net arm using train-internal selection only."""

import argparse
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.btxrd import _decode_labelme_polygon_mask, resolve_btxrd_root
from datasets.common import make_segmentation_image_transform, make_segmentation_mask_transform
from frozen_io import load_split_rows_without_annotations, locate_verified_image, sha256_file
from models.losses import bce_dice_loss
from models.unet import architecture_metadata, build_segmentation_model
from x4_contract import (
    CANONICAL_SPLIT_SHA256,
    PSEUDO_STUDENT_ARMS,
    STUDENT_ARMS,
    STUDENT_SEEDS,
    THRESHOLD_GRID,
    load_x4_protocol,
)
from x4_training_targets import validate_x4_target_bundle


RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
IMAGE_SIZE = 448
BATCH_SIZE = 8
EPOCHS = 30
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
POS_WEIGHT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=STUDENT_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=STUDENT_SEEDS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--inner-split-manifest", type=Path, required=True)
    parser.add_argument("--expected-inner-split-sha256", required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--expected-target-freeze-sha256")
    parser.add_argument("--resnet18-weight", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class X4StudentDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        rows: list[dict[str, str]],
        *,
        pseudo_root: Path | None,
        pseudo_rows: dict[str, dict[str, str]] | None,
        augment: bool,
    ) -> None:
        self.root = resolve_btxrd_root(dataset_root)
        self.rows = rows
        self.pseudo_root = pseudo_root
        self.pseudo_rows = pseudo_rows
        self.augment = augment
        self.image_transform = make_segmentation_image_transform(IMAGE_SIZE)
        self.mask_transform = make_segmentation_mask_transform(IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.rows)

    def _target(self, row: dict[str, str], size: tuple[int, int]) -> Image.Image:
        width, height = size
        if self.pseudo_rows is not None:
            relative = Path(self.pseudo_rows[row["image_id"]]["mask_path"])
            assert self.pseudo_root is not None
            with Image.open(self.pseudo_root / relative) as handle:
                return handle.convert("L").resize((width, height), Image.Resampling.NEAREST)
        if int(row["tumor"]) == 0:
            return Image.new("L", (width, height), 0)
        annotation = self.root / "Annotations" / f"{Path(row['image_id']).stem}.json"
        mask = _decode_labelme_polygon_mask(annotation, height=height, width=width)
        return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = locate_verified_image(self.root, row)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        mask = self._target(row, image.size)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return (
            self.image_transform(image),
            (self.mask_transform(mask) > 0.5).float(),
            row["image_id"],
        )


def threshold_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: tuple[float, ...] = THRESHOLD_GRID,
) -> dict[float, dict[str, float | int]]:
    probabilities = probabilities.float().flatten(1)
    targets = targets.float().flatten(1)
    target_sum = targets.sum(1)
    positive = target_sum > 0
    empty = ~positive
    output: dict[float, dict[str, float | int]] = {}
    for threshold in thresholds:
        predictions = probabilities >= threshold
        pred_sum = predictions.sum(1).float()
        intersection = (predictions.float() * targets).sum(1)
        dice = (2 * intersection + 1.0e-6) / (pred_sum + target_sum + 1.0e-6)
        output[threshold] = {
            "positive_dice_sum": float(dice[positive].sum().item()),
            "positive_count": int(positive.sum().item()),
            "empty_correct": int((pred_sum[empty] == 0).sum().item()),
            "empty_count": int(empty.sum().item()),
        }
    return output


def merge_threshold_metrics(
    totals: dict[float, dict[str, float | int]],
    update: dict[float, dict[str, float | int]],
) -> None:
    for threshold in THRESHOLD_GRID:
        for key, value in update[threshold].items():
            totals[threshold][key] += value


def finalize_threshold_metrics(
    totals: dict[float, dict[str, float | int]],
) -> dict[float, dict[str, float]]:
    result = {}
    for threshold, row in totals.items():
        result[threshold] = {
            "target_positive_dice": float(row["positive_dice_sum"]) / max(1, int(row["positive_count"])),
            "target_empty_specificity": float(row["empty_correct"]) / max(1, int(row["empty_count"])),
        }
    return result


def choose_threshold(metrics: dict[float, dict[str, float]]) -> tuple[float, dict[str, float]]:
    threshold = max(
        sorted(metrics),
        key=lambda value: (
            metrics[value]["target_positive_dice"],
            metrics[value]["target_empty_specificity"],
            -value,
        ),
    )
    return threshold, metrics[threshold]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    train: bool,
) -> tuple[float, dict[float, dict[str, float]], int]:
    model.train(train)
    loss_sum = 0.0
    images_seen = 0
    amp_skips = 0
    totals = {
        threshold: {
            "positive_dice_sum": 0.0,
            "positive_count": 0,
            "empty_correct": 0,
            "empty_count": 0,
        }
        for threshold in THRESHOLD_GRID
    }
    for images, targets, _image_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=True):
            logits = model(images)
            loss = bce_dice_loss(logits, targets, pos_weight=POS_WEIGHT)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite X4 student loss")
        if train:
            optimizer.zero_grad(set_to_none=True)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            amp_skips += int(scaler.get_scale() < scale_before)
        batch = int(images.shape[0])
        loss_sum += float(loss.item()) * batch
        images_seen += batch
        merge_threshold_metrics(
            totals,
            threshold_metrics(torch.sigmoid(logits.detach()).cpu(), targets.detach().cpu()),
        )
    if not images_seen:
        raise RuntimeError("empty X4 student loader")
    return loss_sum / images_seen, finalize_threshold_metrics(totals), amp_skips


def bare_state(model: nn.Module) -> dict[str, torch.Tensor]:
    source = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    if sha256_file(args.inner_split_manifest) != args.expected_inner_split_sha256:
        raise ValueError("X4 inner split SHA-256 mismatch")
    if sha256_file(args.resnet18_weight) != RESNET18_SHA256:
        raise ValueError("X4 ResNet18 ImageNet weight SHA-256 mismatch")
    if args.seed not in STUDENT_SEEDS:
        raise ValueError("X4 student seed differs")
    if args.arm in PSEUDO_STUDENT_ARMS and (
        args.target_root is None or args.expected_target_freeze_sha256 is None
    ):
        raise ValueError("X4 pseudo student requires a frozen target bundle")
    if args.arm == "fully_supervised" and (
        args.target_root is not None or args.expected_target_freeze_sha256 is not None
    ):
        raise ValueError("fully supervised X4 arm must read canonical train GT directly")
    if not torch.cuda.is_available():
        raise RuntimeError("X4 student training requires CUDA")

    repo_root = Path(__file__).resolve().parents[1]
    protocol, protocol_sha = load_x4_protocol(repo_root)
    canonical = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split="train",
        allow_test=False,
    )
    by_id = {row["image_id"]: row for row in canonical}
    inner_rows = read_csv(args.inner_split_manifest)
    if len(inner_rows) != 2981 or set(row["image_id"] for row in inner_rows) != set(by_id):
        raise ValueError("X4 inner split cohort differs from canonical train")
    roles_by_group: dict[str, set[str]] = {}
    for row in inner_rows:
        roles_by_group.setdefault(row["group_id"], set()).add(row["inner_role"])
    if any(len(roles) != 1 for roles in roles_by_group.values()):
        raise ValueError("X4 inner split has group leakage")
    train_rows = [by_id[row["image_id"]] for row in inner_rows if row["inner_role"] == "inner_train"]
    holdout_rows = [
        by_id[row["image_id"]] for row in inner_rows if row["inner_role"] == "inner_holdout"
    ]
    if len(train_rows) != 2516 or len(holdout_rows) != 465:
        raise ValueError("X4 inner split counts differ")

    pseudo_rows = None
    target_freeze = None
    if args.arm in PSEUDO_STUDENT_ARMS:
        assert args.target_root is not None and args.expected_target_freeze_sha256 is not None
        pseudo_rows, target_freeze = validate_x4_target_bundle(
            args.target_root,
            arm=args.arm,
            split_sha256=CANONICAL_SPLIT_SHA256,
            expected_freeze_sha256=args.expected_target_freeze_sha256,
            canonical_train_rows=canonical,
        )

    seed_everything(args.seed)
    train_dataset = X4StudentDataset(
        args.dataset_root,
        train_rows,
        pseudo_root=args.target_root,
        pseudo_rows=pseudo_rows,
        augment=True,
    )
    holdout_dataset = X4StudentDataset(
        args.dataset_root,
        holdout_rows,
        pseudo_root=args.target_root,
        pseudo_rows=pseudo_rows,
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda:0")
    model = build_segmentation_model("resnet18_unet", pretrained=True).to(device)
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if args.multi_gpu and len(devices) > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True)
    history_path = args.output_dir / "training_history.csv"
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = -1
    best_threshold = 0.5
    amp_skips = 0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "inner_holdout_loss",
            "selected_threshold",
            "inner_target_positive_dice",
            "inner_target_empty_specificity",
            "amp_skipped_steps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            train_loss, _train_thresholds, train_skips = run_epoch(
                model, train_loader, device, optimizer, scaler, train=True
            )
            holdout_loss, holdout_thresholds, holdout_skips = run_epoch(
                model, holdout_loader, device, optimizer, scaler, train=False
            )
            amp_skips += train_skips + holdout_skips
            threshold, selected = choose_threshold(holdout_thresholds)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "inner_holdout_loss": holdout_loss,
                "selected_threshold": threshold,
                "inner_target_positive_dice": selected["target_positive_dice"],
                "inner_target_empty_specificity": selected["target_empty_specificity"],
                "amp_skipped_steps": amp_skips,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            key = (
                -holdout_loss,
                selected["target_positive_dice"],
                selected["target_empty_specificity"],
                -threshold,
            )
            checkpoint = {
                "schema_version": 1,
                "stage": "x4_matched_student_checkpoint_v1",
                "arm": args.arm,
                "seed": args.seed,
                "epoch": epoch,
                "model_state_dict": bare_state(model),
                "architecture": architecture_metadata("resnet18_unet"),
                "model_architecture": "resnet18_unet",
                "pretrained_encoder": True,
                "image_size": IMAGE_SIZE,
                "decision_threshold": threshold,
                "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
                "inner_split_sha256": args.expected_inner_split_sha256,
                "x4_protocol_sha256": protocol_sha,
                "target_freeze_sha256": args.expected_target_freeze_sha256,
                "supervision_mode": (
                    "fully_supervised" if args.arm == "fully_supervised" else "image_label_only_pseudo_mask"
                ),
                "outer_validation_checkpoint_selection": False,
                "test_evaluated": False,
            }
            torch.save(checkpoint, args.output_dir / "last_student.pt")
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_threshold = threshold
                torch.save(checkpoint, args.output_dir / "best_student.pt")

    best_path = args.output_dir / "best_student.pt"
    last_path = args.output_dir / "last_student.pt"
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "stage": "x4_matched_student_training_v1",
        "arm": args.arm,
        "seed": args.seed,
        "train_images": len(train_rows),
        "inner_holdout_images": len(holdout_rows),
        "outer_validation_images_used": 0,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "epochs_completed": EPOCHS,
        "amp_skipped_steps": amp_skips,
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "inner_split_sha256": args.expected_inner_split_sha256,
        "x4_protocol_sha256": protocol_sha,
        "target_freeze_sha256": args.expected_target_freeze_sha256,
        "target_freeze": target_freeze,
        "architecture": architecture_metadata("resnet18_unet"),
        "scientific_config": protocol["matched_student"],
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "training_history_sha256": sha256_file(history_path),
        "cuda_devices": devices,
        "data_parallel": isinstance(model, nn.DataParallel),
        "spatial_ground_truth_training": args.arm == "fully_supervised",
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
