from __future__ import annotations

"""Freeze fully-supervised predictions without opening spatial annotations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.btxrd import resolve_btxrd_root
from datasets.common import make_segmentation_image_transform
from evaluation.frozen_test_guard import verify_frozen_test_config
from frozen_io import load_split_rows_without_annotations, sha256_file
from models.unet import architecture_name_from_metadata, build_segmentation_model


EXPECTED_COUNTS = {
    "val": {"images": 371, "tumor": 184, "normal": 187},
    "test": {"images": 373, "tumor": 187, "normal": 186},
}


class ImageOnlyDataset(Dataset):
    def __init__(self, image_root: Path, rows: list[dict[str, str]], image_size: int) -> None:
        self.image_root = image_root
        self.rows = rows
        self.transform = make_segmentation_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id = self.rows[index]["image_id"]
        path = self.image_root / image_id
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as handle:
            tensor = self.transform(handle.convert("RGB"))
        return tensor, image_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if abs(args.threshold - 0.20) > 1.0e-12 or args.image_size != 448:
        raise ValueError("fully-supervised test freeze is locked to 448 px and threshold 0.20")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("fully-supervised checkpoint SHA-256 mismatch")
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
        requested_checkpoint=args.checkpoint,
        checkpoint_any_of=("supervised_unet_checkpoint",),
    )
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
        allow_test=args.split == "test",
    )
    counts = {
        "images": len(rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
    }
    if counts != EXPECTED_COUNTS[args.split]:
        raise ValueError(f"canonical {args.split} counts differ: {counts}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("split_manifest_sha256") != args.expected_split_sha256:
        raise ValueError("fully-supervised checkpoint split SHA-256 mismatch")
    if checkpoint.get("supervision_mode") != "fully_supervised_comparison":
        raise ValueError("checkpoint is not marked fully_supervised_comparison")
    if checkpoint.get("ground_truth_spatial_supervision") is not True:
        raise ValueError("checkpoint does not record spatial GT supervision")
    if checkpoint.get("wsss_eligible") is not False:
        raise ValueError("fully-supervised checkpoint must be ineligible for WSSS")
    if checkpoint.get("comparison_only") is not True:
        raise ValueError("fully-supervised checkpoint must be comparison-only")
    if checkpoint.get("validation_ground_truth_checkpoint_selection") is not True:
        raise ValueError("fully-supervised checkpoint selection provenance is missing")
    if checkpoint.get("train_split") != "train" or checkpoint.get("val_split") != "val":
        raise ValueError("fully-supervised checkpoint was not selected on canonical train/val")
    if int(checkpoint.get("image_size", -1)) != args.image_size:
        raise ValueError("fully-supervised checkpoint image size mismatch")

    architecture = architecture_name_from_metadata(checkpoint.get("architecture"))
    model = build_segmentation_model(architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("fully-supervised prediction freeze requires CUDA")
    device = torch.device("cuda:0")
    model.to(device).eval()
    device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    data_parallel = len(device_names) > 1
    inference_model: nn.Module = nn.DataParallel(model) if data_parallel else model

    btxrd_root = resolve_btxrd_root(args.dataset_root)
    dataset = ImageOnlyDataset(btxrd_root / "images", rows, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True)
    row_by_id = {row["image_id"]: row for row in rows}
    manifest_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for images, image_ids in loader:
            logits = inference_model(images.to(device, non_blocking=True))
            predictions = torch.sigmoid(logits).cpu().numpy()[:, 0] >= args.threshold
            for prediction, image_id_raw in zip(predictions, image_ids):
                image_id = str(image_id_raw)
                mask = prediction.astype(np.uint8) * 255
                relative = Path("masks") / f"{Path(image_id).stem}.png"
                output_path = args.output_dir / relative
                Image.fromarray(mask, mode="L").save(output_path)
                source = row_by_id[image_id]
                manifest_rows.append(
                    {
                        "image_id": image_id,
                        "group_id": source["group_id"],
                        "tumor": source["tumor"],
                        "mask_path": relative.as_posix(),
                        "mask_sha256": sha256_file(output_path),
                        "height": int(mask.shape[0]),
                        "width": int(mask.shape[1]),
                        "positive_pixels": int(prediction.sum()),
                    }
                )
    if len(manifest_rows) != counts["images"]:
        raise RuntimeError("fully-supervised prediction cohort is incomplete")
    manifest_path = args.output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    freeze = {
        "stage": "fully_supervised_prediction_freeze_v1",
        "cohort_split": args.split,
        "split_sha256": args.expected_split_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "architecture": architecture,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "images": counts["images"],
        "tumor_images": counts["tumor"],
        "normal_images": counts["normal"],
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "predictions_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": counts["images"] if args.split == "test" else 0,
        "test_evaluated": False,
        "comparison_only": True,
        "wsss_eligible": False,
        "cuda_device_names": device_names,
        "data_parallel": data_parallel,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
