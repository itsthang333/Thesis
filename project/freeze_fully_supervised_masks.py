from __future__ import annotations

"""Freeze fully-supervised validation masks without opening spatial annotations."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import make_segmentation_image_transform
from datasets.btxrd import resolve_btxrd_root
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
)
from models.unet import architecture_name_from_metadata, build_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


class ImageOnlyDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], image_size: int) -> None:
        self.root = root
        self.rows = rows
        self.transform = make_segmentation_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = locate_verified_image(self.root, row)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            native_width, native_height = image.size
            tensor = self.transform(image)
        return tensor, row["image_id"], native_height, native_width


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("fully-supervised checkpoint SHA-256 mismatch")
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    if len(rows) != 371 or sum(int(row["tumor"]) for row in rows) != 184:
        raise ValueError("fully mask freeze requires canonical 371/184 validation")
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    dataset = ImageOnlyDataset(btxrd_root, rows, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # The hash-locked checkpoint is an output of the audited in-repository trainer.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("split_manifest_sha256") != args.expected_split_sha256:
        raise ValueError("checkpoint was trained against a different split manifest")
    if int(checkpoint.get("image_size", args.image_size)) != args.image_size:
        raise ValueError("requested image size differs from the checkpoint protocol")
    architecture = architecture_name_from_metadata(checkpoint.get("architecture"))
    model = build_segmentation_model(architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, object]] = []
    row_by_id = {row["image_id"]: row for row in rows}
    with torch.inference_mode():
        for images, image_ids, native_heights, native_widths in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for probability, image_id, native_height, native_width in zip(
                probabilities, image_ids, native_heights, native_widths
            ):
                mask = np.asarray(probability >= args.threshold, dtype=np.uint8) * 255
                mask_path = mask_dir / f"{Path(str(image_id)).stem}.png"
                Image.fromarray(mask, mode="L").save(mask_path, optimize=True)
                row = row_by_id[str(image_id)]
                manifest_rows.append({
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "tumor": row["tumor"],
                    "native_height": int(native_height),
                    "native_width": int(native_width),
                    "mask_path": str(mask_path.relative_to(args.output_dir)).replace("\\", "/"),
                    "mask_sha256": sha256_file(mask_path),
                    "pred_area_pixels_448": int((mask > 0).sum()),
                })

    manifest_rows.sort(key=lambda row: str(row["image_id"]))
    manifest_path = args.output_dir / "mask_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    freeze = {
        "schema_version": 1,
        "stage": "fully_supervised_validation_mask_freeze_v1",
        "split": "val",
        "split_sha256": args.expected_split_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "architecture": architecture,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "images": len(manifest_rows),
        "tumor_images": sum(int(row["tumor"]) for row in manifest_rows),
        "mask_manifest_sha256": sha256_file(manifest_path),
        "masks_frozen_before_spatial_ground_truth": True,
        "spatial_ground_truth_used": False,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "mask_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "mask_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
