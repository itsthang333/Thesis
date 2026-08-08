from __future__ import annotations

"""Freeze one X4 student's outer-validation predictions before opening GT."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.btxrd import resolve_btxrd_root
from datasets.common import make_segmentation_image_transform
from frozen_io import load_split_rows_without_annotations, locate_verified_image, sha256_file
from models.unet import architecture_name_from_metadata, build_segmentation_model
from x4_contract import (
    CANONICAL_SPLIT_SHA256,
    RESNET18_IMAGENET1K_V1_SHA256,
    STUDENT_ARMS,
    STUDENT_SEEDS,
    load_x4_protocol,
)


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


def latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency values must be non-empty")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("latency values must be finite and non-negative")
    return {
        "images": len(values),
        "median_seconds_per_image": float(np.median(array)),
        "iqr_low_seconds_per_image": float(np.percentile(array, 25)),
        "iqr_high_seconds_per_image": float(np.percentile(array, 75)),
        "mean_seconds_per_image": float(np.mean(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=STUDENT_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=STUDENT_SEEDS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("X4 student checkpoint SHA-256 mismatch")
    protocol, protocol_sha = load_x4_protocol(Path(__file__).resolve().parents[1])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != "x4_matched_student_checkpoint_v1"
        or checkpoint.get("arm") != args.arm
        or int(checkpoint.get("seed", -1)) != args.seed
        or checkpoint.get("split_manifest_sha256") != CANONICAL_SPLIT_SHA256
        or checkpoint.get("x4_protocol_sha256") != protocol_sha
        or checkpoint.get("encoder_weight_sha256") != RESNET18_IMAGENET1K_V1_SHA256
        or checkpoint.get("outer_validation_checkpoint_selection") is not False
        or checkpoint.get("test_evaluated") is not False
    ):
        raise ValueError("X4 student checkpoint violates the freeze boundary")
    image_size = int(checkpoint["image_size"])
    threshold = float(checkpoint["decision_threshold"])
    if image_size != int(protocol["matched_student"]["input_size"]):
        raise ValueError("X4 student image size differs")
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split="val",
        allow_test=False,
    )
    if len(rows) != 371 or sum(int(row["tumor"]) for row in rows) != 184:
        raise ValueError("X4 prediction freeze requires canonical 371/184 validation")
    if not torch.cuda.is_available():
        raise RuntimeError("X4 prediction freeze requires CUDA")

    architecture = architecture_name_from_metadata(checkpoint.get("architecture"))
    model = build_segmentation_model(architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    inference_model: nn.Module = nn.DataParallel(model) if len(devices) > 1 else model
    dataset = ImageOnlyDataset(resolve_btxrd_root(args.dataset_root), rows, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    args.output_dir.mkdir(parents=True)
    mask_root = args.output_dir / "masks"
    probability_root = args.output_dir / "probabilities"
    mask_root.mkdir()
    probability_root.mkdir()
    row_by_id = {row["image_id"]: row for row in rows}
    manifest_rows: list[dict[str, object]] = []
    warmup_iterations = 3
    warmup_images = next(iter(loader))[0].to(device, non_blocking=True)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = inference_model(warmup_images)
    torch.cuda.synchronize()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    latency_seconds_per_image: list[float] = []
    timed_inference_start = time.perf_counter()
    with torch.inference_mode():
        for images, image_ids, heights, widths in loader:
            torch.cuda.synchronize()
            batch_start = time.perf_counter()
            probabilities = torch.sigmoid(
                inference_model(images.to(device, non_blocking=True))
            ).cpu().numpy()[:, 0]
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_start
            latency_seconds_per_image.extend(
                [batch_elapsed / len(image_ids)] * len(image_ids)
            )
            for probability, image_id_raw, native_height, native_width in zip(
                probabilities, image_ids, heights, widths
            ):
                image_id = str(image_id_raw)
                stem = Path(image_id).stem
                mask = np.asarray(probability >= threshold, dtype=np.uint8) * 255
                mask_relative = Path("masks") / f"{stem}.png"
                probability_relative = Path("probabilities") / f"{stem}.npz"
                mask_path = args.output_dir / mask_relative
                probability_path = args.output_dir / probability_relative
                Image.fromarray(mask, mode="L").save(mask_path, optimize=True)
                np.savez_compressed(
                    probability_path,
                    schema_version=np.asarray(1, dtype=np.int32),
                    probability=np.asarray(probability, dtype=np.float16),
                )
                source = row_by_id[image_id]
                manifest_rows.append(
                    {
                        "image_id": image_id,
                        "group_id": source["group_id"],
                        "tumor": source["tumor"],
                        "native_height": int(native_height),
                        "native_width": int(native_width),
                        "mask_path": mask_relative.as_posix(),
                        "mask_sha256": sha256_file(mask_path),
                        "probability_path": probability_relative.as_posix(),
                        "probability_sha256": sha256_file(probability_path),
                        "positive_pixels_448": int((mask > 0).sum()),
                    }
                )
    if len(manifest_rows) != 371:
        raise RuntimeError("X4 validation prediction cohort is incomplete")
    manifest_path = args.output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    timed_inference_elapsed = time.perf_counter() - timed_inference_start
    storage_bytes_before_freeze = sum(
        path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file()
    )
    device_memory = [
        {
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
    ]
    freeze = {
        "schema_version": 1,
        "stage": "x4_student_prediction_freeze_v1",
        "arm": args.arm,
        "seed": args.seed,
        "split": "val",
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "x4_protocol_sha256": protocol_sha,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "architecture": architecture,
        "image_size": image_size,
        "threshold": threshold,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "predictions_frozen_before_spatial_ground_truth": True,
        "spatial_ground_truth_used": False,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "cuda_devices": devices,
        "x12_efficiency": {
            "stage": "matched_student_online_inference_and_freeze",
            "same_gpu_requirement": "compare only bundles produced on the same declared GPU type",
            "batch_size": args.batch_size,
            "warmup_iterations": warmup_iterations,
            "timed_images": len(latency_seconds_per_image),
            "timed_inference_elapsed_seconds": timed_inference_elapsed,
            "latency": latency_summary(latency_seconds_per_image),
            "device_memory": device_memory,
            "storage_bytes_before_prediction_freeze_json": storage_bytes_before_freeze,
            "offline_pseudo_label_generation_included": False,
        },
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
