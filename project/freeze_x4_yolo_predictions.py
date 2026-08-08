from __future__ import annotations

"""Freeze native-resolution binary-union YOLO masks before the common evaluator opens GT."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import resolve_btxrd_root
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
    validate_sha256,
)
from train_x4_yolov8s_seg import ULTRALYTICS_VERSION


def union_instance_masks(values: np.ndarray | None, *, height: int, width: int) -> np.ndarray:
    if values is None or np.asarray(values).size == 0:
        return np.zeros((height, width), dtype=bool)
    masks = np.asarray(values)
    if masks.ndim != 3:
        raise ValueError("YOLO instance masks must be NxHxW")
    union = np.any(masks > 0.5, axis=0)
    if union.shape != (height, width):
        union = np.asarray(
            Image.fromarray((union * 255).astype(np.uint8), mode="L").resize(
                (width, height), Image.Resampling.NEAREST
            )
        ) > 0
    return union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--expected-training-report-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=600)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    split_sha = validate_sha256(args.expected_split_sha256, name="split SHA-256")
    report_path = args.training_root / "training_report.json"
    if sha256_file(report_path) != validate_sha256(
        args.expected_training_report_sha256, name="training report SHA-256"
    ):
        raise ValueError("X4 YOLO training report changed")
    training = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = args.training_root / "best.pt"
    checkpoint_sha = validate_sha256(args.expected_checkpoint_sha256, name="checkpoint SHA-256")
    if (
        sha256_file(checkpoint) != checkpoint_sha
        or training.get("best_checkpoint_sha256") != checkpoint_sha
        or training.get("test_images_read") != 0
        or training.get("test_evaluated") is not False
    ):
        raise ValueError("X4 YOLO training/checkpoint binding differs")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError("X4 YOLO Ultralytics version differs")
    split_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=split_sha, split="val", allow_test=False
    )
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("X4 YOLO validation cohort differs")
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    image_paths = [locate_verified_image(btxrd_root, row) for row in split_rows]
    by_id = {row["image_id"]: row for row in split_rows}
    args.output_dir.mkdir(parents=True)
    mask_root = args.output_dir / "masks"
    mask_root.mkdir()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    started = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina_masks=True,
        device=args.device,
        verbose=False,
        save=False,
    )
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    seen: set[str] = set()
    for result in results:
        image_id = Path(result.path).name
        if image_id not in by_id or image_id in seen:
            raise ValueError(f"Unexpected or duplicate YOLO result: {image_id}")
        seen.add(image_id)
        split_row = by_id[image_id]
        height, width = tuple(int(value) for value in result.orig_shape)
        values = None if result.masks is None else result.masks.data.detach().cpu().numpy()
        union = union_instance_masks(values, height=height, width=width)
        mask_path = mask_root / f"{Path(image_id).stem}.png"
        Image.fromarray((union * 255).astype(np.uint8), mode="L").save(mask_path)
        latency = sum(float(result.speed.get(key, 0.0)) for key in ("preprocess", "inference", "postprocess")) / 1000.0
        latencies.append(latency)
        rows.append(
            {
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "tumor": split_row["tumor"],
                "native_height": height,
                "native_width": width,
                "instances": 0 if values is None else len(values),
                "predicted_pixels": int(union.sum()),
                "mask_path": mask_path.relative_to(args.output_dir).as_posix(),
                "mask_sha256": sha256_file(mask_path),
                "latency_seconds": latency,
            }
        )
    elapsed = time.perf_counter() - started
    if seen != set(by_id):
        raise ValueError("X4 YOLO did not freeze exactly 371 validation predictions")
    rows.sort(key=lambda row: str(row["image_id"]))
    manifest_path = args.output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    peak_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    ordered_latency = sorted(latencies)
    q1, q3 = np.percentile(np.asarray(ordered_latency), (25, 75)).tolist()
    report = {
        "schema_version": 1,
        "stage": "x4_yolo_prediction_freeze_v1",
        "split_sha256": split_sha,
        "training_report_sha256": args.expected_training_report_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "retina_masks": True,
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "median_seconds_per_image": statistics.median(latencies),
        "iqr_seconds_per_image": q3 - q1,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "predictions_frozen_before_spatial_ground_truth": True,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
