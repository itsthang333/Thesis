from __future__ import annotations

"""Freeze annotation-free LayerCAM masks for the X4 CAM baseline.

The generator is deliberately narrower than the SAM pseudo-mask pipeline.  A
hash-locked one-logit DenseNet-121 checkpoint is evaluated at 320 pixels and
the tumor-logit LayerCAM is binarized by the frozen within-image percentile
rule.  Normal images are explicit empty masks.  No spatial annotation is read.

The same entrypoint supports canonical train (student targets) and validation
(direct pseudo-mask evaluation).  Validation masks are frozen before a later,
separate evaluator is allowed to open validation polygons.
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.btxrd import resolve_btxrd_root
from datasets.common import make_classification_transform
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
)
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from x4_contract import CANONICAL_SPLIT_SHA256, load_x4_protocol


EXPECTED = {
    "train": {"images": 2981, "tumor": 1488, "normal": 1493},
    "val": {"images": 371, "tumor": 184, "normal": 187},
}
IMAGE_SIZE = 320
CAM_PERCENTILE = 90.0
LAYER_WEIGHTS = (0.2, 0.3, 0.5)


def percentile_cam_mask(values: np.ndarray, percentile: float = CAM_PERCENTILE) -> np.ndarray:
    """Apply the frozen G4/X4 CAM rule without silently filling constant maps."""
    cam = np.asarray(values, dtype=np.float32)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("LayerCAM must be one finite 2-D map")
    if not 0.0 < percentile < 100.0:
        raise ValueError("CAM percentile must lie strictly inside (0,100)")
    if float(cam.max()) - float(cam.min()) <= 1.0e-8:
        return np.zeros(cam.shape, dtype=bool)
    threshold = float(np.percentile(cam, percentile))
    return cam >= threshold


def resize_binary_native(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or width <= 0 or height <= 0:
        raise ValueError("invalid CAM mask/native geometry")
    if binary.shape == (height, width):
        return binary
    return np.asarray(
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0


class TumorImageDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]]) -> None:
        if any(int(row["tumor"]) != 1 for row in rows):
            raise ValueError("TumorImageDataset may contain tumor rows only")
        self.root = root
        self.rows = rows
        self.transform = make_classification_transform(
            IMAGE_SIZE, augment=False, preprocessing_mode="none", normalization="imagenet"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = locate_verified_image(self.root, row)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            tensor = self.transform(image)
        return tensor, row["image_id"], width, height


def checkpoint_model(
    path: Path,
    *,
    expected_sha256: str,
    expected_seed: int,
    device: torch.device,
) -> tuple[DenseNet121AnatomyClassifier, dict[str, object]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("X4 CAM checkpoint SHA-256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "target_columns": ["tumor"],
        "task": "multi-label",
        "num_classes": 1,
        "image_size": IMAGE_SIZE,
        "normalization": "imagenet",
        "split_manifest_sha256": CANONICAL_SPLIT_SHA256,
        "seed": expected_seed,
    }
    for key, expected in required.items():
        if state.get(key) != expected:
            raise ValueError(
                f"X4 CAM checkpoint metadata differs for {key}: "
                f"{state.get(key)!r} != {expected!r}"
            )
    model = DenseNet121AnatomyClassifier(num_classes=1, pretrained=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, {
        "checkpoint_sha256": actual,
        "checkpoint_seed": int(state["seed"]),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_selection_metric": state.get("checkpoint_selection_metric"),
        "checkpoint_best_metric": float(state["best_metric"]),
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-seed", type=int, default=42)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid CAM data-loader configuration")
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    protocol, protocol_sha = load_x4_protocol(args.repo_root)
    root = resolve_btxrd_root(args.dataset_root)
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split=args.split,
        allow_test=False,
    )
    counts = {
        "images": len(rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
    }
    if counts != EXPECTED[args.split]:
        raise ValueError(f"X4 CAM cohort differs: {counts}")

    requested = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested.type == "cuda":
        torch.cuda.reset_peak_memory_stats(requested)
    model, checkpoint = checkpoint_model(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_seed=args.expected_checkpoint_seed,
        device=requested,
    )
    layercam = LayerCAM(
        model, device=requested, layer_weights=LAYER_WEIGHTS, gradient_mode="positive"
    )

    mask_root = args.output_dir / "masks"
    mask_root.mkdir(parents=True, exist_ok=False)
    output_by_id: dict[str, dict[str, object]] = {}
    constant_maps = 0
    foreground_pixels = 0
    inference_seconds: list[float] = []

    # Materialize every normal target explicitly without running the classifier.
    for row in rows:
        if int(row["tumor"]) != 0:
            continue
        image_path = locate_verified_image(root, row)
        with Image.open(image_path) as handle:
            width, height = handle.size
        native = np.zeros((height, width), dtype=np.uint8)
        relative = Path("masks") / f"{Path(row['image_id']).stem}.png"
        output = args.output_dir / relative
        Image.fromarray(native, mode="L").save(output, optimize=True)
        output_by_id[row["image_id"]] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "tumor": row["tumor"],
            "mask_path": relative.as_posix(),
            "mask_height": height,
            "mask_width": width,
            "mask_foreground_pixels": 0,
            "mask_sha256": sha256_file(output),
            "source": "explicit_empty_normal",
            "cam_percentile": CAM_PERCENTILE,
        }

    tumor_rows = [row for row in rows if int(row["tumor"]) == 1]
    tumor_by_id = {row["image_id"]: row for row in tumor_rows}
    loader = DataLoader(
        TumorImageDataset(root, tumor_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=requested.type == "cuda",
    )
    try:
        for tensors, image_ids, widths, heights in loader:
            tensors = tensors.to(requested, non_blocking=True)
            if requested.type == "cuda":
                torch.cuda.synchronize(requested)
            batch_started = time.perf_counter()
            result = layercam.cam_for_class(tensors, class_index=0)
            if requested.type == "cuda":
                torch.cuda.synchronize(requested)
            batch_seconds = float(time.perf_counter() - batch_started)
            inference_seconds.extend([batch_seconds / len(image_ids)] * len(image_ids))
            cams = result.cam.detach().cpu().numpy()
            for index, image_id in enumerate(image_ids):
                cam = cams[index]
                constant = float(cam.max()) - float(cam.min()) <= 1.0e-8
                constant_maps += int(constant)
                square = percentile_cam_mask(cam)
                width, height = int(widths[index]), int(heights[index])
                native = resize_binary_native(square, width=width, height=height)
                positive = int(native.sum())
                foreground_pixels += positive
                relative = Path("masks") / f"{Path(image_id).stem}.png"
                output = args.output_dir / relative
                Image.fromarray(native.astype(np.uint8) * 255, mode="L").save(
                    output, optimize=True
                )
                row = tumor_by_id[image_id]
                output_by_id[image_id] = {
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "tumor": row["tumor"],
                    "mask_path": relative.as_posix(),
                    "mask_height": height,
                    "mask_width": width,
                    "mask_foreground_pixels": positive,
                    "mask_sha256": sha256_file(output),
                    "source": "binary_densenet121_320_layercam_p90",
                    "cam_percentile": CAM_PERCENTILE,
                }
    finally:
        layercam.close()

    if set(output_by_id) != {row["image_id"] for row in rows}:
        raise RuntimeError("X4 CAM output cohort is incomplete")
    manifest_rows = [output_by_id[row["image_id"]] for row in rows]
    manifest_path = args.output_dir / "x4_cam_mask_manifest.csv"
    manifest_sha = write_manifest(manifest_path, manifest_rows)
    output_bytes = int(
        sum(path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file())
    )
    q1, q3 = (
        np.percentile(np.asarray(inference_seconds, dtype=np.float64), [25, 75]).tolist()
        if inference_seconds
        else [0.0, 0.0]
    )
    freeze = {
        "schema_version": 1,
        "stage": "x4_cam_mask_freeze_v1",
        "study": protocol["study"],
        "split": args.split,
        "source_commit": args.source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        **checkpoint,
        "generator": "DenseNet121/320 positive-gradient LayerCAM",
        "layer_weights": list(LAYER_WEIGHTS),
        "cam_percentile": CAM_PERCENTILE,
        "constant_map_rule": "empty",
        "manifest_sha256": manifest_sha,
        "images": counts["images"],
        "tumor_images": counts["tumor"],
        "normal_images": counts["normal"],
        "constant_tumor_maps": constant_maps,
        "total_foreground_pixels": foreground_pixels,
        "native_resolution_masks": True,
        "normal_targets_explicitly_empty": True,
        "train_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "masks_frozen_before_outer_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
        "device": str(requested),
        "hardware": torch.cuda.get_device_name(requested) if requested.type == "cuda" else platform.processor(),
        "batch_size": args.batch_size,
        "timed_tumor_images": len(inference_seconds),
        "cam_time_per_image_seconds_median": (
            float(statistics.median(inference_seconds)) if inference_seconds else 0.0
        ),
        "cam_time_per_image_seconds_iqr": [float(q1), float(q3)],
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(requested)) if requested.type == "cuda" else 0
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_bytes_before_freeze": output_bytes,
    }
    freeze_path = args.output_dir / "x4_cam_mask_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**freeze, "x4_cam_mask_freeze_sha256": sha256_file(freeze_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
