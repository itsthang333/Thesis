from __future__ import annotations

"""Freeze native-resolution X4 W2 S2C pseudo masks before spatial GT."""

import argparse
import csv
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.btxrd import resolve_btxrd_root
from datasets.s2c import BTXRDS2CDataset, collate_s2c_batch
from frozen_io import load_split_rows_without_annotations, locate_verified_image
from models.s2c import (
    load_s2c_checkpoint,
    normalize_positive_cam,
    select_cam_guided_proposals,
    select_cam_guided_segments,
)
from pseudo.manifest import sha256_file
from x4_contract import CANONICAL_SPLIT_SHA256, load_x4_protocol


EXPECTED = {"train": {"images": 2981, "normal": 1493, "tumor": 1488},
            "val": {"images": 371, "normal": 187, "tumor": 184}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--segment-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def multiscale_cam(
    model: torch.nn.Module,
    images: torch.Tensor,
    scales: tuple[float, ...],
    *,
    output_size: tuple[int, int],
) -> torch.Tensor:
    accumulated = None
    for scale in scales:
        scaled = images if scale == 1.0 else F.interpolate(
            images, scale_factor=scale, mode="bilinear", align_corners=False
        )
        logits = model(scaled)["tumor_cam_logits"]
        positive = torch.relu(F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        ))
        accumulated = positive if accumulated is None else accumulated + positive
    assert accumulated is not None
    return normalize_positive_cam(accumulated)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    if sha256_file(args.segment_cache / "sam_segment_manifest.csv") != args.expected_cache_manifest_sha256:
        raise ValueError("X4 S2C cache-manifest SHA-256 mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("X4 S2C checkpoint SHA-256 mismatch")
    protocol, protocol_sha = load_x4_protocol(args.repo_root)
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split=args.split,
        allow_test=False,
    )
    counts = {
        "images": len(rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
    }
    if counts != EXPECTED[args.split]:
        raise ValueError(f"X4 S2C cohort differs: {counts}")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, checkpoint = load_s2c_checkpoint(args.checkpoint, device=device)
    if checkpoint.get("checkpoint_role") != "fixed_epoch_snapshot":
        raise ValueError("X4 S2C must use a fixed terminal checkpoint")
    metadata = checkpoint.get("training_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("X4 S2C checkpoint lacks training metadata")
    if metadata.get("split_manifest_sha256") != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 S2C checkpoint split differs")
    if metadata.get("outer_validation_images_opened") is not False:
        raise ValueError("X4 S2C generator used outer validation during training")
    if int(checkpoint.get("epoch", -1)) != int(metadata.get("epochs", -2)):
        raise ValueError("X4 S2C checkpoint is not the terminal fixed epoch")
    scales = tuple(float(value) for value in metadata["cpm_scales"])

    dataset = BTXRDS2CDataset(
        root=args.dataset_root,
        split=args.split,
        split_manifest=args.split_manifest,
        segment_cache_dir=args.segment_cache,
        image_size=int(metadata["image_size"]),
        augment=False,
        normalization=str(metadata["normalization"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_s2c_batch,
        pin_memory=device.type == "cuda",
    )
    root = resolve_btxrd_root(args.dataset_root)
    canonical = {row["image_id"]: row for row in rows}
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=False)
    output_rows: dict[str, dict[str, object]] = {}
    inference_times: list[float] = []
    selected_images = selected_segments = tumor_empty = foreground_pixels = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_started = time.perf_counter()
        cam = multiscale_cam(
            model, images, scales,
            output_size=tuple(int(v) for v in batch["segments"].shape[-2:]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = float(time.perf_counter() - batch_started) / len(batch["image_name"])
        inference_times.extend([seconds] * len(batch["image_name"]))
        for index, image_id in enumerate(batch["image_name"]):
            row = canonical[str(image_id)]
            image_path = locate_verified_image(root, row)
            with Image.open(image_path) as source:
                width, height = source.size
            if int(row["tumor"]) == 0:
                selected = torch.zeros_like(cam[index], dtype=torch.bool)
                info = {"selected_ids": [], "reason": "known_image_label_normal"}
                source_name = "explicit_empty_normal"
            else:
                proposals = batch["proposal_masks"][index].to(device)
                proposal_quality = batch["proposal_quality"][index].to(device)
                if proposals.shape[0] > 0:
                    selected, info = select_cam_guided_proposals(
                        cam[index], proposals, proposal_quality,
                        image_is_tumor=True,
                        positive_threshold=float(metadata["cpm_positive_threshold"]),
                        min_positive_score=float(metadata["cpm_min_positive_score"]),
                        min_sam_quality=float(metadata["cpm_min_sam_quality"]),
                        top_k=int(metadata["cpm_top_k"]),
                    )
                    source_name = "s2c_cam_guided_overlapping_sam_proposal"
                else:
                    selected, info = select_cam_guided_segments(
                        cam[index], batch["segments"][index].to(device),
                        batch["quality"][index].to(device),
                        image_is_tumor=True,
                        positive_threshold=float(metadata["cpm_positive_threshold"]),
                        min_positive_score=float(metadata["cpm_min_positive_score"]),
                        min_sam_quality=float(metadata["cpm_min_sam_quality"]),
                        top_k=int(metadata["cpm_top_k"]),
                    )
                    source_name = "s2c_cam_guided_disjoint_sam_segment"
            square = selected.detach().cpu().numpy().astype(np.uint8) * 255
            native = np.asarray(Image.fromarray(square, mode="L").resize(
                (width, height), Image.Resampling.NEAREST
            )) > 0
            foreground = int(native.sum())
            chosen = len(info["selected_ids"])
            selected_images += int(chosen > 0)
            selected_segments += chosen
            tumor_empty += int(int(row["tumor"]) == 1 and foreground == 0)
            foreground_pixels += foreground
            relative = Path("masks") / f"{Path(str(image_id)).stem}.png"
            output_path = args.output_dir / relative
            Image.fromarray(native.astype(np.uint8) * 255, mode="L").save(output_path, optimize=True)
            output_rows[str(image_id)] = {
                "image_id": image_id,
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "mask_path": relative.as_posix(),
                "mask_height": height,
                "mask_width": width,
                "mask_foreground_pixels": foreground,
                "mask_sha256": sha256_file(output_path),
                "source": source_name,
                "selected_segment_count": chosen,
                "selection_reason": info["reason"],
            }

    if set(output_rows) != set(canonical):
        raise RuntimeError("X4 S2C output cohort is incomplete")
    ordered = [output_rows[row["image_id"]] for row in rows]
    manifest_path = args.output_dir / "x4_s2c_mask_manifest.csv"
    manifest_sha = write_manifest(manifest_path, ordered)
    q1, q3 = np.percentile(np.asarray(inference_times), [25, 75]).tolist()
    freeze = {
        "schema_version": 1,
        "stage": "x4_s2c_mask_freeze_v1",
        "study": protocol["study"],
        "split": args.split,
        "source_commit": args.source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_epoch": checkpoint["epoch"],
        "segment_cache_manifest_sha256": args.expected_cache_manifest_sha256,
        "generator": "binary DenseNet121-FPN stride4 + SSC + delayed CPM + SAM proposals",
        "manifest_sha256": manifest_sha,
        "images": counts["images"],
        "normal_images": counts["normal"],
        "tumor_images": counts["tumor"],
        "selected_images": selected_images,
        "selected_segments": selected_segments,
        "tumor_empty_masks": tumor_empty,
        "total_foreground_pixels": foreground_pixels,
        "native_resolution_masks": True,
        "normal_targets_explicitly_empty": True,
        "training_spatial_annotations_read": 0,
        "outer_validation_annotations_read": 0,
        "masks_frozen_before_outer_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "batch_size": args.batch_size,
        "time_per_image_seconds_median": float(statistics.median(inference_times)),
        "time_per_image_seconds_iqr": [float(q1), float(q3)],
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    freeze_path = args.output_dir / "x4_s2c_mask_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "freeze_sha256": sha256_file(freeze_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
