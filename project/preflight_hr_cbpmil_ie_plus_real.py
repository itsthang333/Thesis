from __future__ import annotations

"""Real-data AMP stress gate for HR-CBPMIL-IE+ v2.1.

This is a scientific-neutral implementation check.  It uses only canonical
training images, image labels and the already-frozen candidate gallery.  It
does not save weights and never opens validation annotations or test data.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from data.hr_cbpmil_bags import (
    HRCBPMILBagDataset,
    build_cluster_cache,
    collate_hr_cbpmil_bags,
    load_candidate_masks,
    load_cluster_cache,
)
from frozen_io import load_split_rows_without_annotations, sha256_file
from models.hr_cbpmil_ie_plus import HRCBPMILIEPlus, check_finite, hr_cbpmil_loss
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest

try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--train-pseudo-manifest-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--cluster-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    if resource is not None:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1024 if os.name != "nt" else 1)
    return 0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    if args.steps < 100 or args.steps > 200:
        raise ValueError("The frozen numerical stress gate requires 100-200 optimizer steps")
    if not torch.cuda.is_available():
        raise RuntimeError("Real preflight requires CUDA")
    if sha256_file(args.classifier_checkpoint) != args.expected_classifier_sha256:
        raise ValueError("Classifier SHA-256 mismatch")
    seed_all(args.seed)

    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="train",
    )
    candidate_rows, audit = validate_candidate_diagnostics_manifest(
        args.train_candidate_root,
        expected_image_names=[row["image_id"] for row in rows],
        split="train",
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
    )
    if audit.get("cohort") != "all":
        raise ValueError("Real preflight requires the exact all-image gallery")
    clusters = (
        load_cluster_cache(args.cluster_cache, candidate_rows)
        if args.cluster_cache.is_file()
        else build_cluster_cache(args.train_candidate_root, candidate_rows, args.cluster_cache)
    )
    dataset = HRCBPMILBagDataset(
        rows,
        dataset_root=args.dataset_root,
        candidate_root=args.train_candidate_root,
        candidate_rows=candidate_rows,
        cluster_cache=clusters,
        augment=False,
    )

    tumor_index = next(index for index, row in enumerate(rows) if int(row["tumor"]) == 1)
    normal_index = next(index for index, row in enumerate(rows) if int(row["tumor"]) == 0)
    maximum_index = max(
        range(len(rows)),
        key=lambda index: int(candidate_rows[Path(rows[index]["image_id"]).stem]["candidate_count"]),
    )
    tiny_index: int | None = None
    tiny_candidate_index: int | None = None
    tiny_area: int | None = None
    for index, row in enumerate(rows):
        stem = Path(row["image_id"]).stem
        masks = load_candidate_masks(args.train_candidate_root, candidate_rows[stem])
        areas = masks.sum(axis=(1, 2), dtype=np.int64)
        local = int(np.argmin(areas))
        area = int(areas[local])
        if tiny_area is None or area < tiny_area:
            tiny_index, tiny_candidate_index, tiny_area = index, local, area
        if area == 1:
            break
    if tiny_index is None or tiny_candidate_index is None or tiny_area != 1:
        raise RuntimeError("Real repaired one-pixel proposal was not found")

    def opposite(index: int) -> int:
        return normal_index if int(rows[index]["tumor"]) == 1 else tumor_index

    case_indices = [
        (tumor_index, normal_index),
        (tiny_index, opposite(tiny_index)),
        (maximum_index, opposite(maximum_index)),
    ]
    device = torch.device("cuda:0")
    batches: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for left, right in case_indices:
        batch = collate_hr_cbpmil_bags([dataset[left], dataset[right]])
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        batches.append(batch)
        case_rows.append(
            {
                "image_ids": list(batch["image_id"]),
                "candidate_counts": batch["candidate_valid"].sum(dim=1).cpu().tolist(),
                "labels": batch["binary_label"].cpu().tolist(),
            }
        )

    model = HRCBPMILIEPlus(args.classifier_checkpoint).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    torch.cuda.reset_peak_memory_stats()
    telemetry: list[dict[str, object]] = []
    for step in range(1, args.steps + 1):
        batch = batches[(step - 1) % len(batches)]
        epoch_number = 4 if step <= args.steps // 2 else 7
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(
                batch["image"],
                batch["candidate_masks"],
                batch["candidate_valid"],
                batch["cluster_ids"],
            )
            losses = hr_cbpmil_loss(
                output,
                batch["binary_label"],
                batch["class10_label"],
                batch["candidate_valid"],
                epoch_number=epoch_number,
            )
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        check_finite("real_preflight_gradient_norm", gradient_norm)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise FloatingPointError(f"AMP skipped real preflight step {step}")
        if step == 1 or step % 10 == 0 or step == args.steps:
            telemetry.append(
                {
                    "step": step,
                    "case": (step - 1) % len(batches),
                    "epoch_number": epoch_number,
                    "loss": float(losses["total"].detach().cpu()),
                    "intra": float(losses["intra"].detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "max_abs_dense": float(output["dense_logits"].detach().abs().max().cpu()),
                    "max_abs_delta": float(
                        (output["dense_inside"] - output["dense_ring"]).detach().abs().max().cpu()
                    ),
                    "cpu_rss_bytes": current_rss_bytes(),
                    "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "grad_scale": float(scaler.get_scale()),
                }
            )

    post_warmup = [row for row in telemetry if int(row["step"]) >= 20]
    rss_growth = max(int(row["cpu_rss_bytes"]) for row in post_warmup) - min(
        int(row["cpu_rss_bytes"]) for row in post_warmup
    )
    reserved_growth = max(int(row["gpu_reserved_bytes"]) for row in post_warmup) - min(
        int(row["gpu_reserved_bytes"]) for row in post_warmup
    )
    memory_stationary = rss_growth <= 768 * 1024**2 and reserved_growth <= 512 * 1024**2
    result = {
        "stage": "hr_cbpmil_ie_plus_real_amp_preflight_v2_1",
        "numerical_implementation": "hr_cbpmil_ie_plus_v2.1_amp_safe",
        "steps": args.steps,
        "cases": case_rows,
        "tiny_candidate": {
            "image_id": rows[tiny_index]["image_id"],
            "candidate_index": tiny_candidate_index,
            "area_pixels_320": tiny_area,
        },
        "maximum_candidate_count": max(
            int(row["candidate_count"]) for row in candidate_rows.values()
        ),
        "memory_stationary": memory_stationary,
        "rss_growth_after_warmup_bytes": rss_growth,
        "gpu_reserved_growth_after_warmup_bytes": reserved_growth,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "telemetry": telemetry,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if not memory_stationary:
        raise RuntimeError(f"Real preflight memory is not stationary: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
