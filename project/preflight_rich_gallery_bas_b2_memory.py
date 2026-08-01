from __future__ import annotations

"""Bounded T4x2 memory preflight for the frozen 448-pixel BAS recipe.

This utility uses random tensors only.  It never opens BTXRD images, labels,
annotations, validation polygons, candidate scores, or test data.  Its sole
purpose is to determine the largest physical batch that fits the exact BAS
forward/backward path while preserving effective batch 32 by accumulation.
"""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import nn

from mae_reconstruction_io import sha256_file
from models.bas_candidate_localizer import (
    BASLossConfig,
    BASResNet50Localizer,
    bas_activation_suppression_loss,
)


EXPECTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
IMAGE_SIZE = 448
EFFECTIVE_BATCH = 32
PHYSICAL_BATCH_CANDIDATES = (32, 16, 8, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(args.pretrained_checkpoint) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("ImageNet ResNet-50 checkpoint hash mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("memory preflight requires exactly two CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"memory preflight requires T4 x2, got {device_names}")

    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)
    torch.use_deterministic_algorithms(True)
    state = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
    model = BASResNet50Localizer(pretrained=False, backbone_state_dict=state).cuda()
    parallel = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
        momentum=0.9,
        weight_decay=5.0e-4,
        nesterov=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    attempts: list[dict[str, object]] = []
    selected: int | None = None

    for physical_batch in PHYSICAL_BATCH_CANDIDATES:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        started = time.perf_counter()
        try:
            images = torch.randn(
                physical_batch,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                dtype=torch.float32,
            )
            labels = (torch.arange(physical_batch) % 2).long()
            images = images.cuda(non_blocking=False)
            labels = labels.cuda(non_blocking=False)
            with torch.cuda.amp.autocast(enabled=True):
                output = parallel(images, labels)
                full_ce = F.cross_entropy(output.class_logits, labels)
                foreground_ce = F.cross_entropy(output.foreground_logits, labels)
                bas = bas_activation_suppression_loss(
                    output,
                    labels,
                    config=BASLossConfig(area_weight=1.2),
                )
                loss = full_ce + 0.5 * foreground_ce + bas
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            attempt = {
                "physical_batch": physical_batch,
                "status": "PASS",
                "loss": float(loss.detach()),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": [
                    int(torch.cuda.max_memory_allocated(index)) for index in range(2)
                ],
                "peak_reserved_bytes": [
                    int(torch.cuda.max_memory_reserved(index)) for index in range(2)
                ],
            }
            attempts.append(attempt)
            selected = physical_batch
            del images, labels, output, full_ce, foreground_ce, bas, loss
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        except torch.OutOfMemoryError as error:
            attempts.append(
                {
                    "physical_batch": physical_batch,
                    "status": "OOM",
                    "error": str(error),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": [
                        int(torch.cuda.max_memory_allocated(index)) for index in range(2)
                    ],
                    "peak_reserved_bytes": [
                        int(torch.cuda.max_memory_reserved(index)) for index in range(2)
                    ],
                }
            )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    if selected is None:
        raise RuntimeError(f"no BAS physical batch fits T4x2: {attempts}")
    if EFFECTIVE_BATCH % selected:
        raise RuntimeError("selected physical batch cannot preserve effective batch 32")
    result = {
        "stage": "rich_gallery_bas_b2_memory_preflight_v1",
        "image_size": IMAGE_SIZE,
        "effective_batch": EFFECTIVE_BATCH,
        "selected_physical_batch": selected,
        "gradient_accumulation_steps": EFFECTIVE_BATCH // selected,
        "cuda_device_names": device_names,
        "attempts": attempts,
        "data_opened": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
