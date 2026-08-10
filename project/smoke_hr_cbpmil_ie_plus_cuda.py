from __future__ import annotations

"""Worst-case CUDA/AMP optimizer-step preflight for HR-CBPMIL-IE+."""

import argparse
import json
from pathlib import Path

import torch

from frozen_io import sha256_file
from models.hr_cbpmil_ie_plus import HRCBPMILIEPlus, hr_cbpmil_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requires a GPU")
    if sha256_file(args.classifier_checkpoint) != args.expected_classifier_sha256:
        raise ValueError("Classifier SHA-256 mismatch")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    model = HRCBPMILIEPlus(args.classifier_checkpoint).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    image = torch.randn((2, 3, 640, 640), device=device)
    masks = torch.zeros((2, 243, 320, 320), dtype=torch.uint8, device=device)
    for batch in range(2):
        for index in range(243):
            side = 2 + (index % 62)
            y0 = (17 * index + 11 * batch) % (320 - side)
            x0 = (29 * index + 7 * batch) % (320 - side)
            masks[batch, index, y0 : y0 + side, x0 : x0 + side] = 1
    valid = torch.ones((2, 243), dtype=torch.bool, device=device)
    clusters = torch.arange(243, device=device, dtype=torch.int32)[None].expand(2, -1).clone()
    labels = torch.tensor([1, 0], device=device)
    classes = torch.tensor([8, 0], device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(image, masks, valid, clusters)
        losses = hr_cbpmil_loss(output, labels, classes, valid, epoch_number=7)
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    result = {
        "stage": "hr_cbpmil_ie_plus_cuda_amp_preflight_v1",
        "device": torch.cuda.get_device_name(0),
        "cuda_devices": torch.cuda.device_count(),
        "batch_size": 2,
        "candidates_per_image": 243,
        "image_size": 640,
        "mask_size": 320,
        "loss": float(losses["total"].detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "amp_step_skipped": bool(scaler.get_scale() < previous_scale),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "all_outputs_finite": all(
            bool(torch.isfinite(value).all()) for value in output.values() if torch.is_floating_point(value)
        ),
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if result["amp_step_skipped"] or not result["all_outputs_finite"]:
        raise RuntimeError(f"CUDA/AMP preflight failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
