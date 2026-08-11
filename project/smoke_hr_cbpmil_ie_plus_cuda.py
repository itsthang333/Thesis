from __future__ import annotations

"""Worst-case CUDA/AMP optimizer-step preflight for HR-CBPMIL-IE+."""

import argparse
import json
import os
from pathlib import Path

import torch

from frozen_io import sha256_file
from models.hr_cbpmil_ie_plus import HRCBPMILIEPlus, check_finite, hr_cbpmil_loss

try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
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
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    image = torch.randn((2, 3, 640, 640), device=device)
    masks = torch.zeros((2, 243, 320, 320), dtype=torch.uint8, device=device)
    for batch in range(2):
        for index in range(243):
            side = 1 if index == 0 else 2 + (index % 62)
            y0 = (17 * index + 11 * batch) % (320 - side)
            x0 = (29 * index + 7 * batch) % (320 - side)
            masks[batch, index, y0 : y0 + side, x0 : x0 + side] = 1
    valid = torch.ones((2, 243), dtype=torch.bool, device=device)
    clusters = torch.arange(243, device=device, dtype=torch.int32)[None].expand(2, -1).clone()
    labels = torch.tensor([1, 0], device=device)
    classes = torch.tensor([8, 0], device=device)
    torch.cuda.reset_peak_memory_stats()
    telemetry: list[dict[str, object]] = []
    amp_skipped_steps = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        # Exercise the exact epoch-4 activation and the maximum scheduled weight.
        epoch_number = 4 if step <= args.steps // 2 else 7
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(image, masks, valid, clusters)
            losses = hr_cbpmil_loss(output, labels, classes, valid, epoch_number=epoch_number)
        for name, value in output.items():
            if torch.is_floating_point(value):
                check_finite(f"smoke_output_{name}", value)
        for name, value in losses.items():
            check_finite(f"smoke_loss_{name}", value)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        check_finite("smoke_gradient_norm", gradient_norm)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            amp_skipped_steps += 1
            raise FloatingPointError(f"AMP skipped smoke optimizer step {step}")
        if step == 1 or step % 10 == 0 or step == args.steps:
            telemetry.append(
                {
                    "step": step,
                    "epoch_number": epoch_number,
                    "loss": float(losses["total"].detach().cpu()),
                    "intra": float(losses["intra"].detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "max_abs_dense": float(output["dense_logits"].detach().abs().max().cpu()),
                    "max_abs_delta": float(
                        (output["dense_inside"] - output["dense_ring"]).detach().abs().max().cpu()
                    ),
                    "grad_scale": float(scaler.get_scale()),
                    "cpu_rss_bytes": current_rss_bytes(),
                    "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
                }
            )
    post_warmup = [row for row in telemetry if int(row["step"]) >= 20]
    if post_warmup:
        rss_growth = max(int(row["cpu_rss_bytes"]) for row in post_warmup) - min(
            int(row["cpu_rss_bytes"]) for row in post_warmup
        )
        reserved_growth = max(int(row["gpu_reserved_bytes"]) for row in post_warmup) - min(
            int(row["gpu_reserved_bytes"]) for row in post_warmup
        )
    else:
        rss_growth = 0
        reserved_growth = 0
    memory_stationary = rss_growth <= 768 * 1024**2 and reserved_growth <= 512 * 1024**2
    result = {
        "stage": "hr_cbpmil_ie_plus_cuda_amp_preflight_v2_1",
        "numerical_implementation": "hr_cbpmil_ie_plus_v2.1_amp_safe",
        "device": torch.cuda.get_device_name(0),
        "cuda_devices": torch.cuda.device_count(),
        "steps": args.steps,
        "batch_size": 2,
        "candidates_per_image": 243,
        "image_size": 640,
        "mask_size": 320,
        "loss": float(losses["total"].detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "amp_skipped_steps": amp_skipped_steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "all_outputs_finite": all(
            bool(torch.isfinite(value).all()) for value in output.values() if torch.is_floating_point(value)
        ),
        "memory_stationary": memory_stationary,
        "rss_growth_after_warmup_bytes": rss_growth,
        "gpu_reserved_growth_after_warmup_bytes": reserved_growth,
        "telemetry": telemetry,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if result["amp_skipped_steps"] or not result["all_outputs_finite"] or not memory_stationary:
        raise RuntimeError(f"CUDA/AMP preflight failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
