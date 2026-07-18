from __future__ import annotations

import torch
from torch import nn


def resolve_gpu_count(requested: int) -> int:
    """Resolve 0=auto, 1=single GPU, 2=two GPUs, with clear failures."""
    if requested not in (0, 1, 2):
        raise ValueError("--num-gpus must be 0 (auto), 1, or 2")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested == 0:
        return min(available, 2)
    if available < requested:
        raise RuntimeError(
            f"--num-gpus {requested} requested, but only {available} CUDA device(s) are visible"
        )
    return requested


def prepare_data_parallel(model: nn.Module, num_gpus: int) -> tuple[nn.Module, torch.device]:
    """Move a model to CUDA/CPU and wrap it only when two GPUs are requested."""
    if num_gpus <= 0:
        device = torch.device("cpu")
        return model.to(device), device
    device = torch.device("cuda:0")
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    return model, device


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return checkpoint/CAM-compatible module without a DataParallel prefix."""
    return model.module if isinstance(model, nn.DataParallel) else model
