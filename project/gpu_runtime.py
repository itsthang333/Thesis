from __future__ import annotations

"""Resource-only CUDA placement shared by T4x2 and single-A100 runs."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CudaRuntime:
    primary_device: torch.device
    device_names: tuple[str, ...]
    encoder_data_parallel: bool


def require_cuda_runtime() -> CudaRuntime:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("This stage requires at least one visible CUDA device")
    names = tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )
    return CudaRuntime(
        primary_device=torch.device("cuda:0"),
        device_names=names,
        encoder_data_parallel=len(names) > 1,
    )


def place_frozen_encoder(module: nn.Module, runtime: CudaRuntime) -> nn.Module:
    module = module.to(runtime.primary_device).eval()
    if runtime.encoder_data_parallel:
        module = nn.DataParallel(
            module,
            device_ids=list(range(len(runtime.device_names))),
            output_device=0,
        ).eval()
    return module
