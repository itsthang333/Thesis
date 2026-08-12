from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path

from btxrd_wsss.config import PipelineConfig


def run_preflight(config: PipelineConfig, *, require_assets: bool = False) -> dict[str, object]:
    package_names = (
        "numpy",
        "PIL",
        "scipy",
        "sklearn",
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "cv2",
        "open_clip",
        "segment_anything",
    )
    packages = {name: importlib.util.find_spec(name) is not None for name in package_names}
    torch = None
    if packages["torch"]:
        import torch as torch_package

        torch = torch_package
    gpus = []
    if torch is not None:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(properties.total_memory / 2**30, 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    assets = {
        "data_root": Path(config.data.root).exists(),
        "manifest": Path(config.data.manifest).exists(),
        "sam_checkpoint": Path(config.sam.checkpoint).exists(),
    }
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "packages": packages,
        "assets": assets,
        "rtx_5090_ready": bool(
            len(gpus) == 1 and "5090" in gpus[0]["name"] and gpus[0]["vram_gib"] >= 30
        ),
    }
    if require_assets and (
        not all(packages.values()) or not all(assets.values()) or not report["rtx_5090_ready"]
    ):
        raise RuntimeError(json.dumps(report, indent=2))
    return report
