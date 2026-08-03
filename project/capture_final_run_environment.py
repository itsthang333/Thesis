from __future__ import annotations

"""Capture the exact software/GPU environment without opening dataset files."""

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command(*values: str) -> str:
    return subprocess.check_output(values, cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch

    devices: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    document = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": command("git", "rev-parse", "HEAD"),
            "git_branch": command("git", "branch", "--show-current"),
            "git_dirty": bool(command("git", "status", "--porcelain")),
        },
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "devices": devices,
        },
        "pip_freeze": command(sys.executable, "-m", "pip", "freeze").splitlines(),
        "dataset_files_opened": 0,
        "test_images_read": 0,
        "spatial_ground_truth_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cuda_devices": devices}, indent=2))


if __name__ == "__main__":
    main()
