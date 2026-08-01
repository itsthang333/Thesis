from __future__ import annotations

"""Offline Kaggle wrapper for the random-tensor BAS-B2 memory preflight."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
EXPECTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_unique(name: str) -> Path:
    matches = sorted(path for path in INPUT.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name}: {matches}")
    return matches[0]


def main() -> None:
    runner = find_unique("preflight_rich_gallery_bas_b2_memory.py")
    weight = find_unique("resnet50-11ad3fa6.pth")
    if sha256_file(weight) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("ImageNet weight SHA-256 mismatch")
    source_root = runner.parent.parent
    os.environ["PYTHONPATH"] = str(source_root)
    os.environ["PYTHONPYCACHEPREFIX"] = "/tmp/btxrd_bas_b2_pycache"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    output = WORKING / "rich_gallery_bas_b2_memory_preflight.json"
    command = [
        sys.executable,
        str(runner),
        "--pretrained-checkpoint",
        str(weight),
        "--output",
        str(output),
    ]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=source_root, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
