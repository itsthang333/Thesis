from __future__ import annotations

"""Thin fail-closed Kaggle wrapper for an X4 YOLO evaluation contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INPUT = Path("/kaggle/input")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(name: str, expected: str | None = None) -> Path:
    matches = [path for path in INPUT.rglob(name) if path.is_file()]
    if expected is not None:
        matches = [path for path in matches if sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def validate_contract(contract: dict[str, object]) -> None:
    required_strings = (
        "runtime_manifest_sha256",
        "split_sha256",
        "training_bundle_name",
        "training_bundle_sha256",
        "training_receipt_name",
        "training_receipt_sha256",
        "runner_sha256",
        "freeze_runner_sha256",
        "evaluator_sha256",
        "output_prefix",
    )
    if (
        contract.get("schema_version") != 1
        or contract.get("stage") != "x4_yolo_kaggle_evaluation_contract_v1"
        or contract.get("test_images_read") != 0
        or contract.get("test_evaluated") is not False
        or not isinstance(contract.get("seed"), int)
        or any(not isinstance(contract.get(key), str) for key in required_strings)
    ):
        raise RuntimeError("X4 YOLO evaluation contract differs")


def main() -> None:
    os.environ.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPYCACHEPREFIX": "/kaggle/working/pycache",
            "WANDB_DISABLED": "true",
            "MALLOC_ARENA_MAX": "2",
        }
    )
    contract_path = exact_file("evaluation_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    runner = exact_file("run_x4_yolo_evaluation_kaggle.py", str(contract["runner_sha256"]))
    source_root = runner.parent.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "project"), str(source_root))
    )
    command = [
        sys.executable,
        str(runner),
        "--input-root",
        str(INPUT),
        "--expected-runtime-manifest-sha256",
        str(contract["runtime_manifest_sha256"]),
        "--expected-split-sha256",
        str(contract["split_sha256"]),
        "--training-bundle-name",
        str(contract["training_bundle_name"]),
        "--expected-training-bundle-sha256",
        str(contract["training_bundle_sha256"]),
        "--training-receipt-name",
        str(contract["training_receipt_name"]),
        "--expected-training-receipt-sha256",
        str(contract["training_receipt_sha256"]),
        "--expected-freeze-runner-sha256",
        str(contract["freeze_runner_sha256"]),
        "--expected-evaluator-sha256",
        str(contract["evaluator_sha256"]),
        "--seed",
        str(contract["seed"]),
        "--output-prefix",
        str(contract["output_prefix"]),
        "--device",
        "0",
        "--batch",
        "8",
    ]
    print(json.dumps({"contract_sha256": sha256(contract_path), "command": command}), flush=True)
    subprocess.run(command, cwd=source_root, env=environment, check=True)


if __name__ == "__main__":
    main()
