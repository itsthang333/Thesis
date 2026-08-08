from __future__ import annotations

"""Offline Kaggle entrypoint for one matched X4 fully-supervised seed.

Kaggle executes uploaded Python kernels as ``/kaggle/src/script.py``.  A staged
payload must therefore bind ``KERNEL_SEED`` explicitly; filename parsing is
retained only for local/staging validation.
Only canonical train polygons are opened by the underlying matched trainer;
outer validation and test remain closed during training and checkpoint choice.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


INPUT = Path("/kaggle/input")
SOURCE_COMMIT = "458ab52f145583fe97485a419a230c848f68b46d"
PROTOCOL_SHA256 = "ecf39ca2a45d3bae5f42689c0605b78c6e4db3968aadfbf82036a118d46f1824"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
INNER_SPLIT_SHA256 = "d641bc1c2847aefab28759f41f93ea2b1489d7ad1c33beb90f11d2403eb4115b"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
TRAINER_SHA256 = "e94a9f2b7587881d407a9048b4d86a390cf380403e23c9d13868e8e584188e65"
TARGET_IO_SHA256 = "fc82186e8530b41d8798ab9df1ce8bd347d017e933fe298f5d8f41ead906cada"
UNET_SHA256 = "b57f4c00c78640ad6aec2a9195192ab64ddfe8660362e6ac23373ffe3104f4ad"
LOSSES_SHA256 = "330b1d4eb536c078ea73f4ea68b1ba34ea3392704c104dd99b5f7950a8d51e7c"
KERNEL_SEED: int | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(name: str, expected: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if path.is_file() and sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {matches}")
    return matches[0]


def dataset_root() -> Path:
    roots = sorted({
        path.parent
        for path in INPUT.rglob("images")
        if path.is_dir() and (path.parent / "Annotations").is_dir()
    })
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one BTXRD root, found {roots}")
    return roots[0]


def seed_from_script_name(path: Path) -> int:
    match = re.search(r"seed(42|43|44)(?:\D|$)", path.stem.lower())
    if match is None:
        raise ValueError(f"X4 seed is absent from script filename: {path.name}")
    return int(match.group(1))


def resolve_seed(path: Path, kernel_seed: int | None = KERNEL_SEED) -> int:
    if kernel_seed is not None:
        if kernel_seed not in (42, 43, 44):
            raise ValueError(f"unsupported X4 kernel seed: {kernel_seed}")
        return kernel_seed
    return seed_from_script_name(path)


def main() -> None:
    started = time.perf_counter()
    seed = resolve_seed(Path(__file__))
    trainer = exact_file("train_x4_matched_student.py", TRAINER_SHA256)
    project = trainer.parent
    locked = {
        project / "x4_training_targets.py": TARGET_IO_SHA256,
        project / "models" / "unet.py": UNET_SHA256,
        project / "models" / "losses.py": LOSSES_SHA256,
    }
    for path, expected in locked.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"source dependency differs: {path}")
    split = exact_file("canonical_split_manifest_85511.csv", SPLIT_SHA256)
    inner = exact_file("x4_inner_split.csv", INNER_SPLIT_SHA256)
    resnet = exact_file("resnet18-f37072fd.pth", RESNET18_SHA256)
    data = dataset_root()
    os.environ.update({
        "PYTHONHASHSEED": str(seed),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MALLOC_ARENA_MAX": "2",
    })
    output = Path(f"/kaggle/working/x4_fully_supervised_student_seed{seed}")
    command = [
        sys.executable,
        str(trainer),
        "--arm", "fully_supervised",
        "--seed", str(seed),
        "--dataset-root", str(data),
        "--split-manifest", str(split),
        "--inner-split-manifest", str(inner),
        "--expected-inner-split-sha256", INNER_SPLIT_SHA256,
        "--resnet18-weight", str(resnet),
        "--num-workers", "0",
        "--multi-gpu",
        "--output-dir", str(output),
    ]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=project.parent, check=True)
    metadata = json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "stage": "x4_matched_student_training_v1",
        "arm": "fully_supervised",
        "seed": seed,
        "epochs_completed": 30,
        "outer_validation_images_used": 0,
        "split_manifest_sha256": SPLIT_SHA256,
        "inner_split_sha256": INNER_SPLIT_SHA256,
        "x4_protocol_sha256": PROTOCOL_SHA256,
        "target_freeze_sha256": None,
        "spatial_ground_truth_training": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": metadata.get(key), "expected": expected}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if differences:
        raise RuntimeError(f"fully-supervised terminal metadata differs: {differences}")
    for name, key in (
        ("best_student.pt", "best_checkpoint_sha256"),
        ("last_student.pt", "last_checkpoint_sha256"),
        ("training_history.csv", "training_history_sha256"),
    ):
        if sha256(output / name) != metadata[key]:
            raise RuntimeError(f"student artifact hash differs: {name}")
    archive = Path(shutil.make_archive(
        f"/kaggle/working/x4_fully_supervised_student_seed{seed}_bundle",
        "zip",
        root_dir="/kaggle/working",
        base_dir=output.name,
    ))
    receipt = {
        "schema_version": 1,
        "stage": "x4_fully_supervised_student_training_wrapper_v1",
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "inner_split_sha256": INNER_SPLIT_SHA256,
        "seed": seed,
        "best_checkpoint_sha256": metadata["best_checkpoint_sha256"],
        "last_checkpoint_sha256": metadata["last_checkpoint_sha256"],
        "training_archive_sha256": sha256(archive),
        "outer_validation_images_used": 0,
        "spatial_ground_truth_training": True,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt_path = Path(
        f"/kaggle/working/x4_fully_supervised_student_seed{seed}_receipt.json"
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(output)
    print(json.dumps({**receipt, "receipt_sha256": sha256(receipt_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
