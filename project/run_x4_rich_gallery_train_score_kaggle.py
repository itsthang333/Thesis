from __future__ import annotations

"""Freeze G1 choices for the 2,981-image X4 train cohort on Kaggle.

This reuses the exact Geometry-v3 candidate gallery and fixed G1 checkpoint.
It performs inference only and never opens BTXRD polygons or test images.  The
result supplies the missing train-time choice manifest required by the matched
Rich-Gallery pseudo-U-Net arm.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SOURCE_COMMIT = "458ab52f145583fe97485a419a230c848f68b46d"
PROTOCOL_SHA256 = "66b56661ca91052c6bb1958c92d433cfcf9303f07624423068109e0b90eda454"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
TRAIN_CANDIDATE_MANIFEST_SHA256 = "e260be427d3a35d1b6305f17cc8e2e3ed53eb92641a9f19e6cfa6c8b10f8a436"
TRAIN_PSEUDO_MANIFEST_SHA256 = "649ee4232bbcca930c099e888708fa6894a34229ce08e1b80a17446c745a1f13"
G1_CHECKPOINT_SHA256 = "634e1200330e87692fab4a2e35ba70806790937d7b19ed8b0a3c4968471bfe8c"
RAD_CONFIG_SHA256 = "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede"
RAD_PREPROCESSOR_SHA256 = "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56"
RAD_WEIGHT_SHA256 = "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae"
TRANSFORMERS_WHEEL_SHA256 = "99bbcddd6570f080aee81f67844f4b46c8025bbdbdb86eafb82cc7d6aaafb190"
TOKENIZERS_WHEEL_SHA256 = "51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff"
HUB_WHEEL_SHA256 = "7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d"
SCORER_SHA256 = "c5b634a71135f6fd49a75b58f52986f374f59696e4b3c2384e44432a43831fb2"
PROBE_SHA256 = "5a2c95db2d73b3544ac21b01385c27ae3514d6a3a0e6abe31798af167a5307b3"
G1_MODEL_SHA256 = "c82fa61c1e9a33f969ffb172c5099aedbbb70d5f28a323cfd638808d3974967f"
NOMINAL_MEMORY_SHA256 = "a7fbf3e4042623b1b2817f8bb02a87072db55c21bb5b9380cc259aca3a7dd526"
GPU_RUNTIME_SHA256 = "15c159d253969c1396f597281d59fd2f5e473d82435f937e7076e3b3e500d5b2"
FROZEN_IO_SHA256 = "423c5b9eef87a59d9f457bc4acc1f6795cff9c4b1c875f25d651b5aa88987a2d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(name: str, expected: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if path.is_file() and sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name} with SHA {expected}, found {matches}")
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


def install_runtime() -> None:
    wheels = [
        exact_file("transformers-4.50.2-py3-none-any.whl", TRANSFORMERS_WHEEL_SHA256),
        exact_file(
            "tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            TOKENIZERS_WHEEL_SHA256,
        ),
        exact_file("huggingface_hub-0.36.0-py3-none-any.whl", HUB_WHEEL_SHA256),
    ]
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        *[str(path) for path in wheels],
    ]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=WORKING, check=True)


def main() -> None:
    started = time.perf_counter()
    os.environ.update({
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    install_runtime()
    scorer = exact_file("score_final_rich_gallery.py", SCORER_SHA256)
    project = scorer.parent
    locked = {
        project / "run_rad_dino_mask_bag_mil_probe.py": PROBE_SHA256,
        project / "models" / "rad_dino_mask_bag_mil.py": G1_MODEL_SHA256,
        project / "models" / "nominal_patch_memory.py": NOMINAL_MEMORY_SHA256,
        project / "gpu_runtime.py": GPU_RUNTIME_SHA256,
        project / "frozen_io.py": FROZEN_IO_SHA256,
    }
    for path, expected in locked.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"locked scoring dependency differs: {path}")
    split = exact_file("canonical_split_manifest_85511.csv", SPLIT_SHA256)
    candidate_manifest = exact_file(
        "candidate_diagnostics_manifest.csv", TRAIN_CANDIDATE_MANIFEST_SHA256
    )
    candidate_root = candidate_manifest.parent
    checkpoint = exact_file("rad_dino_mask_bag_mil.pt", G1_CHECKPOINT_SHA256)
    model_weight = exact_file("model.safetensors", RAD_WEIGHT_SHA256)
    model_dir = model_weight.parent
    if sha256(model_dir / "config.json") != RAD_CONFIG_SHA256:
        raise RuntimeError("Rad-DINO config differs")
    if sha256(model_dir / "preprocessor_config.json") != RAD_PREPROCESSOR_SHA256:
        raise RuntimeError("Rad-DINO preprocessor differs")
    output = WORKING / "x4_rich_gallery_train_g1_scores"
    command = [
        sys.executable,
        str(scorer),
        "--split", "train",
        "--dataset-root", str(dataset_root()),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA256,
        "--model-dir", str(model_dir),
        "--expected-config-sha256", RAD_CONFIG_SHA256,
        "--expected-preprocessor-sha256", RAD_PREPROCESSOR_SHA256,
        "--expected-weight-sha256", RAD_WEIGHT_SHA256,
        "--candidate-root", str(candidate_root),
        "--candidate-manifest-sha256", TRAIN_CANDIDATE_MANIFEST_SHA256,
        "--pseudo-manifest-sha256", TRAIN_PSEUDO_MANIFEST_SHA256,
        "--g1-checkpoint", str(checkpoint),
        "--expected-g1-checkpoint-sha256", G1_CHECKPOINT_SHA256,
        "--source-commit", SOURCE_COMMIT,
        "--protocol-sha256", PROTOCOL_SHA256,
        "--output-dir", str(output),
        "--encoder-batch-size", "8",
        "--maximum-candidates", "243",
        "--seed", "42",
    ]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=project.parent, check=True)
    freeze_path = output / "diagnostic_freeze.json"
    evidence_path = output / "descriptor_evidence_manifest.csv"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "stage": "rich_gallery_g1_all_candidate_score_freeze_v1",
        "cohort_split": "train",
        "split_sha256": SPLIT_SHA256,
        "baseline_checkpoint_sha256": G1_CHECKPOINT_SHA256,
        "candidate_manifest_sha256": TRAIN_CANDIDATE_MANIFEST_SHA256,
        "pseudo_manifest_sha256": TRAIN_PSEUDO_MANIFEST_SHA256,
        "images": 2981,
        "validation_images": 0,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    differences = {
        key: {"actual": freeze.get(key), "expected": expected}
        for key, expected in required.items()
        if freeze.get(key) != expected
    }
    if differences:
        raise RuntimeError(f"train G1 score freeze differs: {differences}")
    if freeze.get("descriptor_evidence_manifest_sha256") != sha256(evidence_path):
        raise RuntimeError("train G1 evidence manifest differs")
    archive = Path(shutil.make_archive(
        str(WORKING / "x4_rich_gallery_train_g1_scores"),
        "zip",
        root_dir=WORKING,
        base_dir=output.name,
    ))
    receipt = {
        "schema_version": 1,
        "stage": "x4_rich_gallery_train_g1_score_wrapper_v1",
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "candidate_manifest_sha256": TRAIN_CANDIDATE_MANIFEST_SHA256,
        "pseudo_manifest_sha256": TRAIN_PSEUDO_MANIFEST_SHA256,
        "g1_checkpoint_sha256": G1_CHECKPOINT_SHA256,
        "evidence_manifest_sha256": sha256(evidence_path),
        "freeze_sha256": sha256(freeze_path),
        "archive_sha256": sha256(archive),
        "images": 2981,
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt_path = WORKING / "x4_rich_gallery_train_g1_scores_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(output)
    print(json.dumps({**receipt, "receipt_sha256": sha256(receipt_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
