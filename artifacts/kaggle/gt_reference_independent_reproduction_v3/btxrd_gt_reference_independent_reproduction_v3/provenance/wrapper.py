from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/btxrd_gt_reference_independent_reproduction_v3")
RUN_DIR = WORK / "fs_resnet18_pw10_full_448_seed42"
EVAL_DIR = WORK / "evaluation"
WORK.mkdir(parents=True, exist_ok=True)

EXPECTED_SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
EXPECTED_SOURCE_CANONICAL_LF_SHA256 = {
    "train_segmentation.py": "5c940cf86ff95d395b320d9f973b625d3d15b5296820606a654a404259891fd4",
    "models/unet.py": "aa867b3c95aeed4e906dd03203283bee7d5dd717e1144ba626bd41150c88bf64",
    "models/losses.py": "330b1d4eb536c078ea73f4ea68b1ba34ea3392704c104dd99b5f7950a8d51e7c",
    "datasets/btxrd.py": "e0b78250577804092549bce848476c55f1ed432530238c6a1b070642af720ae9",
    "evaluate_unet.py": "61cf37093f5e4d335d47d33c2d2e7e6c14cd712badc761afc27aeed30f0860eb",
    "evaluation/segmentation_metrics.py": "070ebb9f1092dac5fe87ff7c1acc7470d5834ce4e1dae0373b8ef60783cf314a",
}
REFERENCE = {
    "checkpoint_sha256": "05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31",
    "best_epoch": 20,
    "last_completed_epoch": 30,
    "selected_threshold": 0.2,
    "selected_mean_tumor_dice": 0.4951316962732512,
    "fixed_0_5_mean_tumor_dice": 0.489941358174933,
    "small_lt_1pct_mean_tumor_dice": 0.32895493248574226,
    "medium_1_to_5pct_mean_tumor_dice": 0.6624417783635557,
    "large_ge_5pct_mean_tumor_dice": 0.6937033565801355,
}
THRESHOLD_GRID = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def find_bundle_root() -> Path:
    matches = [
        path.parent
        for path in INPUT.rglob("split_manifest.csv")
        if (path.parent / "project" / "train_segmentation.py").is_file()
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(f"Expected one direct research bundle, found {matches}")
    archives = [
        path.parent
        for path in INPUT.rglob("split_manifest.csv")
        if (path.parent / "project.zip").is_file()
    ]
    if len(archives) != 1:
        raise RuntimeError(
            f"Expected one direct or archived research bundle, found {archives}"
        )
    extracted = WORK / "frozen_source_bundle"
    project = extracted / "project"
    project.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archives[0] / "project.zip") as archive:
        archive.extractall(project)
    shutil.copy2(archives[0] / "split_manifest.csv", extracted / "split_manifest.csv")
    return extracted.resolve()


def find_btxrd_root() -> Path:
    candidates = [
        INPUT / "btxrd-raw" / "BTXRD",
        INPUT / "datasets" / "itsthang333" / "btxrd-raw" / "BTXRD",
        *sorted(INPUT.glob("**/BTXRD")),
    ]
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "Annotations").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate BTXRD/images and BTXRD/Annotations")


def find_unique_file(name: str) -> Path:
    matches = sorted(INPUT.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r}, found {matches}")
    return matches[0]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


BUNDLE_ROOT = find_bundle_root()
PROJECT = BUNDLE_ROOT / "project"
SPLIT_MANIFEST = BUNDLE_ROOT / "split_manifest.csv"
DATA_ROOT = find_btxrd_root()
RESNET18_WEIGHT = find_unique_file("resnet18-f37072fd.pth")

if sha256(SPLIT_MANIFEST) != EXPECTED_SPLIT_SHA256:
    raise RuntimeError("Clean split manifest SHA-256 mismatch")
for relative, expected in EXPECTED_SOURCE_CANONICAL_LF_SHA256.items():
    actual = canonical_text_sha256(PROJECT / relative)
    if actual != expected:
        raise RuntimeError(
            f"Frozen source mismatch for {relative}: {actual} != {expected}"
        )

os.environ["PYTHONPATH"] = str(PROJECT)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_HOME"] = str(WORK / "torch")
weights_cache = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
weights_cache.mkdir(parents=True, exist_ok=True)
shutil.copy2(RESNET18_WEIGHT, weights_cache / RESNET18_WEIGHT.name)

import torch  # noqa: E402
import torchvision  # noqa: E402
import PIL  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle GPU is not available")

started_at = datetime.now(timezone.utc).isoformat()
start_monotonic = time.monotonic()
environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pillow": PIL.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_count": torch.cuda.device_count(),
    "cudnn_deterministic_before_training_process": (
        torch.backends.cudnn.deterministic
    ),
    "cudnn_benchmark_before_training_process": torch.backends.cudnn.benchmark,
    "training_source_sets_cudnn_deterministic": True,
    "training_source_sets_cudnn_benchmark_false": True,
    "torch_deterministic_algorithms_explicitly_enabled": (
        torch.are_deterministic_algorithms_enabled()
    ),
    "num_workers": 4,
    "multi_gpu_training": False,
    "data_root": str(DATA_ROOT),
    "split_manifest": str(SPLIT_MANIFEST),
    "split_sha256": sha256(SPLIT_MANIFEST),
    "reproduction_mode": "independent epoch-1 training with frozen seed-42 contract",
    "source_canonical_lf_sha256": EXPECTED_SOURCE_CANONICAL_LF_SHA256,
}
print(json.dumps(environment, indent=2), flush=True)

RUN_DIR.mkdir(parents=True, exist_ok=True)
train_command = [
    sys.executable, "-u", str(PROJECT / "train_segmentation.py"),
    "--pipeline-profile", "btxrd_best",
    "--data-root", str(DATA_ROOT),
    "--train-split", "train",
    "--val-split", "val",
    "--split-manifest", str(SPLIT_MANIFEST),
    "--image-size", "448",
    "--model-architecture", "resnet18_unet",
    "--batch-size", "8",
    "--lr", "0.0001",
    "--weight-decay", "0.0001",
    "--epochs", "35",
    "--seed", "42",
    "--num-workers", "4",
    "--output-dir", str(RUN_DIR),
    "--early-stop-patience", "10",
    "--checkpoint-dice-tolerance", "0.0001",
    "--pos-weight-mode", "manual",
    "--pos-weight-value", "10",
    "--pos-weight-clamp-min", "1",
    "--pos-weight-clamp-max", "20",
    "--pos-weight-fixed-reference", "10",
]
run_logged(train_command, RUN_DIR / "cloud_train.log")

BEST_CHECKPOINT = RUN_DIR / "best_unet.pt"
if not BEST_CHECKPOINT.is_file():
    raise FileNotFoundError(BEST_CHECKPOINT)

EVAL_DIR.mkdir(parents=True, exist_ok=True)
fixed_command = [
    sys.executable, "-u", str(PROJECT / "evaluate_unet.py"),
    "--data-root", str(DATA_ROOT),
    "--split", "val",
    "--split-manifest", str(SPLIT_MANIFEST),
    "--checkpoint", str(BEST_CHECKPOINT),
    "--image-size", "448",
    "--batch-size", "8",
    "--num-workers", "4",
    "--threshold", "0.5",
    "--threshold-grid", *[str(value) for value in THRESHOLD_GRID],
    "--output-csv", str(EVAL_DIR / "fixed_per_image.csv"),
    "--output-json", str(EVAL_DIR / "fixed_summary.json"),
    "--bootstrap-iterations", "5000",
    "--bootstrap-seed", "42",
]
run_logged(fixed_command, EVAL_DIR / "fixed_eval.log")

selection = json.loads(
    (EVAL_DIR / "fixed_per_image_threshold_selection.json").read_text(encoding="utf-8")
)
selected_threshold = float(selection["selected"]["threshold"])
selected_command = [
    sys.executable, "-u", str(PROJECT / "evaluate_unet.py"),
    "--data-root", str(DATA_ROOT),
    "--split", "val",
    "--split-manifest", str(SPLIT_MANIFEST),
    "--checkpoint", str(BEST_CHECKPOINT),
    "--image-size", "448",
    "--batch-size", "8",
    "--num-workers", "4",
    "--threshold", str(selected_threshold),
    "--output-csv", str(EVAL_DIR / "selected_per_image.csv"),
    "--output-json", str(EVAL_DIR / "selected_summary.json"),
    "--bootstrap-iterations", "5000",
    "--bootstrap-seed", "42",
]
run_logged(selected_command, EVAL_DIR / "selected_eval.log")

with (EVAL_DIR / "selected_per_image_subgroups.csv").open(
    "r", encoding="utf-8", newline=""
) as handle:
    subgroup_rows = list(csv.DictReader(handle))
lesion_size_rows = {
    row["subgroup_value"]: row
    for row in subgroup_rows
    if row["subgroup_field"] == "lesion_size"
    and row["subgroup_value"] in {
        "small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct"
    }
}
if {
    name: int(row["tumor_images"]) for name, row in lesion_size_rows.items()
} != {
    "small_lt_1pct": 94,
    "medium_1_to_5pct": 72,
    "large_ge_5pct": 18,
}:
    raise RuntimeError("Frozen lesion-size subgroup population mismatch")
with (RUN_DIR / "training_log.csv").open("r", encoding="utf-8", newline="") as handle:
    training_rows = list(csv.DictReader(handle))
best_state = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
fixed_summary = json.loads((EVAL_DIR / "fixed_summary.json").read_text(encoding="utf-8"))
selected_summary = json.loads((EVAL_DIR / "selected_summary.json").read_text(encoding="utf-8"))
if (
    int(selected_summary["images"]) != 371
    or int(selected_summary["tumor_images"]) != 184
    or int(selected_summary["normal_images"]) != 187
):
    raise RuntimeError("Independent reproduction validation population is incomplete")
candidate_subgroups = {
    name: float(row["mean_tumor_dice"])
    for name, row in lesion_size_rows.items()
}
comparison = {
    "checkpoint_sha256_equal": sha256(BEST_CHECKPOINT) == REFERENCE["checkpoint_sha256"],
    "best_epoch_delta": int(best_state["best_epoch"]) - REFERENCE["best_epoch"],
    "last_completed_epoch_delta": (
        int(training_rows[-1]["epoch"]) - REFERENCE["last_completed_epoch"]
    ),
    "selected_threshold_delta": selected_threshold - REFERENCE["selected_threshold"],
    "selected_mean_tumor_dice_delta": (
        float(selected_summary["mean_tumor_dice"])
        - REFERENCE["selected_mean_tumor_dice"]
    ),
    "fixed_0_5_mean_tumor_dice_delta": (
        float(fixed_summary["mean_tumor_dice"])
        - REFERENCE["fixed_0_5_mean_tumor_dice"]
    ),
    "small_lt_1pct_mean_tumor_dice_delta": (
        candidate_subgroups["small_lt_1pct"]
        - REFERENCE["small_lt_1pct_mean_tumor_dice"]
    ),
    "medium_1_to_5pct_mean_tumor_dice_delta": (
        candidate_subgroups["medium_1_to_5pct"]
        - REFERENCE["medium_1_to_5pct_mean_tumor_dice"]
    ),
    "large_ge_5pct_mean_tumor_dice_delta": (
        candidate_subgroups["large_ge_5pct"]
        - REFERENCE["large_ge_5pct_mean_tumor_dice"]
    ),
}
summary = {
    "protocol": (
        "independent epoch-1 GT training; clean validation only; test untouched; "
        "mean per-image Dice over all 184 tumor validation images"
    ),
    "started_at_utc": started_at,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": time.monotonic() - start_monotonic,
    "environment": environment,
    "training": {
        "start_epoch": 1,
        "last_completed_epoch": int(training_rows[-1]["epoch"]),
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_positive_dice_at_0_5": float(best_state["best_metric"]),
        "early_stop_patience": 10,
        "max_epoch": 35,
        "checkpoint_sha256": sha256(BEST_CHECKPOINT),
    },
    "fixed_0_5": fixed_summary,
    "selected_threshold": selected_threshold,
    "selected": selected_summary,
    "selected_lesion_size_subgroups": lesion_size_rows,
    "authoritative_reference": REFERENCE,
    "reference_comparison": comparison,
    "artifact_sha256": {
        "checkpoint": sha256(BEST_CHECKPOINT),
        "training_log": sha256(RUN_DIR / "training_log.csv"),
        "selected_per_image": sha256(EVAL_DIR / "selected_per_image.csv"),
        "selected_subgroups": sha256(EVAL_DIR / "selected_per_image_subgroups.csv"),
        "threshold_selection": sha256(
            EVAL_DIR / "fixed_per_image_threshold_selection.json"
        ),
    },
    "test_evaluated": False,
}
(WORK / "convergence_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2), flush=True)
