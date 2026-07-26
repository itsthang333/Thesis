from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
OUTPUT = WORK / "btxrd_rad_dino_affinity_decoder_probe_val_v1"
SOURCE_REPOSITORY = "https://github.com/itsthang333/Thesis.git"
CHECKOUT_COMMIT = "a716d059648924b5bb7ccf76f41549d4715ec89c"
SCIENTIFIC_SOURCE_COMMIT = "38b5bb4b9d7a846862443b442ff406f0ab41d3bd"
PROTOCOL_SHA256 = "53e2bb82ef35862b6c3e20387edbe60776f9d1ba46da516b9d5116db3fa2e7cf"
GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
FROZEN_SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
GIT_BASELINE_PER_IMAGE_SHA256 = (
    "f685e85b22ff5e3e48ecdf659d8f1c0f9f60cf13e9ffa69783305d4819aff8c9"
)
FROZEN_BASELINE_PER_IMAGE_SHA256 = (
    "c7bd20412913157b8d6f17b69ce4ed01495645a1c2a91b17a0a37166737f844c"
)
MODEL_ID = "microsoft/rad-dino"
MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
MODEL_HASHES = {
    "config.json": "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede",
    "preprocessor_config.json": "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56",
    "model.safetensors": "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae",
}
SOURCE_HASHES = {
    "run_rad_dino_affinity_decoder_probe.py": (
        "8b4f47c059ce91561b6678abcaef2f083cef35bfa0ea0c8d4d4ee9c30f5141df"
    ),
    "models/rad_dino_affinity_decoder.py": (
        "bb6583bb80a7c2f5a80519b859746fdfcb53fe78c0fbea40c04d07187f12f90a"
    ),
    "models/nominal_patch_memory.py": (
        "a7fbf3e4042623b1b2817f8bb02a87072db55c21bb5b9380cc259aca3a7dd526"
    ),
    "models/mae_reconstruction.py": (
        "7baa1a664d0523d454ab19bc1420959ef5115e31da2a242345c29f1dab7417f9"
    ),
    "mae_reconstruction_io.py": (
        "1b14cae95354db53fc2117254701920870510d4b8bb1005f1400340ad4c8859c"
    ),
    "generate_nominal_patch_memory_saliency.py": (
        "e4a454bbe26f1191f9b077d64affc58563919cc892bde71b585b72ebc140eee3"
    ),
    "compare_nominal_patch_memory_arms.py": (
        "4ec0b65c33d47dda20317b94d577489038bf87c8c235bb03d6d9a1d0497a035e"
    ),
    "datasets/btxrd.py": (
        "d8f0804be4e81cdb4d58e4673708c1067eb7d9b49b42bb78cb6051188c156001"
    ),
    "datasets/common.py": (
        "1927eb358a9db1a0e9c2571be5e222c3edd9d69814dfb4bc3375bd3f8593b98a"
    ),
}
TEST_HASH = "66c8634058666a63bb0178d2927453bce352a32d2447ffac3c4a3abc825a5ace"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def checkout_source() -> tuple[Path, Path, Path, Path]:
    repository = TEMP / "thesis_source"
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            SOURCE_REPOSITORY,
            str(repository),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "--detach", CHECKOUT_COMMIT],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != CHECKOUT_COMMIT:
        raise RuntimeError("Checkout commit mismatch")
    project = repository / "project"
    for relative, expected in SOURCE_HASHES.items():
        if sha256(project / relative) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    test_path = repository / "tests/test_rad_dino_affinity_decoder.py"
    if sha256(test_path) != TEST_HASH:
        raise RuntimeError("Affinity-decoder test hash mismatch")
    protocol = (
        repository
        / "artifacts/research_protocols/rad_dino_affinity_decoder_probe_val_v1.json"
    )
    if sha256(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("Protocol hash mismatch")
    parsed = json.loads(protocol.read_text(encoding="utf-8"))
    if (
        parsed["status"] != "predeclared_before_execution"
        or parsed["source"]["commit"] != SCIENTIFIC_SOURCE_COMMIT
        or parsed["weak_supervision_contract"]["consumer_trained"]
        or parsed["weak_supervision_contract"]["test_evaluated"]
    ):
        raise RuntimeError("Protocol contract mismatch")
    git_split = repository / "artifacts/data_audit/split_manifest.csv"
    if sha256(git_split) != GIT_SPLIT_SHA256:
        raise RuntimeError("Git split hash mismatch")
    split = TEMP / "frozen_split_manifest.csv"
    split.write_bytes(git_split.read_bytes().replace(b"\n", b"\r\n"))
    if sha256(split) != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen CRLF split hash mismatch")
    git_baseline = (
        repository
        / "artifacts/kaggle/nominal_patch_memory_probe_val_v1"
        / "single_scale/per_image.csv"
    )
    if sha256(git_baseline) != GIT_BASELINE_PER_IMAGE_SHA256:
        raise RuntimeError("Git nominal baseline hash mismatch")
    baseline = TEMP / "frozen_nominal_single_scale_per_image.csv"
    baseline.write_bytes(git_baseline.read_bytes().replace(b"\n", b"\r\n"))
    if sha256(baseline) != FROZEN_BASELINE_PER_IMAGE_SHA256:
        raise RuntimeError("Frozen CRLF nominal baseline hash mismatch")
    return project.resolve(), split.resolve(), protocol.resolve(), baseline.resolve()


def find_btxrd_root() -> Path:
    for candidate in [
        INPUT / "btxrd-raw/BTXRD",
        *sorted(INPUT.glob("**/BTXRD")),
    ]:
        if (candidate / "images").is_dir() and (candidate / "Annotations").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("BTXRD root not found")


def audit_split(split: Path) -> None:
    with split.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["eligible"] == "1"
        ]
    counts = {
        "train": sum(row["split"] == "train" for row in rows),
        "train_normal": sum(
            row["split"] == "train" and row["tumor"] == "0" for row in rows
        ),
        "train_tumor": sum(
            row["split"] == "train" and row["tumor"] == "1" for row in rows
        ),
        "validation": sum(row["split"] == "val" for row in rows),
        "validation_normal": sum(
            row["split"] == "val" and row["tumor"] == "0" for row in rows
        ),
        "validation_tumor": sum(
            row["split"] == "val" and row["tumor"] == "1" for row in rows
        ),
    }
    if counts != {
        "train": 2981,
        "train_normal": 1493,
        "train_tumor": 1488,
        "validation": 371,
        "validation_normal": 187,
        "validation_tumor": 184,
    }:
        raise RuntimeError(f"Split cohort mismatch: {counts}")


def download_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    model_dir = TEMP / "rad-dino"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        allow_patterns=list(MODEL_HASHES),
    )
    for name, expected in MODEL_HASHES.items():
        if sha256(model_dir / name) != expected:
            raise RuntimeError(f"Model hash mismatch: {name}")
    return model_dir


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    TEMP.mkdir(parents=True, exist_ok=True)
    log = TEMP / "affinity_decoder_execution.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["TOKENIZERS_PARALLELISM"] = "false"
    project, split, protocol, baseline = checkout_source()
    data = find_btxrd_root()
    audit_split(split)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "transformers==4.50.2",
        ],
        cwd=TEMP,
        env=env,
        log=log,
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_rad_dino_affinity_decoder.py",
            "-q",
        ],
        cwd=project.parent,
        env=env,
        log=log,
    )
    model = download_model_snapshot()
    run(
        [
            sys.executable,
            "run_rad_dino_affinity_decoder_probe.py",
            "--dataset-root",
            str(data),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            FROZEN_SPLIT_SHA256,
            "--model-dir",
            str(model),
            "--expected-config-sha256",
            MODEL_HASHES["config.json"],
            "--expected-preprocessor-sha256",
            MODEL_HASHES["preprocessor_config.json"],
            "--expected-weight-sha256",
            MODEL_HASHES["model.safetensors"],
            "--baseline-per-image",
            str(baseline),
            "--expected-baseline-per-image-sha256",
            FROZEN_BASELINE_PER_IMAGE_SHA256,
            "--source-commit",
            SCIENTIFIC_SOURCE_COMMIT,
            "--protocol-sha256",
            PROTOCOL_SHA256,
            "--output-dir",
            str(OUTPUT),
            "--scratch-dir",
            str(TEMP / "rad_dino_affinity_scratch"),
        ],
        cwd=project,
        env=env,
        log=log,
    )
    run_manifest_path = OUTPUT / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        run_manifest["source_commit"] != SCIENTIFIC_SOURCE_COMMIT
        or run_manifest["protocol_sha256"] != PROTOCOL_SHA256
        or run_manifest["cohort"]
        != {
            "train": 2981,
            "train_normal": 1493,
            "train_tumor": 1488,
            "validation": 371,
            "validation_tumor": 184,
            "validation_normal": 187,
        }
        or not run_manifest["validation_gt_read_only_after_prediction_freeze"]
        or run_manifest["consumer_trained"]
        or run_manifest["test_evaluated"]
    ):
        raise RuntimeError("Completed run manifest violates frozen contract")
    prediction_maps = list((OUTPUT / "predictions/maps").glob("*.npy"))
    if len(prediction_maps) != 371:
        raise RuntimeError("Completed output lacks 371 prediction maps")
    wrapper_path = Path(__file__).resolve()
    wrapper_provenance = {
        "run_id": OUTPUT.name,
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "wrapper_sha256": sha256(wrapper_path),
        "split_sha256": sha256(split),
        "baseline_per_image_sha256": sha256(baseline),
        "model_hashes": MODEL_HASHES,
        "run_manifest_sha256": sha256(run_manifest_path),
        "prediction_maps": len(prediction_maps),
        "consumer_trained": False,
        "test_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "wrapper_provenance.json").write_text(
        json.dumps(wrapper_provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(log, OUTPUT / "execution.log")
    print(json.dumps(wrapper_provenance, indent=2))


if __name__ == "__main__":
    main()
