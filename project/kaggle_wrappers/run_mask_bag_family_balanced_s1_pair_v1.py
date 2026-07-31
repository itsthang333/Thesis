from __future__ import annotations

"""Fail-closed Kaggle bootstrap for the predeclared S1 matched pair."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-family-balanced-s1-pair-v1"
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "f3da1817ee3491f04e8c86335556762ebc675d8d"
PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1.json"
)
PROTOCOL_SHA256 = "62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_WRAPPER_AUDIT_SHA256 = "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
BASELINE_ARCHIVE_SHA256 = "8857eb6d1393683a21efaed3e3f33dee763e32203ac7665a76ff9fd809eee0c4"
TRANSPORT_AUDIT_SHA256 = "9377ce5bb86e5d3aea32117b1f1f338824cfa94ff6e655f543489733204bb3da"
BASELINE = {
    "freeze": "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3",
    "checkpoint": "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069",
    "manifest": "a810e1fcc4c4422d207eb020a70313caf5d3402bf30c277331247a30555678ee",
    "source_commit": "fda732941664e67d4b87a8c3cba071b6979b2214",
    "protocol": "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555",
}
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE = WORK / "s1_source"
RUNTIME = WORK / "s1_runtime"
OUTPUT = WORK / "btxrd_mask_bag_family_balanced_s1_pair_v1"
POOL_MODES = ("standard", "family_balanced")


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    print(f"$ {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def unique(name: str, expected_hash: str) -> Path:
    matches = [
        path
        for path in INPUT.rglob(name)
        if path.is_file() and hash_file(path) == expected_hash
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}/{expected_hash}, found {matches}")
    return matches[0]


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Archive member escapes destination: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Archive contains symlink: {member.filename}")
        archive.extractall(destination)


def clone_and_verify() -> dict[str, str]:
    if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1:
        raise RuntimeError("S1 launch binding is not frozen")
    if len(CHECKOUT_COMMIT) != 40 or any(
        character not in "0123456789abcdef" for character in CHECKOUT_COMMIT
    ):
        raise RuntimeError("Invalid bound S1 checkout")
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY,
            str(SOURCE),
        ],
        cwd=WORK,
    )
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, CHECKOUT_COMMIT], cwd=SOURCE)
    protocol_path = SOURCE / PROTOCOL_RELATIVE
    if hash_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("S1 protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hashes = protocol.get("scientific_source", {}).get(
        "canonical_lf_source_hashes", {}
    )
    if len(hashes) != 12:
        raise RuntimeError("S1 protocol source closure mismatch")
    for relative, expected in hashes.items():
        path = SOURCE / relative
        if not path.is_file() or canonical_hash(path) != expected:
            raise RuntimeError(f"S1 source hash mismatch: {relative}")
    return hashes


def install_runtime() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
            "torch==2.5.1+cu121",
            "torchvision==0.20.1+cu121",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        cwd=SOURCE,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "transformers==4.50.2",
        ],
        cwd=SOURCE,
    )


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S1 requires exactly two CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"S1 requires T4 x2, got {names}")
    checksums = []
    for index in range(2):
        layer = torch.nn.Conv2d(3, 5, 3, padding=1).to(f"cuda:{index}").eval()
        values = (
            torch.arange(3072, dtype=torch.float32, device=f"cuda:{index}")
            .reshape(1, 3, 32, 32)
            / 1024
        )
        with torch.inference_mode():
            output = layer(values)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"Non-finite convolution on cuda:{index}")
        checksums.append(float(output.sum().cpu()))
    return {
        "cuda_device_count": 2,
        "cuda_device_names": names,
        "real_convolution_checksums": checksums,
    }


def prepare_split() -> Path:
    source = SOURCE / "artifacts/kaggle/wsl_source_consensus_val_v1/frozen_split_manifest.csv"
    if hash_file(source) != GIT_SPLIT_SHA256 or b"\r" in source.read_bytes():
        raise RuntimeError("S1 canonical split mismatch")
    target = RUNTIME / "frozen_split_manifest.csv"
    target.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    if hash_file(target) != SPLIT_SHA256:
        raise RuntimeError("S1 frozen split reconstruction mismatch")
    return target


def prepare_baseline() -> tuple[Path, dict[str, object]]:
    audit_path = unique("transport_audit.json", TRANSPORT_AUDIT_SHA256)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("validation_gt_included") is not False
        or audit.get("consumer_trained") is not False
        or audit.get("test_evaluated") is not False
    ):
        raise RuntimeError("S1 baseline transport safety mismatch")
    archive = unique("square_corrected_baseline.zip.bin", BASELINE_ARCHIVE_SHA256)
    extracted = RUNTIME / "baseline"
    safe_extract(archive, extracted)
    roots = []
    for freeze in extracted.rglob("prediction_freeze.json"):
        root = freeze.parent
        if (
            hash_file(freeze) == BASELINE["freeze"]
            and hash_file(root / "rad_dino_mask_bag_mil.pt") == BASELINE["checkpoint"]
            and hash_file(root / "predictions/prediction_manifest.csv")
            == BASELINE["manifest"]
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S1 baseline root, found {roots}")
    return roots[0], {
        "baseline_archive_sha256": hash_file(archive),
        "transport_audit_sha256": hash_file(audit_path),
    }


def find_cache() -> tuple[Path, dict[str, object]]:
    freeze_path = unique("selector_cache_freeze.json", CACHE_FREEZE_SHA256)
    root = freeze_path.parent.resolve()
    audit_path = root / "wrapper_output_audit.json"
    if hash_file(audit_path) != CACHE_WRAPPER_AUDIT_SHA256:
        raise RuntimeError("S1 cache audit hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        freeze.get("cohort") != {"train": 2981, "validation": 371}
        or audit.get("physical_cache_records_verified") != 3352
    ):
        raise RuntimeError("S1 cache contract mismatch")
    for payload in (freeze, audit):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S1 cache safety mismatch")
    return root, {
        "selector_cache_freeze_sha256": hash_file(freeze_path),
        "selector_cache_wrapper_audit_sha256": hash_file(audit_path),
        "physical_cache_records_verified": 3352,
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_output(
    source_hashes: dict[str, str],
    cache: dict[str, object],
    baseline: dict[str, object],
    t4: dict[str, object],
) -> None:
    import numpy as np

    pair_path = OUTPUT / "pair_prediction_freeze.json"
    run_path = OUTPUT / "run_manifest.json"
    family_manifest = OUTPUT / "candidate_families/candidate_family_manifest.csv"
    initial_state = OUTPUT / "matched_initial_state.pt"
    for path in (pair_path, run_path, family_manifest, initial_state):
        if not path.is_file():
            raise RuntimeError(f"S1 pair output missing: {path}")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        pair.get("protocol_sha256") != PROTOCOL_SHA256
        or pair.get("sole_changed_variable") != "standard_vs_family_balanced_bag_pool"
        or set(pair.get("arms", {})) != set(POOL_MODES)
        or run_manifest.get("validated_cache_records")
        != {"train": 2981, "validation": 371}
        or hash_file(family_manifest)
        != pair.get("candidate_family_manifest_sha256")
    ):
        raise RuntimeError("S1 pair freeze/run contract mismatch")
    family_rows = _rows(family_manifest)
    if len(family_rows) != 371 or len({row["image_id"] for row in family_rows}) != 371:
        raise RuntimeError("S1 candidate-family cohort mismatch")
    for row in family_rows:
        path = OUTPUT / "candidate_families" / row["family_path"]
        with np.load(path, allow_pickle=False) as payload:
            indices = payload["candidate_indices"]
            family_ids = payload["family_ids"]
        if (
            hash_file(path) != row["family_sha256"]
            or indices.dtype != np.int64
            or family_ids.dtype != np.int64
            or family_ids.shape != indices.shape
            or len(indices) != int(row["candidate_count"])
            or np.any(family_ids < 0)
        ):
            raise RuntimeError(f"S1 family payload mismatch: {row['image_id']}")
    for mode in POOL_MODES:
        root = OUTPUT / mode
        freeze_path = root / "prediction_freeze.json"
        prediction_path = root / "predictions/prediction_manifest.csv"
        score_path = root / "candidate_scores/candidate_score_manifest.csv"
        for path in (
            freeze_path,
            prediction_path,
            score_path,
            root / "descriptor_residual.pt",
            root / "training_history.json",
        ):
            if not path.is_file():
                raise RuntimeError(f"S1 {mode} output missing: {path}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            freeze.get("arm") != mode
            or freeze.get("protocol_sha256") != PROTOCOL_SHA256
            or freeze.get("validation_predictions") != 371
            or freeze.get("candidate_family_manifest_sha256")
            != hash_file(family_manifest)
            or pair["arms"][mode].get("prediction_freeze_sha256")
            != hash_file(freeze_path)
        ):
            raise RuntimeError(f"S1 {mode} freeze mismatch")
        for payload in (freeze, pair, run_manifest):
            if (
                payload.get("validation_gt_read") is not False
                or payload.get("consumer_trained") is not False
                or payload.get("test_evaluated") is not False
            ):
                raise RuntimeError("S1 output safety mismatch")
        predictions = _rows(prediction_path)
        scores = _rows(score_path)
        if (
            len(predictions) != 371
            or len(scores) != 371
            or len({row["image_id"] for row in predictions}) != 371
            or len({row["image_id"] for row in scores}) != 371
        ):
            raise RuntimeError(f"S1 {mode} output cohort mismatch")
        for row in predictions:
            path = root / "predictions" / row["map_path"]
            values = np.load(path, allow_pickle=False)
            if (
                hash_file(path) != row["map_sha256"]
                or values.shape != (320, 320)
                or values.dtype != np.float16
                or not np.isfinite(values).all()
            ):
                raise RuntimeError(f"S1 {mode} prediction map mismatch: {row['image_id']}")
        for row in scores:
            path = root / "candidate_scores" / row["score_path"]
            with np.load(path, allow_pickle=False) as payload:
                indices = payload["candidate_indices"]
                logits = payload["candidate_logits"]
            if (
                hash_file(path) != row["score_sha256"]
                or indices.dtype != np.int64
                or logits.dtype != np.float32
                or logits.shape != indices.shape
                or not np.isfinite(logits).all()
            ):
                raise RuntimeError(f"S1 {mode} candidate score mismatch: {row['image_id']}")
    wrapper_audit = {
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hashes": source_hashes,
        "cache": cache,
        "baseline": baseline,
        "t4x2": t4,
        "physical_prediction_maps_verified": 742,
        "physical_candidate_score_payloads_verified": 742,
        "physical_candidate_family_payloads_verified": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "python": platform.python_version(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "wrapper_output_audit.json").write_text(
        json.dumps(wrapper_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    os.environ.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    RUNTIME.mkdir(parents=True, exist_ok=False)
    source_hashes = clone_and_verify()
    install_runtime()
    t4 = verify_t4x2()
    split = prepare_split()
    baseline_root, baseline = prepare_baseline()
    cache_root, cache = find_cache()
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_run_mask_bag_family_balanced_pair.py",
            "tests/test_mask_bag_pooling_residual_training.py",
            "tests/test_mask_bag_relational_selector.py",
        ],
        cwd=SOURCE,
    )
    run([sys.executable, "-m", "pytest", "-q"], cwd=SOURCE)
    run(
        [
            sys.executable,
            str(SOURCE / "project/run_mask_bag_family_balanced_pair.py"),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            SPLIT_SHA256,
            "--selector-cache-root",
            str(cache_root),
            "--expected-selector-cache-freeze-sha256",
            CACHE_FREEZE_SHA256,
            "--baseline-root",
            str(baseline_root),
            "--expected-baseline-checkpoint-sha256",
            BASELINE["checkpoint"],
            "--expected-baseline-source-commit",
            BASELINE["source_commit"],
            "--expected-baseline-protocol-sha256",
            BASELINE["protocol"],
            "--source-commit",
            SOURCE_COMMIT,
            "--protocol-sha256",
            PROTOCOL_SHA256,
            "--output-dir",
            str(OUTPUT),
        ],
        cwd=SOURCE,
    )
    audit_output(source_hashes, cache, baseline, t4)
    shutil.rmtree(SOURCE)
    shutil.rmtree(RUNTIME)


if __name__ == "__main__":
    main()
