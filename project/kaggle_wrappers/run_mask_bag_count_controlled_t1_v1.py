from __future__ import annotations

"""Fail-closed Kaggle bootstrap for the predeclared T1 selector arm."""

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


KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-count-controlled-t1-v1"
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "c7f0937d515ded9bbd8928a2236cbe44b7a25f79"
PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1.json"
)
PROTOCOL_SHA256 = "6a4379e896f3ea3862dce1edcdea20af09a90ec8f9cbbd6eb25bf8eca1306a7c"
POST_FREEZE_ONLY_SOURCE_PATHS = {
    "project/evaluate_mask_bag_selector_arm.py",
    "project/models/mask_bag_ranking_diagnostics.py",
}
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
SOURCE = WORK / "t1_source"
RUNTIME = WORK / "t1_runtime"
OUTPUT = WORK / "btxrd_mask_bag_count_controlled_self_paced_t1_v1"


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    resolved_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved_root and resolved_root not in target.parents:
                raise RuntimeError(f"Archive member escapes destination: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Archive contains a symlink: {member.filename}")
        archive.extractall(destination)


def clone_and_verify() -> dict[str, str]:
    if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1:
        raise RuntimeError("T1 launch binding is not frozen")
    if len(CHECKOUT_COMMIT) != 40 or any(
        character not in "0123456789abcdef" for character in CHECKOUT_COMMIT
    ):
        raise RuntimeError("Invalid bound T1 checkout")
    run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(SOURCE)],
        cwd=WORK,
    )
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, CHECKOUT_COMMIT], cwd=SOURCE)
    protocol_path = SOURCE / PROTOCOL_RELATIVE
    if hash_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("T1 protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hashes = protocol.get("canonical_lf_source_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("T1 protocol source inventory missing")
    for relative, expected in hashes.items():
        path = SOURCE / relative
        if not path.is_file() or canonical_hash(path) != expected:
            raise RuntimeError(f"T1 source hash mismatch: {relative}")
    return {
        relative: expected
        for relative, expected in hashes.items()
        if relative not in POST_FREEZE_ONLY_SOURCE_PATHS
    }


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
        raise RuntimeError("T1 requires exactly two CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"T1 requires T4 x2, got {names}")
    checksums = []
    for index in range(2):
        torch.manual_seed(42 + index)
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
        raise RuntimeError("T1 canonical split mismatch")
    target = RUNTIME / "frozen_split_manifest.csv"
    target.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    if hash_file(target) != SPLIT_SHA256:
        raise RuntimeError("T1 frozen split reconstruction mismatch")
    return target


def prepare_baseline() -> tuple[Path, dict[str, object]]:
    audit_path = unique("transport_audit.json", TRANSPORT_AUDIT_SHA256)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("validation_gt_included") is not False
        or audit.get("consumer_trained") is not False
        or audit.get("test_evaluated") is not False
    ):
        raise RuntimeError("T1 baseline transport safety mismatch")
    archive = unique("square_corrected_baseline.zip.bin", BASELINE_ARCHIVE_SHA256)
    extracted = RUNTIME / "baseline"
    safe_extract(archive, extracted)
    roots = []
    for freeze in extracted.rglob("prediction_freeze.json"):
        root = freeze.parent
        if (
            hash_file(freeze) == BASELINE["freeze"]
            and hash_file(root / "rad_dino_mask_bag_mil.pt") == BASELINE["checkpoint"]
            and hash_file(root / "predictions/prediction_manifest.csv") == BASELINE["manifest"]
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one T1 baseline root, found {roots}")
    return roots[0], {
        "baseline_archive_sha256": hash_file(archive),
        "transport_audit_sha256": hash_file(audit_path),
        "prediction_freeze_sha256": BASELINE["freeze"],
        "checkpoint_sha256": BASELINE["checkpoint"],
        "prediction_manifest_sha256": BASELINE["manifest"],
    }


def find_cache() -> tuple[Path, dict[str, object]]:
    freeze_path = unique("selector_cache_freeze.json", CACHE_FREEZE_SHA256)
    root = freeze_path.parent.resolve()
    audit_path = root / "wrapper_output_audit.json"
    if hash_file(audit_path) != CACHE_WRAPPER_AUDIT_SHA256:
        raise RuntimeError("T1 cache audit hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        freeze.get("cohort") != {"train": 2981, "validation": 371}
        or audit.get("physical_cache_records_verified") != 3352
    ):
        raise RuntimeError("T1 cache contract mismatch")
    for payload in (freeze, audit):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("T1 cache safety mismatch")
    return root, {
        "selector_cache_freeze_sha256": hash_file(freeze_path),
        "selector_cache_wrapper_audit_sha256": hash_file(audit_path),
        "physical_cache_records_verified": 3352,
    }


def _rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
        raise RuntimeError(f"T1 output cohort mismatch: {path}")
    return rows


def _verify_manifest_payloads(
    manifest: Path,
    payload_root: Path,
    *,
    expected: int,
    path_field: str,
    hash_field: str,
) -> None:
    import numpy as np

    for row in _rows(manifest, expected):
        path = payload_root / row[path_field]
        if not path.is_file() or hash_file(path) != row[hash_field]:
            raise RuntimeError(f"T1 output payload hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if not payload.files or any(
                not np.isfinite(payload[name]).all()
                for name in payload.files
                if payload[name].dtype.kind in "fc"
            ):
                raise RuntimeError(f"T1 output payload content mismatch: {path}")


def audit_output(
    source_hashes: dict[str, str],
    cache: dict[str, object],
    baseline: dict[str, object],
    t4: dict[str, object],
) -> None:
    import numpy as np

    freeze_path = OUTPUT / "prediction_freeze.json"
    run_path = OUTPUT / "run_manifest.json"
    required = {
        "crossfit_assignment_sha256": OUTPUT / "crossfit_assignment.json",
        "producer_gate_audit_sha256": OUTPUT / "producer_gate_audit.json",
        "target_freeze_sha256": OUTPUT / "self_paced_targets/target_freeze.json",
        "pretraining_identity_audit_sha256": OUTPUT / "pretraining_identity_audit.json",
        "residual_checkpoint_sha256": OUTPUT / "count_controlled_self_paced_residual.pt",
        "consumer_training_history_sha256": OUTPUT / "consumer_training_history.json",
        "gt_blind_diagnostics_sha256": OUTPUT / "gt_blind_diagnostics.csv",
        "candidate_score_manifest_sha256": OUTPUT / "candidate_scores/candidate_score_manifest.csv",
        "prediction_manifest_sha256": OUTPUT / "predictions/prediction_manifest.csv",
    }
    if not freeze_path.is_file() or not run_path.is_file():
        raise RuntimeError("T1 freeze/run manifest missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("validation_predictions") != 371
        or freeze.get("training_labels") != "image_level_only"
        or freeze.get("confirmation_residual_trained_after_producer_gate") is not True
        or run_manifest.get("validated_cache_records") != {"train": 2981, "validation": 371}
        or run_manifest.get("output_hashes") != freeze
    ):
        raise RuntimeError("T1 freeze/run contract mismatch")
    for payload in (freeze, run_manifest):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("T1 output safety mismatch")
    for key, path in required.items():
        if not path.is_file() or hash_file(path) != freeze.get(key):
            raise RuntimeError(f"T1 frozen artifact mismatch: {key}")
    target_freeze = json.loads(required["target_freeze_sha256"].read_text(encoding="utf-8"))
    target_root = OUTPUT / "self_paced_targets"
    expected_target_names = {
        "negative_targets.csv",
        "positive_targets_stage_1.csv",
        "positive_targets_stage_2.csv",
        "positive_targets_stage_3.csv",
    }
    if set(target_freeze.get("target_hashes", {})) != expected_target_names:
        raise RuntimeError("T1 target inventory mismatch")
    for name, expected_hash in target_freeze["target_hashes"].items():
        path = target_root / name
        if not path.is_file() or hash_file(path) != expected_hash:
            raise RuntimeError(f"T1 target artifact mismatch: {name}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            if not list(csv.DictReader(handle)):
                raise RuntimeError(f"T1 target artifact is empty: {name}")
    oof_payloads = 0
    for fold, expected_rows in enumerate((596, 596, 596, 596, 597)):
        fold_root = OUTPUT / "oof_producers" / f"fold_{fold}"
        inventory = freeze["producer_hashes"][f"fold_{fold}"]
        artifacts = {
            "producer_checkpoint_sha256": fold_root / "producer.pt",
            "training_history_sha256": fold_root / "training_history.json",
            "score_manifest_sha256": fold_root / "score_manifest.csv",
            "fold_audit_sha256": fold_root / "fold_audit.json",
        }
        for key, path in artifacts.items():
            if hash_file(path) != inventory[key]:
                raise RuntimeError(f"T1 OOF fold artifact mismatch: {fold}/{key}")
        _verify_manifest_payloads(
            artifacts["score_manifest_sha256"],
            fold_root / "scores",
            expected=expected_rows,
            path_field="payload_path",
            hash_field="payload_sha256",
        )
        oof_payloads += expected_rows
    _verify_manifest_payloads(
        required["candidate_score_manifest_sha256"],
        OUTPUT / "candidate_scores",
        expected=371,
        path_field="score_path",
        hash_field="score_sha256",
    )
    predictions = _rows(required["prediction_manifest_sha256"], 371)
    for row in predictions:
        path = OUTPUT / "predictions" / row["map_path"]
        values = np.load(path, allow_pickle=False)
        if (
            hash_file(path) != row["map_sha256"]
            or values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
        ):
            raise RuntimeError(f"T1 prediction-map mismatch: {row['image_id']}")
    _verify_manifest_payloads(
        required["gt_blind_diagnostics_sha256"],
        OUTPUT / "validation_residual_evidence",
        expected=371,
        path_field="residual_evidence_path",
        hash_field="residual_evidence_sha256",
    )
    if oof_payloads != 2981:
        raise RuntimeError("T1 OOF physical gate mismatch")
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
        "physical_oof_score_payloads_verified": 2981,
        "physical_target_files_verified": 4,
        "physical_validation_residual_payloads_verified": 371,
        "physical_candidate_score_payloads_verified": 371,
        "physical_prediction_maps_verified": 371,
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
    try:
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
                "tests/test_mask_bag_count_controlled_self_paced.py",
                "tests/test_run_mask_bag_count_controlled_self_paced_t1_arm.py",
                "tests/test_audit_mask_bag_count_controlled_self_paced_t1_output.py",
                "tests/test_mask_bag_crossfit.py",
                "tests/test_mask_bag_selector_cache.py",
                "tests/test_mask_bag_selector_cache_io.py",
            ],
            cwd=SOURCE,
        )
        run([sys.executable, "-m", "pytest", "-q"], cwd=SOURCE)
        run(
            [
                sys.executable,
                str(SOURCE / "project/run_mask_bag_count_controlled_self_paced_t1_arm.py"),
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
    finally:
        for cleanup_path in (SOURCE, RUNTIME):
            if cleanup_path.exists() and cleanup_path.resolve().parent == WORK.resolve():
                shutil.rmtree(cleanup_path)


if __name__ == "__main__":
    main()
