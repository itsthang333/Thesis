from __future__ import annotations

"""Fail-closed Kaggle bootstrap for the predeclared N1 selector arm."""

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


KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-normal-anomaly-n1-v1"
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "c7ba620ce4492485ba0faa6dd42998e267be872d"
PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/rad_dino_mask_bag_normal_only_direct_anomaly_n1_v1.json"
)
PROTOCOL_SHA256 = "1112515d00ed9db80a05670404ad16127109844788a44018834ff82f452d9b7d"
AUDITOR_RELATIVE = Path("project/audit_mask_bag_normal_only_direct_anomaly_n1_output.py")
AUDITOR_SHA256 = "562a3d4bce2ed8e27cc11147577fbca9600c2b19273c16ce1970d1f3faddfdf1"
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
SOURCE = WORK / "n1_source"
RUNTIME = WORK / "n1_runtime"
OUTPUT = WORK / "btxrd_mask_bag_normal_only_direct_anomaly_n1_v1"


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
        raise RuntimeError("N1 launch binding is not frozen")
    if len(CHECKOUT_COMMIT) != 40 or any(
        character not in "0123456789abcdef" for character in CHECKOUT_COMMIT
    ):
        raise RuntimeError("Invalid bound N1 checkout")
    run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(SOURCE)],
        cwd=WORK,
    )
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, CHECKOUT_COMMIT], cwd=SOURCE)
    protocol_path = SOURCE / PROTOCOL_RELATIVE
    auditor_path = SOURCE / AUDITOR_RELATIVE
    if hash_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("N1 protocol hash mismatch")
    if canonical_hash(auditor_path) != AUDITOR_SHA256:
        raise RuntimeError("N1 independent auditor hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hashes = protocol.get("canonical_lf_source_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("N1 protocol source inventory missing")
    for relative, expected in hashes.items():
        path = SOURCE / relative
        if not path.is_file() or canonical_hash(path) != expected:
            raise RuntimeError(f"N1 source hash mismatch: {relative}")
    return hashes


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("N1 requires exactly two CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"N1 requires T4 x2, got {names}")
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
            result = layer(values)
        if not torch.isfinite(result).all():
            raise RuntimeError(f"N1 non-finite convolution on cuda:{index}")
        checksums.append(float(result.sum().cpu()))
    return {
        "cuda_device_count": 2,
        "cuda_device_names": names,
        "real_convolution_checksums": checksums,
    }


def prepare_split() -> Path:
    source = SOURCE / "artifacts/kaggle/wsl_source_consensus_val_v1/frozen_split_manifest.csv"
    if hash_file(source) != GIT_SPLIT_SHA256 or b"\r" in source.read_bytes():
        raise RuntimeError("N1 canonical split mismatch")
    target = RUNTIME / "frozen_split_manifest.csv"
    target.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    if hash_file(target) != SPLIT_SHA256:
        raise RuntimeError("N1 frozen split reconstruction mismatch")
    return target


def prepare_baseline() -> tuple[Path, dict[str, object]]:
    audit_path = unique("transport_audit.json", TRANSPORT_AUDIT_SHA256)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("validation_gt_included") is not False
        or audit.get("consumer_trained") is not False
        or audit.get("test_evaluated") is not False
    ):
        raise RuntimeError("N1 baseline transport safety mismatch")
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
        raise RuntimeError(f"Expected one N1 baseline root, found {roots}")
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
        raise RuntimeError("N1 cache audit hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        freeze.get("cohort") != {"train": 2981, "validation": 371}
        or audit.get("physical_cache_records_verified") != 3352
    ):
        raise RuntimeError("N1 cache contract mismatch")
    for payload in (freeze, audit):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("N1 cache safety mismatch")
    return root, {
        "selector_cache_freeze_sha256": hash_file(freeze_path),
        "selector_cache_wrapper_audit_sha256": hash_file(audit_path),
        "physical_cache_records_verified": 3352,
    }


def write_runtime_binding(source_hashes: dict[str, str]) -> Path:
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260801-codex-n1-normal-only-direct-anomaly-v1",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "independent_auditor_sha256": AUDITOR_SHA256,
        "source_hashes": source_hashes,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    path = RUNTIME / "launch_binding.json"
    path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
        raise RuntimeError(f"N1 output cohort mismatch: {path}")
    return rows


def audit_output(
    source_hashes: dict[str, str],
    cache: dict[str, object],
    baseline: dict[str, object],
    t4: dict[str, object],
) -> None:
    freeze_path = OUTPUT / "prediction_freeze.json"
    independent_path = OUTPUT / "independent_gt_blind_output_audit.json"
    binding_path = OUTPUT / "launch_binding.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("validation_predictions") != 371
        or independent.get("status") != "PASS_GT_BLIND_PHYSICAL_RECOMPUTATION"
        or independent.get("validation_predictions_recomputed") != 371
        or independent.get("prototype_bank_exact") is not True
        or independent.get("binding_sha256") != hash_file(binding_path)
        or binding.get("source_hashes") != source_hashes
    ):
        raise RuntimeError("N1 frozen independent-audit contract mismatch")
    for payload in (freeze, independent, binding):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("N1 output safety mismatch")
    predictions = _rows(OUTPUT / "predictions/prediction_manifest.csv", 371)
    score_rows = _rows(OUTPUT / "candidate_scores/candidate_score_manifest.csv", 371)
    evidence_rows = _rows(OUTPUT / "normal_anomaly_evidence_manifest.csv", 371)
    for row in predictions:
        path = OUTPUT / "predictions" / row["map_path"]
        if not path.is_file() or hash_file(path) != row["map_sha256"]:
            raise RuntimeError(f"N1 prediction map hash mismatch: {row['image_id']}")
    for row in score_rows:
        path = OUTPUT / "candidate_scores" / row["score_path"]
        if not path.is_file() or hash_file(path) != row["score_sha256"]:
            raise RuntimeError(f"N1 score payload hash mismatch: {row['image_id']}")
    for row in evidence_rows:
        path = OUTPUT / "normal_anomaly_evidence" / row["evidence_path"]
        if not path.is_file() or hash_file(path) != row["evidence_sha256"]:
            raise RuntimeError(f"N1 evidence payload hash mismatch: {row['image_id']}")
    wrapper_audit = {
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "independent_auditor_sha256": AUDITOR_SHA256,
        "source_hashes": source_hashes,
        "cache": cache,
        "baseline": baseline,
        "t4x2": t4,
        "launch_binding_sha256": hash_file(binding_path),
        "prediction_freeze_sha256": hash_file(freeze_path),
        "independent_gt_blind_output_audit_sha256": hash_file(independent_path),
        "prototype_bank_recomputed_exact": True,
        "physical_candidate_score_payloads_verified": len(score_rows),
        "physical_view_evidence_payloads_verified": len(evidence_rows),
        "physical_prediction_maps_verified": len(predictions),
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
    os.environ.update({"PYTHONHASHSEED": "42"})
    RUNTIME.mkdir(parents=True, exist_ok=False)
    try:
        source_hashes = clone_and_verify()
        t4 = verify_t4x2()
        split = prepare_split()
        baseline_root, baseline = prepare_baseline()
        cache_root, cache = find_cache()
        binding_path = write_runtime_binding(source_hashes)
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_mask_bag_normal_anomaly.py",
                "tests/test_run_mask_bag_normal_only_direct_anomaly_n1.py",
                "tests/test_audit_mask_bag_normal_only_direct_anomaly_n1_output.py",
                "tests/test_mask_bag_selector_cache.py",
                "tests/test_mask_bag_selector_cache_io.py",
            ],
            cwd=SOURCE,
        )
        run([sys.executable, "-m", "pytest", "-q"], cwd=SOURCE)
        run(
            [
                sys.executable,
                str(SOURCE / "project/run_mask_bag_normal_only_direct_anomaly_n1.py"),
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
                "--expected-baseline-freeze-sha256",
                BASELINE["freeze"],
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
                "--prototype-count",
                "32",
                "--seed",
                "42",
            ],
            cwd=SOURCE,
        )
        shutil.copy2(binding_path, OUTPUT / "launch_binding.json")
        run(
            [
                sys.executable,
                str(SOURCE / AUDITOR_RELATIVE),
                "--protocol",
                str(SOURCE / PROTOCOL_RELATIVE),
                "--binding",
                str(OUTPUT / "launch_binding.json"),
                "--split-manifest",
                str(split),
                "--selector-cache-root",
                str(cache_root),
                "--baseline-root",
                str(baseline_root),
                "--output-root",
                str(OUTPUT),
                "--audit-output",
                str(OUTPUT / "independent_gt_blind_output_audit.json"),
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
