from __future__ import annotations

"""Fail-closed Kaggle bootstrap for the frozen S6 label-granularity pair."""

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


KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-label-granularity-s6-v1"
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "543ee89654a0ed00e80ded16924a760585337924"
PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/rad_dino_mask_bag_label_granularity_s6_v1.json"
)
PROTOCOL_SHA256 = "f4e17d24dfab36f01526550c7dc306fc7549494acc4545153454c61ae926bfc3"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_MANIFEST_SHA256 = "8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e"
CACHE_WRAPPER_AUDIT_SHA256 = "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
BASELINE = {
    "freeze": "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3",
    "checkpoint": "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069",
    "manifest": "a810e1fcc4c4422d207eb020a70313caf5d3402bf30c277331247a30555678ee",
    "source_commit": "fda732941664e67d4b87a8c3cba071b6979b2214",
    "protocol": "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555",
}
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE = WORK / "s6_source"
RUNTIME = WORK / "s6_runtime"
OUTPUT = WORK / "btxrd_mask_bag_label_granularity_s6_pair_v1"


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


def clone_and_verify() -> tuple[dict[str, str], dict[str, object]]:
    if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1:
        raise RuntimeError("S6 launch binding is not frozen")
    if len(CHECKOUT_COMMIT) != 40 or any(
        character not in "0123456789abcdef" for character in CHECKOUT_COMMIT
    ):
        raise RuntimeError("S6 bound checkout is invalid")
    run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(SOURCE)],
        cwd=WORK,
    )
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, CHECKOUT_COMMIT], cwd=SOURCE)
    protocol_path = SOURCE / PROTOCOL_RELATIVE
    if hash_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("S6 protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_PRELAUNCH"
        or protocol.get("scientific_source", {}).get("commit") != SOURCE_COMMIT
    ):
        raise RuntimeError("S6 protocol status/source mismatch")
    verified: dict[str, str] = {}
    for section in ("canonical_lf_source_hashes", "post_freeze_only_source_hashes"):
        hashes = protocol.get(section, {})
        if not isinstance(hashes, dict) or not hashes:
            raise RuntimeError(f"S6 protocol {section} is missing")
        for relative, expected in hashes.items():
            path = SOURCE / relative
            if not path.is_file() or canonical_hash(path) != expected:
                raise RuntimeError(f"S6 source hash mismatch: {relative}")
            if section == "canonical_lf_source_hashes":
                verified[relative] = expected
    return verified, protocol


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S6 requires exactly two CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"S6 requires Tesla T4 x2, got {names}")
    checksums = []
    for index in range(2):
        torch.manual_seed(620 + index)
        layer = torch.nn.Conv2d(3, 5, 3, padding=1).to(f"cuda:{index}").eval()
        values = torch.arange(
            3072, dtype=torch.float32, device=f"cuda:{index}"
        ).reshape(1, 3, 32, 32)
        with torch.inference_mode():
            result = layer(values)
        if not torch.isfinite(result).all():
            raise RuntimeError(f"S6 non-finite T4 guard on cuda:{index}")
        checksums.append(float(result.sum().cpu()))
    return {
        "cuda_device_count": 2,
        "cuda_device_names": names,
        "real_convolution_checksums": checksums,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def prepare_split() -> Path:
    source = SOURCE / "artifacts/kaggle/wsl_source_consensus_val_v1/frozen_split_manifest.csv"
    if hash_file(source) != GIT_SPLIT_SHA256 or b"\r" in source.read_bytes():
        raise RuntimeError("S6 canonical Git split mismatch")
    target = RUNTIME / "frozen_split_manifest.csv"
    target.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    if hash_file(target) != SPLIT_SHA256:
        raise RuntimeError("S6 frozen split reconstruction mismatch")
    return target


def find_baseline() -> tuple[Path, dict[str, object]]:
    roots = []
    for freeze in INPUT.rglob("prediction_freeze.json"):
        root = freeze.parent
        checkpoint = root / "rad_dino_mask_bag_mil.pt"
        manifest = root / "predictions" / "prediction_manifest.csv"
        if (
            hash_file(freeze) == BASELINE["freeze"]
            and checkpoint.is_file()
            and hash_file(checkpoint) == BASELINE["checkpoint"]
            and manifest.is_file()
            and hash_file(manifest) == BASELINE["manifest"]
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S6 baseline root, found {roots}")
    return roots[0], dict(BASELINE)


def find_cache() -> tuple[Path, dict[str, object]]:
    roots = []
    for freeze in INPUT.rglob("selector_cache_freeze.json"):
        root = freeze.parent
        manifest = root / "selector_cache_manifest.csv"
        audit = root / "wrapper_output_audit.json"
        if (
            hash_file(freeze) == CACHE_FREEZE_SHA256
            and manifest.is_file()
            and hash_file(manifest) == CACHE_MANIFEST_SHA256
            and audit.is_file()
            and hash_file(audit) == CACHE_WRAPPER_AUDIT_SHA256
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S6 selector-cache root, found {roots}")
    root = roots[0]
    freeze = json.loads((root / "selector_cache_freeze.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "wrapper_output_audit.json").read_text(encoding="utf-8"))
    for payload in (freeze, audit):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S6 selector-cache safety mismatch")
    if (
        freeze.get("cohort") != {"train": 2981, "validation": 371}
        or audit.get("physical_cache_records_verified") != 3352
    ):
        raise RuntimeError("S6 selector-cache cohort mismatch")
    return root, {
        "freeze_sha256": CACHE_FREEZE_SHA256,
        "manifest_sha256": CACHE_MANIFEST_SHA256,
        "wrapper_audit_sha256": CACHE_WRAPPER_AUDIT_SHA256,
        "physical_cache_records_verified": 3352,
    }


def write_binding(source_hashes: dict[str, str]) -> Path:
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260802-codex-s6-label-granularity-mil-v1",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "source_hashes": source_hashes,
        "validation_subtype_label_used_for_routing": False,
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    path = RUNTIME / "launch_binding.json"
    path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_static_tests() -> None:
    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "project/models/mask_bag_label_granularity.py",
            "project/models/mask_bag_label_granularity_training.py",
            "project/run_mask_bag_label_granularity_s6_pair.py",
            "project/audit_mask_bag_label_granularity_s6_output.py",
        ],
        cwd=SOURCE,
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_mask_bag_label_granularity.py",
            "tests/test_mask_bag_label_granularity_training.py",
            "tests/test_run_mask_bag_label_granularity_s6_pair.py",
            "tests/test_audit_mask_bag_label_granularity_s6_output.py",
            "tests/test_mask_bag_label_granularity_s6_protocol.py",
            "tests/test_rad_dino_mask_bag_mil.py",
            "tests/test_mask_bag_selector_cache.py",
            "tests/test_mask_bag_selector_cache_io.py",
        ],
        cwd=SOURCE,
    )


def _rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
        raise RuntimeError(f"S6 output cohort mismatch: {path}")
    return rows


def audit_wrapper_output(
    *,
    source_hashes: dict[str, str],
    t4: dict[str, object],
    cache: dict[str, object],
    baseline: dict[str, object],
    independent_audit_path: Path,
) -> None:
    import numpy as np

    pair_path = OUTPUT / "prediction_pair_freeze.json"
    run_path = OUTPUT / "run_manifest.json"
    if not pair_path.is_file() or not run_path.is_file():
        raise RuntimeError("S6 pair/run manifest missing")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_audit_path.read_text(encoding="utf-8"))
    if (
        pair.get("source_commit") != SOURCE_COMMIT
        or pair.get("protocol_sha256") != PROTOCOL_SHA256
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("validation_subtype_label_used_for_routing") is not False
        or set(pair.get("arms", {}))
        != {"coarse_control", "hierarchical_entropy_routed"}
        or run_manifest.get("prediction_pair_freeze_sha256") != hash_file(pair_path)
        or independent.get("status")
        != "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_PASS"
        or independent.get("prediction_pair_freeze_sha256") != hash_file(pair_path)
        or independent.get("physical_candidate_score_payloads") != 742
        or independent.get("physical_prediction_maps") != 742
    ):
        raise RuntimeError("S6 output/independent-audit contract mismatch")
    for payload in (pair, run_manifest, independent):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S6 output safety mismatch")
    score_payloads = 0
    prediction_maps = 0
    for arm in ("coarse_control", "hierarchical_entropy_routed"):
        arm_root = OUTPUT / arm
        freeze_path = arm_root / "prediction_freeze.json"
        if hash_file(freeze_path) != pair["arms"][arm]:
            raise RuntimeError(f"S6 {arm} freeze hash mismatch")
        scores = _rows(
            arm_root / "candidate_scores" / "candidate_score_manifest.csv", 371
        )
        predictions = _rows(
            arm_root / "predictions" / "prediction_manifest.csv", 371
        )
        for row in scores:
            path = arm_root / "candidate_scores" / row["score_path"]
            if hash_file(path) != row["score_sha256"]:
                raise RuntimeError(f"S6 {arm} score hash mismatch")
            with np.load(path, allow_pickle=False) as payload:
                if (
                    set(payload.files)
                    != {"schema_version", "candidate_indices", "candidate_logits"}
                    or int(payload["schema_version"]) != 1
                    or not np.isfinite(payload["candidate_logits"]).all()
                ):
                    raise RuntimeError(f"S6 {arm} score content mismatch")
            score_payloads += 1
        for row in predictions:
            path = arm_root / "predictions" / row["map_path"]
            values = np.load(path, allow_pickle=False)
            if (
                hash_file(path) != row["map_sha256"]
                or values.shape != (320, 320)
                or values.dtype != np.float16
                or not np.isfinite(values).all()
            ):
                raise RuntimeError(f"S6 {arm} map content mismatch")
            prediction_maps += 1
    if score_payloads != 742 or prediction_maps != 742:
        raise RuntimeError("S6 physical output count mismatch")
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
        "prediction_pair_freeze_sha256": hash_file(pair_path),
        "independent_gt_blind_output_audit_sha256": hash_file(independent_audit_path),
        "physical_candidate_score_payloads_verified": score_payloads,
        "physical_prediction_maps_verified": prediction_maps,
        "collaborator_output_accessed": False,
        "validation_subtype_label_used_for_routing": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "wrapper_output_audit.json").write_text(
        json.dumps(wrapper_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    os.environ.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    RUNTIME.mkdir(parents=True, exist_ok=False)
    try:
        source_hashes, _protocol = clone_and_verify()
        t4 = verify_t4x2()
        split = prepare_split()
        baseline_root, baseline = find_baseline()
        cache_root, cache = find_cache()
        binding_path = write_binding(source_hashes)
        run_static_tests()
        run(
            [
                sys.executable,
                str(SOURCE / "project/run_mask_bag_label_granularity_s6_pair.py"),
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
        independent_audit_path = OUTPUT / "independent_gt_blind_output_audit.json"
        run(
            [
                sys.executable,
                str(SOURCE / "project/audit_mask_bag_label_granularity_s6_output.py"),
                "--output-root",
                str(OUTPUT),
                "--protocol",
                str(SOURCE / PROTOCOL_RELATIVE),
                "--binding",
                str(binding_path),
                "--split",
                str(split),
                "--cache-root",
                str(cache_root),
                "--baseline-root",
                str(baseline_root),
                "--output-json",
                str(independent_audit_path),
            ],
            cwd=SOURCE,
        )
        audit_wrapper_output(
            source_hashes=source_hashes,
            t4=t4,
            cache=cache,
            baseline=baseline,
            independent_audit_path=independent_audit_path,
        )
    finally:
        for cleanup_path in (SOURCE, RUNTIME):
            if cleanup_path.exists() and cleanup_path.resolve().parent == WORK.resolve():
                shutil.rmtree(cleanup_path)


if __name__ == "__main__":
    main()
