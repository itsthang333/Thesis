"""Fail-closed Kaggle bootstrap for S10 high-resolution proposal MIL."""

from __future__ import annotations

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
from urllib.request import urlopen


KERNEL = "itsthang333/btxrd-highres-candidate-pmil-s10-v1"
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "3c29686ea80ff6d36e1c8441d1a5dd4826802b71"
PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/highres_candidate_pmil_s10_v1.json"
)
PROTOCOL_SHA256 = "f5aec30235676d8d878573b7acfeef09adaa1d1b0ea7fdc86b4f07b708cb8739"
AUDITOR_RELATIVE = Path("project/audit_highres_candidate_pmil_s10_output.py")
AUDITOR_SHA256 = "489546d45a50bbfd0e2c90b6f29415da87ca5e4624245f576eb21da2666c7f0c"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
TRAIN_CANDIDATE_MANIFEST_SHA256 = "ad3b52d626a46ba92325113a4742aba710167db86f759c77500a76ab280458d1"
TRAIN_PSEUDO_MANIFEST_SHA256 = "5aec58ce402da70189c2776453f614e21e5b46fde36b408fc7198c7eeee5dc21"
VAL_CANDIDATE_MANIFEST_SHA256 = "3e9396f532c793258919a1d99aa3dcef00523436c853207b8d7123e5dc133090"
VAL_PSEUDO_MANIFEST_SHA256 = "286d1fce0bcbd0f96a15b6b386ad27a0edac3500a63c5b87e16f9075d6c6320e"
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
RESNET_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
RESNET_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
RESNET_BYTES = 102_540_417
ARMS = (
    "geometry_v3_plus_upstream_control",
    "control_plus_s10_identity_capacity",
    "s10_pareto_identity_capture_purity",
)
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE = WORK / "s10_source"
RUNTIME = WORK / "s10_runtime"
OUTPUT = WORK / "btxrd_highres_candidate_pmil_s10_v1"


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    print(f"$ {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_and_verify() -> dict[str, str]:
    if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1:
        raise RuntimeError("S10 launch binding is not frozen")
    if len(CHECKOUT_COMMIT) != 40 or any(
        character not in "0123456789abcdef" for character in CHECKOUT_COMMIT
    ):
        raise RuntimeError("S10 checkout commit is invalid")
    run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(SOURCE)],
        cwd=WORK,
    )
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, CHECKOUT_COMMIT], cwd=SOURCE)
    protocol_path = SOURCE / PROTOCOL_RELATIVE
    if hash_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("S10 protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_PRELAUNCH"
        or protocol.get("experiment_id")
        != "EXP-20260803-codex-s10-highres-proposal-pmil-v1"
        or protocol.get("scientific_source", {}).get("commit") != SOURCE_COMMIT
        or protocol.get("representation", {}).get("input_size") != 640
        or protocol.get("training", {}).get("epochs") != 32
        or protocol.get("execution", {}).get("compute") != "private Kaggle T4x2 only"
        or protocol.get("safety")
        != {
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
            "collaborator_output_accessed": False,
        }
    ):
        raise RuntimeError("S10 protocol contract mismatch")
    hashes = protocol.get("canonical_lf_source_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("S10 protocol source inventory is missing")
    for relative, expected in hashes.items():
        path = SOURCE / relative
        if not path.is_file() or canonical_hash(path) != expected:
            raise RuntimeError(f"S10 source hash mismatch: {relative}")
    if canonical_hash(SOURCE / AUDITOR_RELATIVE) != AUDITOR_SHA256:
        raise RuntimeError("S10 independent auditor hash mismatch")
    return hashes


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S10 requires exactly two visible CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"S10 requires T4 x2, got {names}")
    checksums: list[float] = []
    for index in range(2):
        torch.manual_seed(610 + index)
        layer = torch.nn.Conv2d(3, 5, 3, padding=1).to(f"cuda:{index}").eval()
        values = torch.arange(
            3072, dtype=torch.float32, device=f"cuda:{index}"
        ).reshape(1, 3, 32, 32)
        with torch.inference_mode():
            result = layer(values)
        if not torch.isfinite(result).all():
            raise RuntimeError("S10 T4 convolution guard is non-finite")
        checksums.append(float(result.sum().cpu()))
    return {
        "cuda_device_count": 2,
        "cuda_device_names": names,
        "real_convolution_checksums": checksums,
    }


def run_static_tests() -> None:
    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "project/models/highres_candidate_pmil.py",
            "project/run_highres_candidate_pmil_s10.py",
            "project/audit_highres_candidate_pmil_s10_output.py",
            "project/kaggle_wrappers/run_highres_candidate_pmil_s10_v1.py",
        ],
        cwd=SOURCE,
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_highres_candidate_pmil.py",
            "tests/test_run_highres_candidate_pmil_s10.py",
            "tests/test_audit_highres_candidate_pmil_s10_output.py",
        ],
        cwd=SOURCE,
    )


def download_resnet() -> tuple[Path, dict[str, object]]:
    path = RUNTIME / "resnet50-11ad3fa6.pth"
    with urlopen(RESNET_URL, timeout=600) as response, path.open("xb") as handle:
        shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
    if path.stat().st_size != RESNET_BYTES or hash_file(path) != RESNET_SHA256:
        raise RuntimeError("S10 public ResNet-50 weight mismatch")
    return path, {
        "url": RESNET_URL,
        "bytes": RESNET_BYTES,
        "sha256": RESNET_SHA256,
        "checkpoint_redistributed": False,
    }


def prepare_split() -> Path:
    source = SOURCE / "artifacts/kaggle/wsl_source_consensus_val_v1/frozen_split_manifest.csv"
    if hash_file(source) != GIT_SPLIT_SHA256 or b"\r" in source.read_bytes():
        raise RuntimeError("S10 canonical split mismatch")
    target = RUNTIME / "frozen_split_manifest.csv"
    target.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    if hash_file(target) != SPLIT_SHA256:
        raise RuntimeError("S10 frozen split reconstruction mismatch")
    return target


def find_dataset_root() -> Path:
    candidates = [
        INPUT / "btxrd-raw" / "BTXRD",
        INPUT / "datasets" / "itsthang333" / "btxrd-raw" / "BTXRD",
        *sorted(INPUT.glob("**/BTXRD")),
    ]
    valid = {
        path.resolve()
        for path in candidates
        if (path / "images").is_dir()
        and ((path / "dataset.csv").is_file() or (path / "dataset.xlsx").is_file())
    }
    if len(valid) != 1:
        raise RuntimeError(f"Expected one S10 BTXRD root, found {sorted(map(str, valid))}")
    return next(iter(valid))


def find_candidate_root(
    *, manifest_sha: str, pseudo_sha: str, expected_split: str
) -> tuple[Path, dict[str, object]]:
    roots = []
    for manifest in INPUT.rglob("candidate_diagnostics_manifest.csv"):
        root = manifest.parent
        pseudo = root / "pseudo_mask_manifest.csv"
        if (
            hash_file(manifest) == manifest_sha
            and pseudo.is_file()
            and hash_file(pseudo) == pseudo_sha
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S10 {expected_split} candidate root, found {roots}")
    return roots[0], {
        "split": expected_split,
        "candidate_manifest_sha256": manifest_sha,
        "pseudo_manifest_sha256": pseudo_sha,
    }


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
        raise RuntimeError(f"Expected one S10 baseline root, found {roots}")
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
        raise RuntimeError(f"Expected one S10 selector-cache root, found {roots}")
    root = roots[0]
    freeze = json.loads((root / "selector_cache_freeze.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "wrapper_output_audit.json").read_text(encoding="utf-8"))
    for payload in (freeze, audit):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S10 selector-cache safety mismatch")
    if freeze.get("cohort") != {"train": 2981, "validation": 371}:
        raise RuntimeError("S10 selector-cache cohort mismatch")
    return root, {
        "freeze_sha256": CACHE_FREEZE_SHA256,
        "manifest_sha256": CACHE_MANIFEST_SHA256,
        "wrapper_audit_sha256": CACHE_WRAPPER_AUDIT_SHA256,
    }


def write_binding(source_hashes: dict[str, str]) -> Path:
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260803-codex-s10-highres-proposal-pmil-v1",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "independent_auditor_sha256": AUDITOR_SHA256,
        "source_hashes": source_hashes,
        "collaborator_output_accessed": False,
        "annotation_paths_resolved": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    path = RUNTIME / "launch_binding.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(binding, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
        raise RuntimeError(f"S10 output cohort mismatch: {path}")
    return rows


def audit_wrapper_output(
    *,
    source_hashes: dict[str, str],
    t4: dict[str, object],
    resnet: dict[str, object],
    train_candidates: dict[str, object],
    val_candidates: dict[str, object],
    baseline: dict[str, object],
    cache: dict[str, object],
) -> None:
    triple_path = OUTPUT / "prediction_triple_freeze.json"
    independent_path = OUTPUT / "independent_gt_blind_output_audit.json"
    binding_path = OUTPUT / "launch_binding.json"
    run_path = OUTPUT / "run_manifest.json"
    triple = json.loads(triple_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        triple.get("source_commit") != SOURCE_COMMIT
        or triple.get("protocol_sha256") != PROTOCOL_SHA256
        or triple.get("all_arms_physically_frozen_before_validation_gt") is not True
        or set(triple.get("arms", {})) != set(ARMS)
        or independent.get("status")
        != "PREDICTION_TRIPLE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS"
        or independent.get("validation_inference_reproduced") != 371
        or independent.get("physical_prediction_maps_verified") != 1113
        or independent.get("physical_candidate_scores_verified") != 1113
        or independent.get("prediction_triple_freeze_sha256") != hash_file(triple_path)
        or binding.get("source_hashes") != source_hashes
        or run_manifest.get("cohort") != {"train": 2981, "validation": 371}
    ):
        raise RuntimeError("S10 frozen output contract mismatch")
    for payload in (triple, independent, binding, run_manifest):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S10 output safety mismatch")
    physical_maps = 0
    physical_scores = 0
    for arm in ARMS:
        freeze_path = OUTPUT / arm / "prediction_freeze.json"
        if hash_file(freeze_path) != triple["arms"][arm]:
            raise RuntimeError("S10 arm freeze mismatch")
        for row in _rows(OUTPUT / arm / "predictions" / "prediction_manifest.csv", 371):
            path = OUTPUT / arm / "predictions" / row["map_path"]
            if not path.is_file() or hash_file(path) != row["map_sha256"]:
                raise RuntimeError("S10 map hash mismatch")
            physical_maps += 1
        for row in _rows(
            OUTPUT / arm / "candidate_scores" / "candidate_score_manifest.csv", 371
        ):
            path = OUTPUT / arm / "candidate_scores" / row["score_path"]
            if not path.is_file() or hash_file(path) != row["score_sha256"]:
                raise RuntimeError("S10 score hash mismatch")
            physical_scores += 1
    evidence_rows = _rows(OUTPUT / "s10_candidate_evidence" / "evidence_manifest.csv", 371)
    for row in evidence_rows:
        path = OUTPUT / "s10_candidate_evidence" / row["evidence_path"]
        if not path.is_file() or hash_file(path) != row["evidence_sha256"]:
            raise RuntimeError("S10 evidence hash mismatch")
    wrapper_audit = {
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "independent_auditor_sha256": AUDITOR_SHA256,
        "source_hashes": source_hashes,
        "t4x2": t4,
        "public_resnet50": resnet,
        "train_candidates": train_candidates,
        "validation_candidates": val_candidates,
        "baseline": baseline,
        "selector_cache": cache,
        "launch_binding_sha256": hash_file(binding_path),
        "prediction_triple_freeze_sha256": hash_file(triple_path),
        "independent_gt_blind_output_audit_sha256": hash_file(independent_path),
        "run_manifest_sha256": hash_file(run_path),
        "physical_prediction_maps_verified": physical_maps,
        "physical_candidate_scores_verified": physical_scores,
        "physical_candidate_evidence_verified": len(evidence_rows),
        "collaborator_output_accessed": False,
        "annotation_paths_resolved": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "python": platform.python_version(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (OUTPUT / "wrapper_output_audit.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(wrapper_audit, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    os.environ.update(
        {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    )
    RUNTIME.mkdir(parents=True, exist_ok=False)
    try:
        source_hashes = clone_and_verify()
        t4 = verify_t4x2()
        run_static_tests()
        resnet_path, resnet_audit = download_resnet()
        split = prepare_split()
        dataset_root = find_dataset_root()
        train_root, train_audit = find_candidate_root(
            manifest_sha=TRAIN_CANDIDATE_MANIFEST_SHA256,
            pseudo_sha=TRAIN_PSEUDO_MANIFEST_SHA256,
            expected_split="train",
        )
        val_root, val_audit = find_candidate_root(
            manifest_sha=VAL_CANDIDATE_MANIFEST_SHA256,
            pseudo_sha=VAL_PSEUDO_MANIFEST_SHA256,
            expected_split="val",
        )
        baseline_root, baseline_audit = find_baseline()
        cache_root, cache_audit = find_cache()
        binding_path = write_binding(source_hashes)
        run(
            [
                sys.executable,
                str(SOURCE / "project/run_highres_candidate_pmil_s10.py"),
                "--dataset-root", str(dataset_root),
                "--split-manifest", str(split),
                "--expected-split-sha256", SPLIT_SHA256,
                "--pretrained-checkpoint", str(resnet_path),
                "--expected-pretrained-sha256", RESNET_SHA256,
                "--selector-cache-root", str(cache_root),
                "--expected-selector-cache-freeze-sha256", CACHE_FREEZE_SHA256,
                "--baseline-root", str(baseline_root),
                "--expected-baseline-checkpoint-sha256", BASELINE["checkpoint"],
                "--expected-baseline-freeze-sha256", BASELINE["freeze"],
                "--expected-baseline-source-commit", BASELINE["source_commit"],
                "--expected-baseline-protocol-sha256", BASELINE["protocol"],
                "--train-candidate-root", str(train_root),
                "--train-candidate-manifest-sha256", TRAIN_CANDIDATE_MANIFEST_SHA256,
                "--train-pseudo-manifest-sha256", TRAIN_PSEUDO_MANIFEST_SHA256,
                "--val-candidate-root", str(val_root),
                "--val-candidate-manifest-sha256", VAL_CANDIDATE_MANIFEST_SHA256,
                "--val-pseudo-manifest-sha256", VAL_PSEUDO_MANIFEST_SHA256,
                "--source-commit", SOURCE_COMMIT,
                "--protocol-sha256", PROTOCOL_SHA256,
                "--output-dir", str(OUTPUT),
                "--batch-size", "4",
                "--epochs", "32",
                "--backbone-lr", "0.00003",
                "--head-lr", "0.0003",
                "--weight-decay", "0.0001",
                "--warmup-epochs", "4",
                "--top-dropout-fraction", "0.2",
                "--maximum-candidates", "81",
                "--num-workers", "4",
                "--seed", "42",
            ],
            cwd=SOURCE,
        )
        shutil.copy2(binding_path, OUTPUT / "launch_binding.json")
        run(
            [
                sys.executable,
                str(SOURCE / AUDITOR_RELATIVE),
                "--output-root", str(OUTPUT),
                "--protocol", str(SOURCE / PROTOCOL_RELATIVE),
                "--binding", str(OUTPUT / "launch_binding.json"),
                "--dataset-root", str(dataset_root),
                "--split-manifest", str(split),
                "--train-candidate-root", str(train_root),
                "--val-candidate-root", str(val_root),
                "--selector-cache-root", str(cache_root),
                "--baseline-root", str(baseline_root),
                "--output-audit", str(OUTPUT / "independent_gt_blind_output_audit.json"),
            ],
            cwd=SOURCE,
        )
        audit_wrapper_output(
            source_hashes=source_hashes,
            t4=t4,
            resnet=resnet_audit,
            train_candidates=train_audit,
            val_candidates=val_audit,
            baseline=baseline_audit,
            cache=cache_audit,
        )
    finally:
        for path in (SOURCE, RUNTIME):
            if path.exists() and path.resolve().parent == WORK.resolve():
                shutil.rmtree(path)


if __name__ == "__main__":
    main()
