"""Fail-closed audit-only Kaggle wrapper for immutable S8 version-1 output."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from hashlib import sha256


KERNEL = "itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1"
PRODUCER_KERNEL = "itsthang333/btxrd-skelex-reconstruction-selector-s8-v1"
PRODUCER_KERNEL_VERSION = 1
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "b4543aeb9430345c9b789384943bd218816a85dd"
AUDITOR_CORRECTION_COMMIT = "969327c4fbbd635fff2e3a00d34d533af8a3c340"
CORRECTION_ADDENDUM_COMMIT = "16f1b61ef99e866dcfced826b4b2ffb76fb0d3b5"
PROTOCOL_RELATIVE = Path("artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json")
PROTOCOL_SHA256 = "7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07"
CORRECTION_RELATIVE = Path("artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_serialized_lcb_audit_correction.json")
CORRECTION_SHA256 = "94e5881f763cc2cb3bd0a3f49cb563f2449140a7c576211252a45579597fc8a2"
AUDITOR_RELATIVE = Path("project/audit_skelex_reconstruction_selector_s8_output.py")
AUDITOR_SHA256 = "c972e1460332119cefd11c1145035a497748d4797482c29c51cf62980c560232"
TEST_RELATIVE = Path("tests/test_audit_skelex_reconstruction_selector_s8_output.py")
TEST_SHA256 = "fc9786e9b0e8bfe43fa3bb9d8cc7d9e9933270caf2f7a6de1eac797d29dc11a6"
PAIR_FREEZE_SHA256 = "b2cfd59fb01046f445d098790efa5a0fdc649bbc80f565439ba51c5cd453fa00"
RUN_MANIFEST_SHA256 = "5bb136f8b6f7a6a173abacce2faf0aad1b7caf9e087adcfd158655d71ff7c510"
DIAGNOSTICS_SHA256 = "98ceacd4a3dd1c32d42105c8cddc436d0e8256dd596c3b250f18fa2e39ecc569"
EVIDENCE_MANIFEST_SHA256 = "be391e11deef6a02a32c85c3bdc861cb05dd357313d89b9177c3bcdfb850cf55"
ARMS = {
    "geometry_v3_plus_upstream_equal_rank": {
        "freeze": "aa8acefaafe9c453dd08e1cba8d71b4a0aae25ecf16662774aebf53d01cf2ccd",
        "scores": "6d08db6c7fd29d5ed2bc7ff57133cd9fdea1adc65ec797f9c3f0ee424b3452fc",
        "predictions": "f66f7370b8b93ddadd1b7eef134ae5735c0a8f6743c97de021eab331d06666bf",
    },
    "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank": {
        "freeze": "a96e8a0bf88201c8b3fffb30c7ec41fd0abb2c3e4b7e9430408665b01dc2a148",
        "scores": "0f66c5ff54ea44f778bc62fb0e71145191e0189bdcc0ac419f87d0b1ec566de3",
        "predictions": "285c026853bb5e91482140e03cc2d47d418f4172716aec0aad32e97e5b9d7309",
    },
}
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE = WORK / "s8_audit_source"
OUTPUT = WORK / "btxrd_skelex_reconstruction_selector_s8_audit_v1"


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


def clone_and_verify() -> None:
    if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1:
        raise RuntimeError("S8 audit-only launch binding is not frozen")
    if any(len(value) != 40 for value in (CHECKOUT_COMMIT, SOURCE_COMMIT, AUDITOR_CORRECTION_COMMIT, CORRECTION_ADDENDUM_COMMIT)):
        raise RuntimeError("S8 audit-only checkout/source commit is unbound")
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(SOURCE)], cwd=WORK)
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    for ancestor in (SOURCE_COMMIT, AUDITOR_CORRECTION_COMMIT, CORRECTION_ADDENDUM_COMMIT):
        run(["git", "merge-base", "--is-ancestor", ancestor, CHECKOUT_COMMIT], cwd=SOURCE)
    if hash_file(SOURCE / PROTOCOL_RELATIVE) != PROTOCOL_SHA256:
        raise RuntimeError("S8 audit-only protocol hash mismatch")
    if hash_file(SOURCE / CORRECTION_RELATIVE) != CORRECTION_SHA256:
        raise RuntimeError("S8 audit-only correction hash mismatch")
    if canonical_hash(SOURCE / AUDITOR_RELATIVE) != AUDITOR_SHA256:
        raise RuntimeError("S8 audit-only auditor source hash mismatch")
    if canonical_hash(SOURCE / TEST_RELATIVE) != TEST_SHA256:
        raise RuntimeError("S8 audit-only regression test hash mismatch")
    correction = json.loads((SOURCE / CORRECTION_RELATIVE).read_text(encoding="utf-8"))
    if (
        correction.get("status") != "FROZEN_AUDITOR_NUMERICAL_REPRODUCIBILITY_CORRECTION"
        or correction.get("correction", {}).get("scientific_algorithm_changed") is not False
        or correction.get("correction", {}).get("prediction_changed") is not False
        or correction.get("correction", {}).get("audit_only_kaggle_t4x2_authorized") is not True
        or correction.get("validation_gt_read") is not False
    ):
        raise RuntimeError("S8 audit-only correction contract mismatch")


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S8 audit-only correction requires exactly two CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in names):
        raise RuntimeError(f"S8 audit-only correction requires T4x2, found {names}")
    return {"cuda_device_count": 2, "cuda_device_names": names}


def find_and_verify_producer_output() -> Path:
    roots: list[Path] = []
    for pair_path in INPUT.rglob("prediction_pair_freeze.json"):
        if hash_file(pair_path) == PAIR_FREEZE_SHA256:
            roots.append(pair_path.parent.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one immutable S8 producer output, found {roots}")
    root = roots[0]
    checks = {
        root / "run_manifest.json": RUN_MANIFEST_SHA256,
        root / "gt_blind_diagnostics.json": DIAGNOSTICS_SHA256,
        root / "reconstruction_evidence/evidence_manifest.json": EVIDENCE_MANIFEST_SHA256,
    }
    for arm, hashes in ARMS.items():
        arm_root = root / arm
        checks[arm_root / "prediction_freeze.json"] = hashes["freeze"]
        checks[arm_root / "candidate_scores/candidate_score_manifest.csv"] = hashes["scores"]
        checks[arm_root / "predictions/prediction_manifest.csv"] = hashes["predictions"]
    for path, expected in checks.items():
        if not path.is_file() or hash_file(path) != expected:
            raise RuntimeError(f"S8 immutable producer output hash mismatch: {path}")
    return root


def run_static_tests() -> None:
    run([sys.executable, "-m", "py_compile", str(AUDITOR_RELATIVE)], cwd=SOURCE)
    run([sys.executable, "-m", "pytest", "-q", str(TEST_RELATIVE)], cwd=SOURCE)


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    clone_and_verify()
    t4 = verify_t4x2()
    run_static_tests()
    producer_root = find_and_verify_producer_output()
    binding = {
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "producer_kernel": PRODUCER_KERNEL,
        "producer_kernel_version": PRODUCER_KERNEL_VERSION,
        "checkout_commit": CHECKOUT_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "correction_sha256": CORRECTION_SHA256,
        "auditor_sha256": AUDITOR_SHA256,
        "prediction_pair_freeze_sha256": PAIR_FREEZE_SHA256,
        "t4x2": t4,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    binding_path = OUTPUT / "audit_launch_binding.json"
    binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path = OUTPUT / "independent_gt_blind_output_audit.json"
    run(
        [
            sys.executable,
            str(SOURCE / AUDITOR_RELATIVE),
            "--output-root",
            str(producer_root),
            "--protocol",
            str(SOURCE / PROTOCOL_RELATIVE),
            "--audit-output",
            str(audit_path),
        ],
        cwd=SOURCE,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED"
        or audit.get("validation_predictions_per_arm") != 371
        or audit.get("physical_prediction_maps_verified") != 742
        or audit.get("pair_freeze_sha256") != PAIR_FREEZE_SHA256
        or audit.get("validation_gt_read") is not False
        or audit.get("consumer_trained") is not False
        or audit.get("test_evaluated") is not False
    ):
        raise RuntimeError("S8 corrected independent audit result mismatch")
    manifest = {
        "status": "CORRECTED_INDEPENDENT_GT_BLIND_AUDIT_PASS",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "producer_kernel": PRODUCER_KERNEL,
        "producer_kernel_version": PRODUCER_KERNEL_VERSION,
        "runtime": {"python": platform.python_version(), **t4},
        "checkout_commit": CHECKOUT_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "correction_sha256": CORRECTION_SHA256,
        "auditor_sha256": AUDITOR_SHA256,
        "prediction_pair_freeze_sha256": PAIR_FREEZE_SHA256,
        "audit_launch_binding_sha256": hash_file(binding_path),
        "independent_gt_blind_output_audit_sha256": hash_file(audit_path),
        "prediction_changed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "collaborator_output_accessed": False,
    }
    (OUTPUT / "corrected_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
