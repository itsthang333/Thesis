"""Static, no-input prelaunch audit for corrected S8 audit-only packages."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


KERNEL = "itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1"
DATASET = "itsthang333/btxrd-skelex-s8-v1-frozen-output"
TEMPLATE_PATH = "project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_audit_v1.py"
TEMPLATE_SHA256 = "5c16ec24124c8871f15567c8e8cd6ff9a272747e4c5332af8a58f74f3940654b"
PROTOCOL_PATH = "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json"
PROTOCOL_SHA256 = "7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07"
CORRECTION_PATH = "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_serialized_lcb_audit_correction.json"
CORRECTION_SHA256 = "94e5881f763cc2cb3bd0a3f49cb563f2449140a7c576211252a45579597fc8a2"
TRANSPORT_CORRECTION_PATH = "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_audit_transport_correction.json"
TRANSPORT_CORRECTION_SHA256 = "ee42bbe43d4f81ffba570a8aa46454cb55acbf9bb6338ed4d746aaf38ce32d1d"
NULL_DEVICE_CORRECTION_PATH = "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_null_device_audit_correction.json"
NULL_DEVICE_CORRECTION_SHA256 = "be1bb0bf1c253ded4999e78fea164abbfb1c4e1ae412e94e55b8ba5fe8e03725"
AUDITOR_PATH = "project/audit_skelex_reconstruction_selector_s8_output.py"
AUDITOR_SHA256 = "043d28da1d5dd206eed824191562d66731968bcc4a34e5ccd72f8ce756dd608c"
AUDITOR_TEST_PATH = "tests/test_audit_skelex_reconstruction_selector_s8_output.py"
AUDITOR_TEST_SHA256 = "7e2a0c4ba34ecc287f565125c21f56c379db7d2515b356e42654da820cdb4555"
ARCHIVE_SHA256 = "c516437824ff7d7e32594bfe02e3f654d98d9976d2ddb40595641bf5f8ca1737"
PAIR_SHA256 = "b2cfd59fb01046f445d098790efa5a0fdc649bbc80f565439ba51c5cd453fa00"
ANCESTORS = (
    "b4543aeb9430345c9b789384943bd218816a85dd",
    "969327c4fbbd635fff2e3a00d34d533af8a3c340",
    "16f1b61ef99e866dcfced826b4b2ffb76fb0d3b5",
    "709819a684e60371f464c373de10e74fe7ccb5b2",
    "b51e248bf403f8e5af7d5d7ad23e0a65ee62f5d7",
)


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def git_bytes(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root).replace(b"\r\n", b"\n")


def audit(package: Path, repository_root: Path, expected_kernel_version: int) -> dict[str, object]:
    if expected_kernel_version < 2:
        raise ValueError("S8 corrected audit-only kernel version must be at least 2")
    wrapper = package / "btxrd-skelex-reconstruction-selector-s8-audit-v1.py"
    binding_path = package / "launch_binding.json"
    metadata_path = package / "kernel-metadata.json"
    if not all(path.is_file() for path in (wrapper, binding_path, metadata_path)):
        raise FileNotFoundError("S8 audit-only v2 package is incomplete")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkout = binding.get("checkout_commit")
    if not isinstance(checkout, str) or len(checkout) != 40:
        raise ValueError("S8 audit-only checkout is not frozen")
    required_binding = {
        "status": "FROZEN_PRELAUNCH",
        "kernel": KERNEL,
        "kernel_version": expected_kernel_version,
        "protocol_sha256": PROTOCOL_SHA256,
        "correction_sha256": CORRECTION_SHA256,
        "transport_correction_sha256": TRANSPORT_CORRECTION_SHA256,
        "null_device_correction_sha256": NULL_DEVICE_CORRECTION_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "transport_dataset": DATASET,
        "transport_archive_sha256": ARCHIVE_SHA256,
        "inverse_reconstruction_matches_template": True,
        "prediction_changed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    if any(binding.get(key) != value for key, value in required_binding.items()):
        raise ValueError("S8 audit-only v2 binding mismatch")
    if (
        metadata.get("id") != KERNEL
        or metadata.get("code_file") != wrapper.name
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not True
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or metadata.get("dataset_sources") != [DATASET]
        or metadata.get("kernel_sources") != []
        or metadata.get("competition_sources") != []
        or metadata.get("model_sources") != []
    ):
        raise ValueError("S8 audit-only v2 metadata/source mismatch")
    bound = canonical_bytes(wrapper)
    if digest(bound) != binding.get("bound_wrapper_sha256"):
        raise ValueError("S8 audit-only bound wrapper hash mismatch")
    replacements = {
        f"KERNEL_VERSION = {expected_kernel_version}".encode(): b"KERNEL_VERSION = 0",
        b"LAUNCH_BINDING_READY = True": b"LAUNCH_BINDING_READY = False",
        f'CHECKOUT_COMMIT = "{checkout}"'.encode(): b'CHECKOUT_COMMIT = "UNBOUND"',
    }
    reconstructed = bound
    for old, new in replacements.items():
        if reconstructed.count(old) != 1:
            raise ValueError(f"S8 audit-only inverse binding count mismatch: {old!r}")
        reconstructed = reconstructed.replace(old, new)
    if digest(reconstructed) != TEMPLATE_SHA256:
        raise ValueError("S8 audit-only bound wrapper does not reconstruct template")
    checks = {
        TEMPLATE_PATH: TEMPLATE_SHA256,
        PROTOCOL_PATH: PROTOCOL_SHA256,
        CORRECTION_PATH: CORRECTION_SHA256,
        TRANSPORT_CORRECTION_PATH: TRANSPORT_CORRECTION_SHA256,
        NULL_DEVICE_CORRECTION_PATH: NULL_DEVICE_CORRECTION_SHA256,
        AUDITOR_PATH: AUDITOR_SHA256,
        AUDITOR_TEST_PATH: AUDITOR_TEST_SHA256,
    }
    for relative, expected in checks.items():
        if digest(git_bytes(repository_root, checkout, relative)) != expected:
            raise ValueError(f"S8 audit-only checkout hash mismatch: {relative}")
    for ancestor in ANCESTORS:
        subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, checkout], cwd=repository_root, check=True)
    text = bound.decode("utf-8")
    required_tokens = (
        f'TRANSPORT_DATASET = "{DATASET}"',
        f'TRANSPORT_ARCHIVE_SHA256 = "{ARCHIVE_SHA256}"',
        f'PAIR_FREEZE_SHA256 = "{PAIR_SHA256}"',
        "safe_extract_transport_archive",
        '"validation_gt_read": False',
        '"consumer_trained": False',
        '"test_evaluated": False',
    )
    if any(token not in text for token in required_tokens):
        raise ValueError("S8 audit-only v2 safety token missing")
    return {
        "schema_version": 1,
        "status": f"S8_AUDIT_ONLY_V{expected_kernel_version}_FROZEN_PRELAUNCH_PASS",
        "kernel": KERNEL,
        "kernel_version": expected_kernel_version,
        "checkout_commit": checkout,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "launch_binding_sha256": digest(canonical_bytes(binding_path)),
        "metadata_sha256": digest(canonical_bytes(metadata_path)),
        "protocol_sha256": PROTOCOL_SHA256,
        "auditor_correction_sha256": CORRECTION_SHA256,
        "transport_correction_sha256": TRANSPORT_CORRECTION_SHA256,
        "null_device_correction_sha256": NULL_DEVICE_CORRECTION_SHA256,
        "corrected_auditor_sha256": AUDITOR_SHA256,
        "corrected_auditor_test_sha256": AUDITOR_TEST_SHA256,
        "transport_dataset": DATASET,
        "transport_archive_sha256": ARCHIVE_SHA256,
        "kernel_sources": [],
        "t4x2_guard_declared": True,
        "scientific_change": False,
        "prediction_changed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "collaborator_output_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    args = parser.parse_args()
    result = audit(
        args.package.resolve(),
        args.repository_root.resolve(),
        args.kernel_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
