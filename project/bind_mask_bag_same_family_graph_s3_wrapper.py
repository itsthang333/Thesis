from __future__ import annotations

"""Bind the finite S3 Kaggle wrapper exactly once before launch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "774cae093545632ab71de07800fd3642669cd04c713bbcbb4168370e6e30f42d"
PROTOCOL_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_same_family_graph_s3_v1.json"
)
PROTOCOL_SHA256 = "7d7636176fc05d407b51a913170ad780e2d43d328d9437b2d9d2656e191471ca"
NUMERIC_IDENTITY_ADDENDUM_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_same_family_graph_s3_v1_posterror_numeric_identity_addendum.json"
)
NUMERIC_IDENTITY_ADDENDUM_SHA256 = (
    "41e88ae7011c3f994f7d47a6a9216730ba9448ccb6f9fc8599d277a0679f0d51"
)
IMPLEMENTATION_SOURCE_OVERRIDES = {
    "project/run_mask_bag_same_family_graph_s3_arm.py": (
        "30e3048a706127e0cea52892d0e682d97e0c81dc8aee2bd05c4254674fabf6db"
    ),
    "tests/test_run_mask_bag_same_family_graph_s3_arm.py": (
        "66a8f81a0dbb7c150a03d95f27693c4d96544c1858f2621f49591655a98d8440"
    ),
}
SOURCE_COMMIT = "293b013cd036d8346fea3852ec3025772172f32d"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-same-family-graph-s3-v1"


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=root
    ).replace(b"\r\n", b"\n")


def bind(
    template: Path,
    output: Path,
    audit_output: Path,
    *,
    repository_root: Path,
    checkout_commit: str,
    kernel_version: int,
) -> dict[str, Any]:
    if output.exists() or audit_output.exists():
        raise FileExistsError("S3 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S3 wrapper template SHA-256 mismatch")
    replacements = {
        b"KERNEL_VERSION = 0": f"KERNEL_VERSION = {kernel_version}".encode(),
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': (
            f'CHECKOUT_COMMIT = "{checkout_commit}"'.encode()
        ),
    }
    bound = template_payload
    for old, new in replacements.items():
        if bound.count(old) != 1:
            raise ValueError(f"S3 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S3 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S3 bound wrapper does not invert to exact template")
    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S3 protocol differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    addendum_payload = _git_bytes(
        repository_root, checkout_commit, NUMERIC_IDENTITY_ADDENDUM_PATH
    )
    if digest(addendum_payload) != NUMERIC_IDENTITY_ADDENDUM_SHA256:
        raise ValueError("S3 numeric identity addendum differs at execution checkout")
    source_hashes = protocol.get("canonical_lf_source_hashes", {})
    effective_source_hashes = {
        relative: IMPLEMENTATION_SOURCE_OVERRIDES.get(relative, expected)
        for relative, expected in source_hashes.items()
    }
    for relative, expected in effective_source_hashes.items():
        if digest(_git_bytes(repository_root, checkout_commit, relative)) != expected:
            raise ValueError(f"S3 source differs at execution checkout: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, checkout_commit],
        cwd=repository_root,
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "schema_version": 2,
        "status": "FROZEN_PRELAUNCH",
        "kernel": KERNEL,
        "kernel_version": kernel_version,
        "checkout_commit": checkout_commit,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "numeric_identity_addendum_sha256": NUMERIC_IDENTITY_ADDENDUM_SHA256,
        "implementation_only_source_overrides": IMPLEMENTATION_SOURCE_OVERRIDES,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "runtime_source_hashes": effective_source_hashes,
        "replacement_count": len(replacements),
        "inverse_reconstruction_matches_template": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if (
        canonical_bytes(output) != bound
        or json.loads(audit_output.read_text(encoding="utf-8")) != binding
    ):
        raise RuntimeError("S3 binding write/read verification failed")
    return binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-binding", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--checkout-commit", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    args = parser.parse_args()
    result = bind(
        args.template.resolve(),
        args.output.resolve(),
        args.launch_binding.resolve(),
        repository_root=args.repository_root.resolve(),
        checkout_commit=args.checkout_commit,
        kernel_version=args.kernel_version,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
