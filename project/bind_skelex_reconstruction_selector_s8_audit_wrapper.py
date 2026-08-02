"""Bind the S8 audit-only correction wrapper exactly once."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "e596c57c4e425d195ec7b732cf9c329cc8b6013f1f0006ac7de7cddcb7827d9c"
TEMPLATE_PATH = "project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_audit_v1.py"
PROTOCOL_PATH = "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json"
PROTOCOL_SHA256 = "7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07"
CORRECTION_PATH = "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_serialized_lcb_audit_correction.json"
CORRECTION_SHA256 = "94e5881f763cc2cb3bd0a3f49cb563f2449140a7c576211252a45579597fc8a2"
AUDITOR_CORRECTION_COMMIT = "969327c4fbbd635fff2e3a00d34d533af8a3c340"
CORRECTION_ADDENDUM_COMMIT = "16f1b61ef99e866dcfced826b4b2ffb76fb0d3b5"
KERNEL = "itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1"
TRANSPORT_DATASET = "itsthang333/btxrd-skelex-s8-v1-frozen-output"
TRANSPORT_ARCHIVE_SHA256 = "c516437824ff7d7e32594bfe02e3f654d98d9976d2ddb40595641bf5f8ca1737"


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
    binding_output: Path,
    *,
    repository_root: Path,
    checkout_commit: str,
    kernel_version: int,
) -> dict[str, Any]:
    if output.exists() or binding_output.exists():
        raise FileExistsError("S8 audit-only bound wrapper or binding already exists")
    if len(checkout_commit) != 40 or any(character not in "0123456789abcdef" for character in checkout_commit):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S8 audit-only wrapper template SHA-256 mismatch")
    if digest(_git_bytes(repository_root, checkout_commit, TEMPLATE_PATH)) != TEMPLATE_SHA256:
        raise ValueError("S8 audit-only template differs at checkout")
    replacements = {
        b"KERNEL_VERSION = 0": f"KERNEL_VERSION = {kernel_version}".encode(),
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{checkout_commit}"'.encode(),
    }
    bound = template_payload
    for old, new in replacements.items():
        if bound.count(old) != 1:
            raise ValueError(f"S8 audit-only replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S8 audit-only inverse count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S8 audit-only wrapper does not invert to template")
    if digest(_git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)) != PROTOCOL_SHA256:
        raise ValueError("S8 audit-only protocol differs at checkout")
    if digest(_git_bytes(repository_root, checkout_commit, CORRECTION_PATH)) != CORRECTION_SHA256:
        raise ValueError("S8 audit-only correction differs at checkout")
    for ancestor in (AUDITOR_CORRECTION_COMMIT, CORRECTION_ADDENDUM_COMMIT):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, checkout_commit],
            cwd=repository_root,
            check=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "status": "FROZEN_PRELAUNCH",
        "kernel": KERNEL,
        "kernel_version": kernel_version,
        "checkout_commit": checkout_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "correction_sha256": CORRECTION_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "transport_dataset": TRANSPORT_DATASET,
        "transport_archive_sha256": TRANSPORT_ARCHIVE_SHA256,
        "inverse_reconstruction_matches_template": True,
        "prediction_changed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    binding_output.parent.mkdir(parents=True, exist_ok=True)
    binding_output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
