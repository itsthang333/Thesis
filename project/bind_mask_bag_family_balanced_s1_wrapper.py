from __future__ import annotations

"""Bind the finite S1 Kaggle wrapper exactly once before launch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "f613b259546552c226f0197e3878442fb6891ed76655b5030b251d24a087a4ba"
PROTOCOL_PATH = (
    "artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1.json"
)
PROTOCOL_SHA256 = "62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413"
SOURCE_COMMIT = "f3da1817ee3491f04e8c86335556762ebc675d8d"
CLAIM_COMMIT = "97db17c16938a8f842546076a26a52e58928b07b"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-family-balanced-s1-pair-v1"


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
        raise FileExistsError("S1 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S1 wrapper template SHA-256 mismatch")
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
            raise ValueError(f"S1 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S1 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S1 bound wrapper does not invert to exact template")

    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S1 protocol differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    source_hashes = protocol.get("scientific_source", {}).get(
        "canonical_lf_source_hashes", {}
    )
    if len(source_hashes) != 12:
        raise ValueError("S1 protocol source closure differs")
    for relative, expected in source_hashes.items():
        if digest(_git_bytes(repository_root, checkout_commit, relative)) != expected:
            raise ValueError(f"S1 source differs at execution checkout: {relative}")
    for ancestor in (SOURCE_COMMIT, CLAIM_COMMIT):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, checkout_commit],
            cwd=repository_root,
            check=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "kernel": KERNEL,
        "kernel_version": kernel_version,
        "checkout_commit": checkout_commit,
        "scientific_source_commit": SOURCE_COMMIT,
        "claim_commit": CLAIM_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "runtime_source_hashes": source_hashes,
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
        raise RuntimeError("S1 binding write/read verification failed")
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
