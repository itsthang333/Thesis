from __future__ import annotations

"""Bind the finite T1 Kaggle wrapper exactly once before launch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "6bbe6a5dda3bb35db88e27b2b1c117b7c8c199ad7a9b004450727ddec9f2c4a0"
PROTOCOL_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_count_controlled_self_paced_t1_v1.json"
)
PROTOCOL_SHA256 = "6a4379e896f3ea3862dce1edcdea20af09a90ec8f9cbbd6eb25bf8eca1306a7c"
SOURCE_COMMIT = "c7f0937d515ded9bbd8928a2236cbe44b7a25f79"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-count-controlled-t1-v1"
POST_FREEZE_ONLY_SOURCE_PATHS = {
    "project/evaluate_mask_bag_selector_arm.py",
    "project/models/mask_bag_ranking_diagnostics.py",
}


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
        raise FileExistsError("T1 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("T1 wrapper template SHA-256 mismatch")
    replacements = {
        b"KERNEL_VERSION = 0": f"KERNEL_VERSION = {kernel_version}".encode(),
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{checkout_commit}"'.encode(),
    }
    bound = template_payload
    for old, new in replacements.items():
        if bound.count(old) != 1:
            raise ValueError(f"T1 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"T1 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("T1 bound wrapper does not invert to exact template")
    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("T1 protocol differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    all_hashes = protocol.get("canonical_lf_source_hashes", {})
    if not isinstance(all_hashes, dict) or not all_hashes:
        raise ValueError("T1 protocol source inventory missing")
    for relative, expected in all_hashes.items():
        if digest(_git_bytes(repository_root, checkout_commit, relative)) != expected:
            raise ValueError(f"T1 source differs at execution checkout: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, checkout_commit],
        cwd=repository_root,
        check=True,
    )
    source_hashes = {
        relative: expected
        for relative, expected in all_hashes.items()
        if relative not in POST_FREEZE_ONLY_SOURCE_PATHS
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260801-codex-t1-count-controlled-self-paced-v1",
        "kernel": KERNEL,
        "kernel_version": kernel_version,
        "checkout_commit": checkout_commit,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "source_hashes": source_hashes,
        "replacement_count": len(replacements),
        "inverse_reconstruction_matches_template": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    binding_output.parent.mkdir(parents=True, exist_ok=True)
    binding_output.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if canonical_bytes(output) != bound or json.loads(
        binding_output.read_text(encoding="utf-8")
    ) != binding:
        raise RuntimeError("T1 binding write/read verification failed")
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
