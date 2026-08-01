from __future__ import annotations

"""Bind the finite S4 Kaggle wrapper exactly once before launch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "af550929841b305f4612d9ba6581e214d9964ec3454c86fbd7c770f6dcefe4db"
PROTOCOL_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_proposal_cluster_s4_v1.json"
)
PROTOCOL_SHA256 = "040227de1347c45bc1823bd5aef5d9614b8005619ecc35d9dceb45bb7eba71e8"
SOURCE_COMMIT = "95c4a3378eaf8463c57d57a0dd4e4cac6c69021f"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-proposal-cluster-s4-v1"
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
        raise FileExistsError("S4 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S4 wrapper template SHA-256 mismatch")
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
            raise ValueError(f"S4 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S4 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S4 bound wrapper does not invert to exact template")
    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S4 protocol differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    all_hashes = protocol.get("canonical_lf_source_hashes", {})
    if not isinstance(all_hashes, dict) or not all_hashes:
        raise ValueError("S4 protocol source inventory missing")
    for relative, expected in all_hashes.items():
        if digest(_git_bytes(repository_root, checkout_commit, relative)) != expected:
            raise ValueError(f"S4 source differs at execution checkout: {relative}")
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
        "experiment_id": "EXP-20260801-codex-s4-oof-proposal-cluster-v1",
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
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if (
        canonical_bytes(output) != bound
        or json.loads(binding_output.read_text(encoding="utf-8")) != binding
    ):
        raise RuntimeError("S4 binding write/read verification failed")
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
