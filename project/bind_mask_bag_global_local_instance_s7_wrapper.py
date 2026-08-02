from __future__ import annotations

"""Bind the frozen S7 Kaggle wrapper exactly once before launch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "ea51733fbf2a0d0a35db55678208c23fc8d360eeecdce887320a6f655c446307"
TEMPLATE_PATH = (
    "project/kaggle_wrappers/run_mask_bag_global_local_instance_s7_v1.py"
)
PROTOCOL_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_global_local_instance_s7_v1.json"
)
PROTOCOL_SHA256 = "81fbb2f40af3a49e4653a15d298858c973e88524dea06fc42c9095cec55579a1"
SOURCE_COMMIT = "0e524807937e6fb6effde1649993825f3923c43f"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-global-local-instance-s7-v1"


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
        raise FileExistsError("S7 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S7 wrapper template SHA-256 mismatch")
    if digest(_git_bytes(repository_root, checkout_commit, TEMPLATE_PATH)) != TEMPLATE_SHA256:
        raise ValueError("S7 wrapper template differs at execution checkout")
    replacements = {
        b"KERNEL_VERSION = 0": f"KERNEL_VERSION = {kernel_version}".encode(),
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{checkout_commit}"'.encode(),
    }
    bound = template_payload
    for old, new in replacements.items():
        if bound.count(old) != 1:
            raise ValueError(f"S7 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S7 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S7 bound wrapper does not invert to exact template")

    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S7 protocol differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    if (
        protocol.get("status") != "FROZEN_PRELAUNCH"
        or protocol.get("scientific_source", {}).get("commit") != SOURCE_COMMIT
        or protocol.get("pre_gt_contract", {}).get("target_snapshots") != 40
        or protocol.get("pre_gt_contract", {}).get(
            "accepted_bag_probability_preserved"
        )
        is not True
    ):
        raise ValueError("S7 protocol source/output contract mismatch")
    source_hashes: dict[str, str] = {}
    for section in ("canonical_lf_source_hashes", "post_freeze_only_source_hashes"):
        hashes = protocol.get(section, {})
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"S7 protocol {section} is missing")
        for relative, expected in hashes.items():
            actual = digest(_git_bytes(repository_root, checkout_commit, relative))
            if actual != expected:
                raise ValueError(
                    f"S7 source differs at execution checkout: {relative}"
                )
            if section == "canonical_lf_source_hashes":
                source_hashes[relative] = actual
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, checkout_commit],
        cwd=repository_root,
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260802-codex-s7-global-local-instance-v1",
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
        "accepted_bag_probability_preserved": True,
        "collaborator_output_accessed": False,
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
        raise RuntimeError("S7 binding write/read verification failed")
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
