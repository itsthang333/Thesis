"""Bind the frozen S8 Kaggle wrapper exactly once before launch."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "718bc39c2aef0f6d96892238d591474acf1d1e481dab3d70ddbd4e61ede0fab8"
TEMPLATE_PATH = (
    "project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_v1.py"
)
PROTOCOL_PATH = (
    "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json"
)
PROTOCOL_SHA256 = "7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07"
ADDENDUM_PATH = (
    "artifacts/research_protocols/"
    "skelex_reconstruction_selector_s8_v1_auditor_completeness_addendum.json"
)
ADDENDUM_SHA256 = "dabee40fc3b607df3f82105ab9122b2b80b37c2305d7c5ed17c1c8ae1c3dca0e"
SOURCE_COMMIT = "b4543aeb9430345c9b789384943bd218816a85dd"
CORRECTION_SOURCE_COMMIT = "bc7816ff6cee5a7c5e954668d1255d1b1ad04533"
EXPERIMENT_ID = "EXP-20260802-codex-s8-skelex-reconstruction-randomization-v1"
KERNEL = "itsthang333/btxrd-skelex-reconstruction-selector-s8-v1"


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=root
    ).replace(b"\r\n", b"\n")


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=True,
    )


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
        raise FileExistsError("S8 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")

    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S8 wrapper template SHA-256 mismatch")
    if digest(_git_bytes(repository_root, checkout_commit, TEMPLATE_PATH)) != TEMPLATE_SHA256:
        raise ValueError("S8 wrapper template differs at execution checkout")

    replacements = {
        b"KERNEL_VERSION = 0": f"KERNEL_VERSION = {kernel_version}".encode(),
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{checkout_commit}"'.encode(),
    }
    bound = template_payload
    for old, new in replacements.items():
        if bound.count(old) != 1:
            raise ValueError(f"S8 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S8 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S8 bound wrapper does not invert to exact template")

    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    addendum_payload = _git_bytes(repository_root, checkout_commit, ADDENDUM_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S8 protocol differs at execution checkout")
    if digest(addendum_payload) != ADDENDUM_SHA256:
        raise ValueError("S8 correction addendum differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    addendum = json.loads(addendum_payload.decode("utf-8"))
    if (
        protocol.get("status") != "FROZEN_PRELAUNCH"
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("scientific_source", {}).get("commit") != SOURCE_COMMIT
        or addendum.get("status")
        != "FROZEN_IMPLEMENTATION_AND_AUDIT_COMPLETENESS_CORRECTION"
        or addendum.get("experiment_id") != EXPERIMENT_ID
        or addendum.get("correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or addendum.get("scientific_change") is not False
    ):
        raise ValueError("S8 frozen protocol/addendum contract mismatch")
    _require_ancestor(repository_root, SOURCE_COMMIT, checkout_commit)
    _require_ancestor(repository_root, CORRECTION_SOURCE_COMMIT, checkout_commit)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": EXPERIMENT_ID,
        "kernel": KERNEL,
        "kernel_version": kernel_version,
        "checkout_commit": checkout_commit,
        "scientific_source_commit": SOURCE_COMMIT,
        "correction_source_commit": CORRECTION_SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "auditor_completeness_addendum_sha256": ADDENDUM_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "replacement_count": len(replacements),
        "inverse_reconstruction_matches_template": True,
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
        raise RuntimeError("S8 binding write/read verification failed")
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
