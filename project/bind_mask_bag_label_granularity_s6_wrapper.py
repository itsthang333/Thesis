from __future__ import annotations

"""Bind the finite S6 Kaggle wrapper exactly once before launch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TEMPLATE_SHA256 = "c04c288501b95f0408c21e9e5cb4eb6bfcb1af159b82572c3eb26d63acf17492"
PROTOCOL_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_label_granularity_s6_v1.json"
)
PROTOCOL_SHA256 = "f4e17d24dfab36f01526550c7dc306fc7549494acc4545153454c61ae926bfc3"
SOURCE_COMMIT = "543ee89654a0ed00e80ded16924a760585337924"
CORRECTION_SOURCE_COMMIT = "7ca2f4dec72af5f509e52786980321d255a7eb68"
CORRECTION_PATH = (
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_label_granularity_s6_v1_map_audit_numeric_correction.json"
)
CORRECTION_SHA256 = "b0dca40bf4f8bd933a902facb7bfdf5ec393c429672b0beb0b0594f2d15dfc63"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-label-granularity-s6-v1"


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
        raise FileExistsError("S6 bound wrapper or launch binding already exists")
    if len(checkout_commit) != 40 or any(
        character not in "0123456789abcdef" for character in checkout_commit
    ):
        raise ValueError("checkout_commit must be a lowercase 40-character Git hash")
    if kernel_version < 1:
        raise ValueError("kernel_version must be positive")
    template_payload = canonical_bytes(template)
    if digest(template_payload) != TEMPLATE_SHA256:
        raise ValueError("S6 wrapper template SHA-256 mismatch")
    replacements = {
        b"KERNEL_VERSION = 0": f"KERNEL_VERSION = {kernel_version}".encode(),
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{checkout_commit}"'.encode(),
    }
    bound = template_payload
    for old, new in replacements.items():
        if bound.count(old) != 1:
            raise ValueError(f"S6 template replacement count differs: {old!r}")
        bound = bound.replace(old, new)
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        if reconstructed.count(new) != 1:
            raise ValueError(f"S6 inverse replacement count differs: {new!r}")
        reconstructed = reconstructed.replace(new, old)
    if reconstructed != template_payload:
        raise ValueError("S6 bound wrapper does not invert to exact template")

    protocol_payload = _git_bytes(repository_root, checkout_commit, PROTOCOL_PATH)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S6 protocol differs at execution checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    if protocol.get("scientific_source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("S6 protocol scientific-source commit mismatch")
    correction_payload = _git_bytes(
        repository_root, checkout_commit, CORRECTION_PATH
    )
    if digest(correction_payload) != CORRECTION_SHA256:
        raise ValueError("S6 auditor correction differs at execution checkout")
    correction = json.loads(correction_payload.decode("utf-8"))
    overrides = correction.get("allowed_canonical_lf_source_overrides")
    if (
        correction.get("status")
        != "FROZEN_IMPLEMENTATION_ONLY_AUDITOR_CORRECTION"
        or correction.get("correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or correction.get("protocol_sha256_unchanged") != PROTOCOL_SHA256
        or correction.get("validation_gt_read") is not False
        or correction.get("consumer_trained") is not False
        or correction.get("test_evaluated") is not False
        or not isinstance(overrides, dict)
        or set(overrides)
        != {
            "project/audit_mask_bag_label_granularity_s6_output.py",
            "tests/test_audit_mask_bag_label_granularity_s6_output.py",
        }
    ):
        raise ValueError("S6 auditor correction contract mismatch")
    all_hashes: dict[str, str] = {}
    for section in ("canonical_lf_source_hashes", "post_freeze_only_source_hashes"):
        hashes = protocol.get(section, {})
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"S6 protocol {section} is missing")
        for relative, expected in hashes.items():
            actual = digest(_git_bytes(repository_root, checkout_commit, relative))
            override = overrides.get(relative)
            if override is None:
                if actual != expected:
                    raise ValueError(
                        f"S6 source differs at execution checkout: {relative}"
                    )
            elif (
                section != "canonical_lf_source_hashes"
                or override.get("protocol_v1_sha256") != expected
                or override.get("corrected_sha256") != actual
            ):
                raise ValueError(
                    f"S6 corrected source differs at execution checkout: {relative}"
                )
            if section == "canonical_lf_source_hashes":
                all_hashes[relative] = actual
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, checkout_commit],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            CORRECTION_SOURCE_COMMIT,
            checkout_commit,
        ],
        cwd=repository_root,
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bound)
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260802-codex-s6-label-granularity-mil-v1",
        "kernel": KERNEL,
        "kernel_version": kernel_version,
        "checkout_commit": checkout_commit,
        "scientific_source_commit": SOURCE_COMMIT,
        "auditor_correction_source_commit": CORRECTION_SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "auditor_numeric_correction_sha256": CORRECTION_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(bound),
        "source_hashes": all_hashes,
        "replacement_count": len(replacements),
        "inverse_reconstruction_matches_template": True,
        "validation_subtype_label_used_for_routing": False,
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
        raise RuntimeError("S6 binding write/read verification failed")
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
