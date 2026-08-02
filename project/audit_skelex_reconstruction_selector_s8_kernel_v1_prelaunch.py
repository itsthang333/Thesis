"""Static, no-input prelaunch audit for the bound S8 Kaggle package."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


EXPERIMENT_ID = "EXP-20260802-codex-s8-skelex-reconstruction-randomization-v1"
TEMPLATE_RELATIVE = Path("project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_v1.py")
PROTOCOL_RELATIVE = Path("artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json")
ADDENDUM_RELATIVE = Path("artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_auditor_completeness_addendum.json")
SOURCE_RELATIVES = (
    Path("project/audit_skelex_reconstruction_selector_s8_output.py"),
    Path("project/run_skelex_reconstruction_selector_s8.py"),
    Path("project/models/skelex_reconstruction_selector.py"),
    Path("tests/test_skelex_reconstruction_selector.py"),
)
TEMPLATE_SHA256 = "718bc39c2aef0f6d96892238d591474acf1d1e481dab3d70ddbd4e61ede0fab8"
PROTOCOL_SHA256 = "7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07"
ADDENDUM_SHA256 = "dabee40fc3b607df3f82105ab9122b2b80b37c2305d7c5ed17c1c8ae1c3dca0e"
SOURCE_COMMIT = "b4543aeb9430345c9b789384943bd218816a85dd"
CORRECTION_SOURCE_COMMIT = "bc7816ff6cee5a7c5e954668d1255d1b1ad04533"


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def git_bytes(root: Path, commit: str, relative: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root
    ).replace(b"\r\n", b"\n")


def audit(package: Path, repository_root: Path) -> dict[str, Any]:
    wrapper = package / "btxrd-skelex-reconstruction-selector-s8-v1.py"
    binding_path = package / "launch_binding.json"
    metadata_path = package / "kernel-metadata.json"
    if not all(path.is_file() for path in (wrapper, binding_path, metadata_path)):
        raise FileNotFoundError("S8 package is incomplete")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkout = binding.get("checkout_commit")
    if not isinstance(checkout, str) or len(checkout) != 40:
        raise ValueError("S8 binding checkout is not frozen")
    if (
        binding.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("kernel_version") != 1
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or binding.get("correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("auditor_completeness_addendum_sha256") != ADDENDUM_SHA256
        or binding.get("inverse_reconstruction_matches_template") is not True
        or binding.get("validation_gt_read") is not False
        or binding.get("consumer_trained") is not False
        or binding.get("test_evaluated") is not False
        or binding.get("collaborator_output_accessed") is not False
    ):
        raise ValueError("S8 binding safety/provenance mismatch")
    if (
        metadata.get("id") != "itsthang333/btxrd-skelex-reconstruction-selector-s8-v1"
        or metadata.get("code_file") != wrapper.name
        or metadata.get("enable_gpu") is not True
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or metadata.get("is_private") is not True
        or metadata.get("enable_internet") is not True
    ):
        raise ValueError("S8 Kaggle metadata mismatch")
    if digest(canonical_bytes(wrapper)) != binding.get("bound_wrapper_sha256"):
        raise ValueError("S8 bound wrapper hash mismatch")
    template = git_bytes(repository_root, checkout, TEMPLATE_RELATIVE)
    if digest(template) != TEMPLATE_SHA256:
        raise ValueError("S8 template differs at checkout")
    if digest(git_bytes(repository_root, checkout, PROTOCOL_RELATIVE)) != PROTOCOL_SHA256:
        raise ValueError("S8 protocol differs at checkout")
    if digest(git_bytes(repository_root, checkout, ADDENDUM_RELATIVE)) != ADDENDUM_SHA256:
        raise ValueError("S8 addendum differs at checkout")
    source_hashes = {}
    for relative in SOURCE_RELATIVES:
        source_hashes[relative.as_posix()] = digest(git_bytes(repository_root, checkout, relative))
    subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, checkout], cwd=repository_root, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", CORRECTION_SOURCE_COMMIT, checkout], cwd=repository_root, check=True)
    wrapper_text = canonical_bytes(wrapper).decode("utf-8")
    required = (
        "KERNEL_VERSION = 1",
        "LAUNCH_BINDING_READY = True",
        f'CHECKOUT_COMMIT = "{checkout}"',
        '"validation_gt_read": False',
        '"consumer_trained": False',
        '"test_evaluated": False',
    )
    if any(token not in wrapper_text for token in required):
        raise ValueError("S8 bound wrapper safety markers missing")
    return {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": EXPERIMENT_ID,
        "kernel": metadata["id"],
        "kernel_version": 1,
        "checkout_commit": checkout,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(canonical_bytes(wrapper)),
        "launch_binding_sha256": digest(canonical_bytes(binding_path)),
        "metadata_sha256": digest(canonical_bytes(metadata_path)),
        "protocol_sha256": PROTOCOL_SHA256,
        "auditor_completeness_addendum_sha256": ADDENDUM_SHA256,
        "source_hashes_at_checkout": source_hashes,
        "t4x2_guard_declared": True,
        "real_cache_opened": False,
        "btxrd_image_opened": False,
        "skelex_inference_run": False,
        "validation_prediction_created": False,
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
    args = parser.parse_args()
    result = audit(args.package.resolve(), args.repository_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
