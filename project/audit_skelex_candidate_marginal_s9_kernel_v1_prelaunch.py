"""Static, no-input prelaunch audit for the bound S9 Kaggle package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


EXPERIMENT_ID = "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
KERNEL = "itsthang333/btxrd-skelex-candidate-marginal-s9-v1"
TEMPLATE_RELATIVE = Path(
    "project/kaggle_wrappers/run_skelex_candidate_marginal_s9_v1.py"
)
PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/skelex_candidate_marginal_s9_v1.json"
)
CORRECTION_RELATIVE = Path(
    "artifacts/research_protocols/"
    "skelex_candidate_marginal_s9_v1_rank_exactness_correction.json"
)
TEMPLATE_SHA256 = "cdfffccb67ac3a7de88b6e1406dac0e89760e49a7904b20d563c5003b3ee919a"
PROTOCOL_SHA256 = "0a303c9c86c3c43c750c85a50087e792bf0942a0b43fc9a1cf9e143c4832ee3d"
SOURCE_COMMIT = "7dcd6c6f055c69f3f048a005ed2fea6177dc7ed8"
CORRECTION_SHA256 = "0ddf17d73c9ddcf24799827a075f41a32e671e15894ae3d6d0780a278edb11a9"
CORRECTION_SOURCE_COMMIT = "cb608cd8ca501e840d4ae7c73cc7592187683a27"
DATASET_SOURCES = [
    "itsthang333/btxrd-raw",
    "itsthang333/btxrd-mask-bag-geometry-v3-train-gallery-v1",
    "itsthang333/btxrd-mask-bag-selector-baseline-v1",
]
KERNEL_SOURCES = ["itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1"]
EXPECTED_SOURCE_HASHES = {
    "SKELEX_CANDIDATE_MARGINAL_S9_DESIGN.md": "326b391c01c63c0b9b50785feb57d5c411cc09302c2d897162a0dddac33bc1c6",
    "project/models/skelex_candidate_marginal.py": "3cd7359e6f1545195b51b1e262e80233d77586049f4b08fba02233a6637c69db",
    "project/run_skelex_candidate_marginal_s9.py": "2310b6ff47b30d2872a89d6da4f331580a5041b3a69be3a82a13442dd95dafb4",
    "project/audit_skelex_candidate_marginal_s9_output.py": "1ee3a18399ba1780912dbffecc5d51ce27861fce45bb89e9fb7d5883e73288fe",
    "tests/test_skelex_candidate_marginal.py": "624c274720380d6314f253206383de72d7c9a99ee4d69c66f5c1edfff8d8c505",
    "tests/test_run_skelex_candidate_marginal_s9.py": "0844c23d54beccf0dcbfd29e14de4ee1490d67bf4178bd3764c9ab7773dcf7ed",
    "tests/test_audit_skelex_candidate_marginal_s9_output.py": "2cab6273f90279f162a846575654614870cc0e0bcb8c2fe24201bdd33252ce8f",
}
EXPECTED_KERNEL_VERSION = 2


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def git_bytes(root: Path, commit: str, relative: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root
    ).replace(b"\r\n", b"\n")


def audit(package: Path, repository_root: Path) -> dict[str, Any]:
    wrapper = package / "btxrd-skelex-candidate-marginal-s9-v1.py"
    binding_path = package / "launch_binding.json"
    metadata_path = package / "kernel-metadata.json"
    if not all(path.is_file() for path in (wrapper, binding_path, metadata_path)):
        raise FileNotFoundError("S9 package is incomplete")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkout = binding.get("checkout_commit")
    if not isinstance(checkout, str) or len(checkout) != 40:
        raise ValueError("S9 binding checkout is not frozen")
    if (
        binding.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("kernel") != KERNEL
        or binding.get("kernel_version") != EXPECTED_KERNEL_VERSION
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or binding.get("correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("rank_exactness_correction_sha256") != CORRECTION_SHA256
        or binding.get("template_sha256") != TEMPLATE_SHA256
        or binding.get("inverse_reconstruction_matches_template") is not True
        or binding.get("validation_gt_read") is not False
        or binding.get("consumer_trained") is not False
        or binding.get("test_evaluated") is not False
        or binding.get("collaborator_output_accessed") is not False
    ):
        raise ValueError("S9 binding safety/provenance mismatch")
    if (
        metadata.get("id") != KERNEL
        or metadata.get("code_file") != wrapper.name
        or metadata.get("enable_gpu") is not True
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or metadata.get("is_private") is not True
        or metadata.get("enable_internet") is not True
        or metadata.get("dataset_sources") != DATASET_SOURCES
        or metadata.get("kernel_sources") != KERNEL_SOURCES
        or metadata.get("competition_sources") != []
        or metadata.get("model_sources") != []
    ):
        raise ValueError("S9 Kaggle metadata mismatch")
    if len(metadata["id"]) > 80 or len(metadata.get("title", "")) > 80:
        raise ValueError("S9 Kaggle slug/title exceeds preflight guard")
    if digest(canonical_bytes(wrapper)) != binding.get("bound_wrapper_sha256"):
        raise ValueError("S9 bound wrapper hash mismatch")
    if digest(git_bytes(repository_root, checkout, TEMPLATE_RELATIVE)) != TEMPLATE_SHA256:
        raise ValueError("S9 template differs at checkout")
    protocol_payload = git_bytes(repository_root, checkout, PROTOCOL_RELATIVE)
    if digest(protocol_payload) != PROTOCOL_SHA256:
        raise ValueError("S9 protocol differs at checkout")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    correction_payload = git_bytes(repository_root, checkout, CORRECTION_RELATIVE)
    if digest(correction_payload) != CORRECTION_SHA256:
        raise ValueError("S9 correction addendum differs at checkout")
    correction = json.loads(correction_payload.decode("utf-8"))
    if (
        correction.get("correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or correction.get("scientific_protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("S9 correction addendum contract mismatch")
    protocol_hashes = protocol.get("canonical_lf_source_hashes", {})
    if protocol_hashes | EXPECTED_SOURCE_HASHES != protocol_hashes:
        raise ValueError("S9 protocol source closure is incomplete")
    corrected = correction.get("corrected_source_hashes", {})
    effective_hashes = {
        relative: corrected.get(relative, expected)
        for relative, expected in protocol_hashes.items()
    }
    source_hashes = {
        relative: digest(git_bytes(repository_root, checkout, Path(relative)))
        for relative in effective_hashes
    }
    if source_hashes != effective_hashes:
        raise ValueError("S9 source closure differs at checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, checkout],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", CORRECTION_SOURCE_COMMIT, checkout],
        cwd=repository_root,
        check=True,
    )
    wrapper_text = canonical_bytes(wrapper).decode("utf-8")
    required = (
        "KERNEL_VERSION = 1",
        "LAUNCH_BINDING_READY = True",
        f'CHECKOUT_COMMIT = "{checkout}"',
        "verify_t4x2()",
        "run_static_tests()",
        '"validation_gt_read": False',
        '"consumer_trained": False',
        '"test_evaluated": False',
    )
    if any(token not in wrapper_text for token in required):
        raise ValueError("S9 bound wrapper safety markers missing")
    return {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "authorized_launch": True,
        "experiment_id": EXPERIMENT_ID,
        "owner": "Codex central workstream",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "research-wsss-improvement",
        "kernel": KERNEL,
        "kernel_version": EXPECTED_KERNEL_VERSION,
        "checkout_commit": checkout,
        "scientific_source_commit": SOURCE_COMMIT,
        "correction_source_commit": CORRECTION_SOURCE_COMMIT,
        "template_sha256": TEMPLATE_SHA256,
        "bound_wrapper_sha256": digest(canonical_bytes(wrapper)),
        "launch_binding_sha256": digest(canonical_bytes(binding_path)),
        "metadata_sha256": digest(canonical_bytes(metadata_path)),
        "protocol_sha256": PROTOCOL_SHA256,
        "rank_exactness_correction_sha256": CORRECTION_SHA256,
        "source_hashes_at_checkout": source_hashes,
        "target_runtime": {
            "python": "Kaggle image runtime; printed before scientific input",
            "transformers": "4.50.2 exact pip pin and import assertion",
            "compute": "NvidiaTeslaT4 x2",
        },
        "focused_tests": "wrapper executes 35-test static S9 correction suite before model/data",
        "input_contract": {
            "dataset_sources": DATASET_SOURCES,
            "kernel_sources": KERNEL_SOURCES,
            "split_sha256": "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c",
            "cohort": {"train": 2981, "validation": 371},
        },
        "output_contract": {
            "prediction_maps": 742,
            "candidate_scores": 742,
            "likelihood_evidence": 371,
            "pair_freeze_before_validation_gt": True,
        },
        "kpf_evidence": {
            "KPF-001": "Git-LF split plus exact one-time CRLF reconstruction hashes",
            "KPF-002": "transformers==4.50.2 pin/assert before model/data",
            "KPF-003": "Python 3.9 focused static suite",
            "KPF-004": "sys.executable/version and dependency assertion in wrapper",
            "KPF-005": "three-field inverse binder plus checkout ancestry",
            "KPF-006": "slug/title static length guard",
            "KPF-011": "recursive exact-hash unique input locators",
            "KPF-013": "NPY-byte feature hashes and serialization-aware auditor",
            "KPF-015": "seed/epochs/order locked by protocol and wrapper",
            "KPF-016": "two-device real CUDA convolution guard",
            "KPF-017": "no immediate/repeated status check declared",
            "KPF-018": "output and runtime roots use exist_ok=False",
            "KPF-019": "one canonical float32 rank aggregate plus exact vector/winner regressions",
        },
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
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
