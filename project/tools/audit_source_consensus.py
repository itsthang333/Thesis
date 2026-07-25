from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import audit_proposal_teacher as base
from audit_wsl_gt_pair import sha256_file


EXPECTED_SOURCE_COMMIT = "80443fddec1cea8333905dff8650f5a2eeacff5d"
EXPECTED_WRAPPER_SHA256 = (
    "ed75e59158e2014532cecba50909c3eca0e72fe8c0c96be0abf6cda76f79b3dc"
)
EXPECTED_SOURCE_HASHES = {
    "generate_pseudo_masks.py": (
        "320cdaceec4def738dec0b7a6b5d57efc52e01107b9f769a4b3265207cdfe15a"
    ),
    "pseudo/mask_selection.py": (
        "8aa289c857a1ce0a98725c44fbee8a35776745bc2244214267b3a882683448f2"
    ),
}
EXPECTED_WEIGHTS = {
    "cam_density": 0.25,
    "global_cam_mass_coverage": 0.15,
    "within_prompt_sam_rank": 0.15,
    "component_local_source_coverage": 0.25,
    "source_density": 0.10,
    "opposite_source_max_iou": 0.10,
}


def validate_source_consensus_contract(
    predeclared: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen = predeclared.get("frozen")
    if not isinstance(frozen, dict):
        raise ValueError("Predeclared frozen source-consensus contract is absent")
    if frozen.get("source_consensus_weights") != EXPECTED_WEIGHTS:
        raise ValueError("Source-consensus weights differ from the predeclared values")
    if frozen.get("support_clip") != "unchanged CAM-only support with kernel 5":
        raise ValueError("Source-consensus support-clip contract changed")
    if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("Source-consensus source commit mismatch")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("Source-consensus source hashes are absent")
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if source_hashes.get(path) != expected:
            raise ValueError(f"Source-consensus source hash mismatch: {path}")
    generation = manifest.get("commands", {}).get("generate")
    if not isinstance(generation, list):
        raise ValueError("Source-consensus generation command is absent")
    selector_positions = [
        index for index, value in enumerate(generation) if value == "--selection-method"
    ]
    if len(selector_positions) != 1:
        raise ValueError("Generation command must declare one selector")
    index = selector_positions[0]
    if index + 1 >= len(generation) or generation[index + 1] != "source_consensus":
        raise ValueError("Generation command did not use source_consensus")
    if "--proposal-teacher-segmentation-checkpoint" not in generation:
        raise ValueError("Source-consensus run omitted the frozen proposal teacher")
    return {
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_hashes": EXPECTED_SOURCE_HASHES,
        "weights": EXPECTED_WEIGHTS,
        "support_clip": frozen["support_clip"],
        "status": "PASS",
    }


def audit_source_consensus(
    root: Path,
    split_manifest: Path,
    baseline_per_image: Path,
    baseline_prompt_quality: Path,
    wrapper: Path,
    *,
    expected_candidate_sha256: str,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    root = root.resolve()
    wrapper = wrapper.resolve()
    if not wrapper.is_file():
        raise FileNotFoundError(wrapper)
    wrapper_sha256 = sha256_file(wrapper)
    if wrapper_sha256 != EXPECTED_WRAPPER_SHA256:
        raise ValueError("Source-consensus wrapper SHA-256 mismatch")

    predeclared_path = root / "predeclared_method.json"
    manifest_path = root / "run_manifest.json"
    if not predeclared_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Source-consensus predeclaration/manifest is incomplete")
    predeclared = json.loads(predeclared_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = validate_source_consensus_contract(predeclared, manifest)

    original_source_commit = base.EXPECTED["source_commit"]
    original_metadata = dict(base.FROZEN_GENERATION_METADATA)
    try:
        base.EXPECTED["source_commit"] = EXPECTED_SOURCE_COMMIT
        base.FROZEN_GENERATION_METADATA.update(
            {
                "selection_method": "source_consensus",
                "proposal_teacher_semantics": (
                    "proposal_components_plus_source_consensus_scoring; "
                    "CAM support clipping unchanged"
                ),
            }
        )
        result = base.audit_proposal_teacher(
            root,
            split_manifest,
            baseline_per_image,
            baseline_prompt_quality,
            expected_candidate_sha256=expected_candidate_sha256,
            iterations=iterations,
            seed=seed,
        )
    finally:
        base.EXPECTED["source_commit"] = original_source_commit
        base.FROZEN_GENERATION_METADATA.clear()
        base.FROZEN_GENERATION_METADATA.update(original_metadata)
    result["audit_role"] = "source-consensus selector validation"
    result["source_consensus_contract"] = contract
    result["wrapper_verification"] = {
        "path": str(wrapper),
        "sha256": wrapper_sha256,
        "status": "PASS",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--baseline-prompt-quality", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    result = audit_source_consensus(
        args.root,
        args.split_manifest,
        args.baseline_per_image,
        args.baseline_prompt_quality,
        args.wrapper,
        expected_candidate_sha256=args.expected_candidate_sha256,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
