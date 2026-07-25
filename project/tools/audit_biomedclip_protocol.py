from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_biomedclip_saliency import sha256_file


EXPECTED_PROTOCOL_SHA256 = (
    "aac8ad62fbaa2d15042853e8ad76bcddf5cf38266d442481751ea0dc0a17f078"
)
EXPECTED_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)
EXPECTED_SOURCE = {
    "implementation_commit": "8a997c87170538f897e6aa3b13b0f6c13e39f32f",
    "generate_biomedclip_saliency.py": (
        "c475f3b8bd16b3b2fd85add21cf35e4b631943ab50be9c4217302dba763ed46c"
    ),
    "models/biomedclip_saliency.py": (
        "f07718a47d71c0aa05c6e110c243d3dfc7197064cca89d5e1dfb44612fd28d5d"
    ),
    "generate_pseudo_masks.py": (
        "354621802dfee4fe39d24325a706ac40e7a1dfce5929ced4a35f530bb95a9124"
    ),
}
EXPECTED_WEIGHT_SHA256 = (
    "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
)
EXPECTED_BASELINE = {
    "per_image_sha256": (
        "fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540"
    ),
    "prompt_quality_sha256": (
        "d1b570ae3a6287fdaf7fc5c28aea864d6883e5c57037542b39b17c4c6ea995e4"
    ),
}


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "biomedclip_tiled_val_v1":
        raise ValueError("BiomedCLIP protocol ID mismatch")
    if protocol.get("status") != "predeclared_before_any_validation_prediction_or_metric":
        raise ValueError("BiomedCLIP protocol was not predeclared")
    if protocol.get("test_evaluated") is not False:
        raise ValueError("BiomedCLIP protocol did not lock test")
    if protocol.get("source_lock") != EXPECTED_SOURCE:
        raise ValueError("BiomedCLIP protocol source lock mismatch")
    population = protocol.get("population", {})
    if population != {
        "split": "val",
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "tumor_groups": 167,
        "subgroups": {
            "small_lt_1pct": 94,
            "medium_1_to_5pct": 72,
            "large_ge_5pct": 18,
        },
    }:
        raise ValueError("BiomedCLIP protocol population mismatch")
    localization = protocol.get("localization_source", {})
    if localization.get("model_weight_sha256") != EXPECTED_WEIGHT_SHA256:
        raise ValueError("BiomedCLIP protocol weight hash mismatch")
    if localization.get("output_grid") != [320, 320]:
        raise ValueError("BiomedCLIP protocol grid mismatch")
    if localization.get("saliency_reduction") != (
        "channelwise mean absolute gradient-times-activation"
    ):
        raise ValueError("BiomedCLIP protocol saliency reduction mismatch")
    if localization.get("target_layer") != "model.visual.trunk.blocks[11].norm1":
        raise ValueError("BiomedCLIP protocol target-layer mismatch")
    if protocol.get("promoted_baseline", {}).get(
        "per_image_sha256"
    ) != EXPECTED_BASELINE["per_image_sha256"]:
        raise ValueError("BiomedCLIP protocol baseline per-image hash mismatch")
    if protocol.get("promoted_baseline", {}).get(
        "prompt_quality_sha256"
    ) != EXPECTED_BASELINE["prompt_quality_sha256"]:
        raise ValueError("BiomedCLIP protocol baseline prompt-quality hash mismatch")
    gates = protocol.get("promotion_gates", {})
    if gates.get("localization_source_to_selector_research") != (
        "small_lt_1pct unclipped single-proposal oracle delta CI95 lower bound > 0 "
        "and overall unclipped oracle mean does not decrease"
    ):
        raise ValueError("BiomedCLIP localization gate mismatch")
    if gates.get("direct_train_pseudo_mask_generation") != (
        "overall final-Dice paired complete-group delta CI95 lower bound > 0 and "
        "small_lt_1pct final Dice does not decrease"
    ):
        raise ValueError("BiomedCLIP direct-promotion gate mismatch")
    forbidden = protocol.get("forbidden", [])
    for required in (
        "train polygons or masks",
        "validation GT before predictions and candidate galleries are frozen",
        "test images or labels",
        "per-image GT oracle routing",
        "lesion-size or subgroup routing",
        "dropping complete misses or changing cohort/metric",
    ):
        if required not in forbidden:
            raise ValueError(f"BiomedCLIP protocol missing prohibition: {required}")


def audit_protocol(
    protocol_path: Path,
    project_root: Path,
    split_manifest: Path,
    baseline_per_image: Path,
    baseline_prompt_quality: Path,
    smoke_v2_result: Path,
) -> dict[str, Any]:
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("BiomedCLIP protocol file SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    for relative, expected_hash in EXPECTED_SOURCE.items():
        if relative == "implementation_commit":
            continue
        if sha256_file(project_root / relative) != expected_hash:
            raise ValueError(f"BiomedCLIP local source hash mismatch: {relative}")
    if sha256_file(split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("BiomedCLIP frozen split artifact mismatch")
    if sha256_file(baseline_per_image) != EXPECTED_BASELINE["per_image_sha256"]:
        raise ValueError("BiomedCLIP promoted baseline per-image artifact mismatch")
    if sha256_file(baseline_prompt_quality) != EXPECTED_BASELINE["prompt_quality_sha256"]:
        raise ValueError("BiomedCLIP promoted baseline prompt artifact mismatch")
    smoke = json.loads(smoke_v2_result.read_text(encoding="utf-8"))
    weights = smoke.get("model_weight_files", [])
    if not any(weight.get("sha256") == EXPECTED_WEIGHT_SHA256 for weight in weights):
        raise ValueError("BiomedCLIP physical model-weight evidence mismatch")
    if smoke.get("validation_masks_read") is not False or smoke.get("test_evaluated") is not False:
        raise ValueError("BiomedCLIP smoke evidence accessed validation/test")
    return {
        "status": "PASS",
        "role": "pre-launch BiomedCLIP validation protocol audit",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_lock": EXPECTED_SOURCE,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "model_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "baseline": EXPECTED_BASELINE,
        "validation_predictions_generated": False,
        "validation_gt_read": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--baseline-prompt-quality", type=Path, required=True)
    parser.add_argument("--smoke-v2-result", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_protocol(
        args.protocol,
        args.project_root,
        args.split_manifest,
        args.baseline_per_image,
        args.baseline_prompt_quality,
        args.smoke_v2_result,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
