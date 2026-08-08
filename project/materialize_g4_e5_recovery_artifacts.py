from __future__ import annotations

"""Materialize the exact G4 E5 thesis artifacts from an audited recovery.

The v4 Kaggle job completed every expensive prediction-first stage and failed
only when the original freezer indexed a sparse G1 array as though it were the
full candidate bank.  This utility does not recompute candidates or metrics.
It verifies the frozen recovery, the two original resource manifests, and the
Kaggle log, then emits the two repository artifacts consumed by the final G4
completion audit.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


STUDY = "G4 E5 exact gallery richness, deduplication, and cap necessity"
SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
EXPECTED_ARMS = {
    "E5_exact__upstream_top1",
    "E5_exact__single_prompt_single_mask",
    "E5_exact__single_prompt_multimask",
    "E5_exact__full_pre_dedup",
    "E5_exact__full_post_dedup",
    "E5_exact__cap243",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def tree_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--choice-freeze", type=Path, required=True)
    parser.add_argument("--anchor-resource", type=Path, required=True)
    parser.add_argument("--addition-resource", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--addition-manifest", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--addition-root", type=Path, required=True)
    parser.add_argument("--pre-dedup-root", type=Path, required=True)
    parser.add_argument("--unified-stage-root", type=Path, required=True)
    parser.add_argument("--kaggle-log", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_resource(
    path: Path, manifest_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    resource = read_json(path)
    manifest = read_json(manifest_path)
    expected = manifest.get("splits", {}).get("val", {}).get("resource_metrics_sha256")
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"Resource manifest hash mismatch: {path}")
    if (
        int(resource.get("images_processed", -1)) != 371
        or resource.get("sam_model_type") != "vit_b"
        or resource.get("spatial_ground_truth_read") is not False
        or resource.get("test_evaluated") is not False
    ):
        raise ValueError(f"Resource contract mismatch: {path}")
    return resource, manifest


def _peak(resource_rows: list[dict[str, object]], key: str) -> int:
    values = [
        int(device[key])
        for row in resource_rows
        for device in row.get("cuda", {}).values()
    ]
    if not values:
        raise ValueError("No CUDA resource telemetry found")
    return max(values)


def _kaggle_elapsed(log_path: Path) -> float:
    rows = json.loads(log_path.read_text(encoding="utf-8"))
    values = [float(row["time"]) for row in rows if "time" in row]
    if not values:
        raise ValueError("Kaggle log has no timestamped rows")
    return max(values)


def main() -> None:
    args = parse_args()
    evaluation = read_json(args.evaluation_summary)
    audit = read_json(args.independent_audit)
    freeze = read_json(args.choice_freeze)
    if (
        evaluation.get("study") != STUDY
        or evaluation.get("split_sha256") != SPLIT_SHA
        or int(evaluation.get("images", -1)) != 371
        or int(evaluation.get("tumor_images", -1)) != 184
        or int(evaluation.get("spatial_annotations_opened", -1)) != 184
        or set(evaluation.get("summaries", {})) != EXPECTED_ARMS
        or evaluation.get("test_evaluated") is not False
        or int(evaluation.get("test_images_read", -1)) != 0
    ):
        raise ValueError("Recovered E5 evaluation contract mismatch")
    if (
        audit.get("pass") is not True
        or audit.get("stage") != "g4_e5_exact_output_audit_v1"
        or audit.get("split_sha256") != SPLIT_SHA
        or int(audit.get("images", -1)) != 371
        or int(audit.get("tumor_images", -1)) != 184
        or int(audit.get("arms", -1)) != 6
        or audit.get("test_evaluated") is not False
        or int(audit.get("test_images_read", -1)) != 0
    ):
        raise ValueError("Recovered E5 independent audit mismatch")
    if (
        freeze.get("stage") != "g4_e5_exact_choice_freeze_v1"
        or freeze.get("split_sha256") != SPLIT_SHA
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or int(freeze.get("selection_rows", -1)) != 2226
        or set(freeze.get("arms", [])) != EXPECTED_ARMS
    ):
        raise ValueError("Recovered E5 choice freeze mismatch")

    anchor, anchor_manifest = _validate_resource(
        args.anchor_resource, args.anchor_manifest
    )
    addition, addition_manifest = _validate_resource(
        args.addition_resource, args.addition_manifest
    )
    resource_rows = [anchor, addition]
    single_output_bytes = int(
        sum(int(row["output_bytes_before_resource_manifest"]) for row in resource_rows)
        + args.anchor_resource.stat().st_size
        + args.addition_resource.stat().st_size
        + (args.anchor_root / "kernel.log").stat().st_size
        + (args.anchor_root / "candidate_supply_manifest.json").stat().st_size
        + (args.addition_root / "kernel.log").stat().st_size
        + (args.addition_root / "candidate_supply_manifest.json").stat().st_size
    )
    result = {
        "schema_version": 1,
        "study": STUDY,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": SPLIT_SHA,
        "choice_freeze_sha256": sha256(args.choice_freeze),
        "evaluation_summary_sha256": sha256(args.evaluation_summary),
        "independent_audit_sha256": sha256(args.independent_audit),
        "arms": evaluation["summaries"],
        "resources": {
            "single_mask_generation_elapsed_seconds": float(
                sum(float(row["elapsed_seconds"]) for row in resource_rows)
            ),
            "single_mask_peak_allocated_bytes": _peak(
                resource_rows, "peak_memory_allocated_bytes"
            ),
            "single_mask_peak_reserved_bytes": _peak(
                resource_rows, "peak_memory_reserved_bytes"
            ),
            "single_mask_output_bytes": single_output_bytes,
            "pre_dedup_output_bytes": tree_bytes(args.pre_dedup_root),
            "unified_stage_output_bytes": tree_bytes(args.unified_stage_root),
            "total_elapsed_seconds": _kaggle_elapsed(args.kaggle_log),
            "total_elapsed_scope": (
                "observed Kaggle v4 wall time through the technical freezer failure; "
                "deterministic local recovery duration was not persisted as telemetry"
            ),
            "anchor_resource_sha256": sha256(args.anchor_resource),
            "addition_resource_sha256": sha256(args.addition_resource),
            "anchor_candidate_supply_manifest_sha256": sha256(args.anchor_manifest),
            "addition_candidate_supply_manifest_sha256": sha256(args.addition_manifest),
            "anchor_mode": anchor_manifest.get("mode"),
            "addition_mode": addition_manifest.get("mode"),
        },
        "recovery": {
            "reason": (
                "Kaggle v4 completed candidate generation and G1 scoring, then failed "
                "because the upstream-only arm indexed sparse G1 evidence as a full bank"
            ),
            "candidate_regeneration_performed": False,
            "gpu_rerun_performed": False,
            "post_dedup_source": freeze.get("post_dedup_source"),
            "evaluation_resumed_from_per_image_rows": True,
            "evaluation_annotations_opened_in_resume_process": int(
                evaluation.get("annotations_opened_in_resume_process", -1)
            ),
        },
        "choices_frozen_before_spatial_gt": True,
        "spatial_annotations_opened": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "e5_exact_results.json"
    audit_path = args.output_dir / "e5_exact_output_audit.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(args.independent_audit, audit_path)
    print(
        json.dumps(
            {
                "complete": True,
                "results_sha256": sha256(result_path),
                "audit_sha256": sha256(audit_path),
                "arm_dice": {
                    arm: values["mean_tumor_dice"]
                    for arm, values in result["arms"].items()
                },
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
