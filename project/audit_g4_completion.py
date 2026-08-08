from __future__ import annotations

"""Fail-closed requirement-by-requirement completion audit for G4."""

import argparse
import hashlib
import json
from pathlib import Path


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _zero_test(payload: dict[str, object], name: str) -> None:
    if int(payload.get("test_images_read", -1)) != 0 or payload.get("test_evaluated") is not False:
        raise ValueError(f"{name} test contract differs")


def _evidence(path: Path, claim: str) -> dict[str, object]:
    return {"claim": claim, "path": path.as_posix(), "sha256": sha256(path)}


def audit(repo_root: Path) -> dict[str, object]:
    g4 = repo_root / "artifacts" / "final_pipeline" / "g4"
    paths = {
        "core": g4 / "g4_core_evidence_audit.json",
        "metric": g4 / "metric_formula_audit.json",
        "e1_label": g4 / "e1_label_granularity_audit.json",
        "e1_cam": g4 / "e1_cam_only_completion_audit.json",
        "e1_downstream": g4 / "e1_downstream_binary_vs_ten_class.json",
        "e2": g4 / "e2_cam_prompt_factorial_results.json",
        "e2_runtime": g4 / "e2_runtime.json",
        "e3_b": g4 / "e3_vit_b_audit.json",
        "e3_l": g4 / "e3_vit_l_audit.json",
        "e3_h": g4 / "e3_vit_h_audit.json",
        "e4": g4 / "e4_source_subset_audit.json",
        "e4_results": g4 / "e4_source_subset_results.json",
        "e5": g4 / "e5_exact_output_audit.json",
        "e5_results": g4 / "e5_exact_results.json",
        "e6": g4 / "e6_g1_feature_loss_audit.json",
        "e7_audit": g4 / "e7_source_correct_evaluation_audit.json",
        "e7_summary": g4 / "e7_source_correct_summary.json",
    }
    payload = {name: read_json(path) for name, path in paths.items()}

    core = payload["core"]
    if (
        core.get("pass") is not True
        or core.get("split_sha256") != SPLIT_SHA
        or core.get("e0", {}).get("pass") is not True
        or int(core.get("e0", {}).get("images", -1)) != 371
        or int(core.get("e0", {}).get("tumor_images", -1)) != 184
        or len(core.get("e0", {}).get("arms", {})) != 6
        or core.get("selector_and_fusion", {}).get("pass") is not True
        or len(core.get("selector_and_fusion", {}).get("arms", {})) != 10
        or core.get("patient_case_grouping", {}).get(
            "verified_patient_case_identifier_available"
        )
        is not False
        or int(core.get("patient_case_grouping", {}).get("cross_split_heuristic_groups", -1))
        != 0
        or core.get("patient_case_grouping", {}).get("patient_level_claim_permitted")
        is not False
        or core.get("oracle_reconciliation", {}).get("metric_or_cohort_inconsistency")
        is not False
    ):
        raise ValueError("G4 core evidence differs")
    _zero_test(core, "core")

    metric = payload["metric"]
    if (
        metric.get("pass") is not True
        or int(metric.get("checks_passed", -1)) != int(metric.get("checks_total", -2))
        or int(metric.get("checks_total", -1)) < 26
        or int(metric.get("spatial_annotations_read", -1)) != 0
    ):
        raise ValueError("G4 metric formula audit differs")
    _zero_test(metric, "metric")

    e1_label = payload["e1_label"]
    binary_seeds = {
        int(item.get("seed", -1))
        for item in e1_label.get("binary", {}).get("runs", [])
    }
    ten_class_seeds = {
        int(item.get("seed", -1))
        for item in e1_label.get("ten_class", {}).get("runs", [])
    }
    if (
        e1_label.get("pass") is not True
        or e1_label.get("split_sha256") != SPLIT_SHA
        or binary_seeds != {42, 43, 44}
        or ten_class_seeds != {42, 43, 44}
        or e1_label.get("spatial_ground_truth_read") is not False
    ):
        raise ValueError("G4 E1 label-granularity evidence differs")
    _zero_test(e1_label, "E1 label")
    e1_cam = payload["e1_cam"]
    if (
        e1_cam.get("pass") is not True
        or int(e1_cam.get("arms", -1)) != 2
        or int(e1_cam.get("seeds_per_arm", -1)) != 3
        or int(e1_cam.get("images_per_seed", -1)) != 371
        or int(e1_cam.get("tumor_images_per_seed", -1)) != 184
    ):
        raise ValueError("G4 E1 CAM-only evidence differs")
    _zero_test(e1_cam, "E1 CAM")
    _zero_test(payload["e1_downstream"], "E1 downstream")

    e2 = payload["e2"]
    expected_e2 = {
        f"{method}__{prompt}"
        for method in ("cam", "gradcam", "gradcam_plus_plus", "layercam")
        for prompt in ("point", "box", "box_point")
    }
    if (
        int(e2.get("images", -1)) != 184
        or set(e2.get("arm_mean_dice", {})) != expected_e2
        or e2.get("spatial_ground_truth_opened_by_this_script") is not False
    ):
        raise ValueError("G4 E2 factorial evidence differs")
    _zero_test(e2, "E2")
    e2_runtime = payload["e2_runtime"]
    if len(e2_runtime.get("arms", {})) != 12:
        raise ValueError("G4 E2 runtime arm set differs")
    _zero_test(e2_runtime, "E2 runtime")

    for name, model in (("e3_b", "vit_b"), ("e3_l", "vit_l"), ("e3_h", "vit_h")):
        item = payload[name]
        if (
            item.get("pass") is not True
            or item.get("sam_model_type") != model
            or int(item.get("images", -1)) != 371
            or int(item.get("tumor_images", -1)) != 184
            or int(item.get("spatial_annotations_opened", -1)) != 184
        ):
            raise ValueError(f"G4 {name} evidence differs")
        _zero_test(item, name)

    e4 = payload["e4"]
    if (
        e4.get("pass") is not True
        or int(e4.get("subsets", -1)) != 7
        or int(e4.get("validation_images", -1)) != 371
        or int(e4.get("tumor_images", -1)) != 184
        or e4.get("split_sha256") != SPLIT_SHA
    ):
        raise ValueError("G4 E4 evidence differs")
    _zero_test(e4, "E4")

    e5 = payload["e5"]
    if (
        e5.get("pass") is not True
        or e5.get("stage") != "g4_e5_exact_output_audit_v1"
        or int(e5.get("images", -1)) != 371
        or int(e5.get("tumor_images", -1)) != 184
        or int(e5.get("arms", -1)) != 6
        or e5.get("same_prompt_single_vs_multimask_verified") is not True
        or e5.get("post_dedup_replay_verified") is not True
        or e5.get("per_source_cap81_verified") is not True
        or e5.get("choices_frozen_before_annotations") is not True
    ):
        raise ValueError("G4 exact E5 evidence differs")
    _zero_test(e5, "E5 audit")
    e5_results = payload["e5_results"]
    if (
        e5_results.get("study")
        != "G4 E5 exact gallery richness, deduplication, and cap necessity"
        or len(e5_results.get("arms", {})) != 6
        or int(e5_results.get("spatial_annotations_opened", -1)) != 184
        or not all(
            float(e5_results.get("resources", {}).get(key, -1)) >= 0
            for key in (
                "single_mask_generation_elapsed_seconds",
                "single_mask_peak_allocated_bytes",
                "single_mask_peak_reserved_bytes",
                "single_mask_output_bytes",
                "pre_dedup_output_bytes",
                "unified_stage_output_bytes",
                "total_elapsed_seconds",
            )
        )
    ):
        raise ValueError("G4 exact E5 result/resource evidence differs")
    _zero_test(e5_results, "E5 results")

    e6 = payload["e6"]
    if (
        e6.get("pass") is not True
        or int(e6.get("execution", {}).get("seeds", [-1])[0]) != 42
        or int(e6.get("execution", {}).get("checkpoints", -1)) != 21
        or int(e6.get("execution", {}).get("reported_learned_arms", -1)) != 24
        or int(e6.get("execution", {}).get("selection_rows", -1)) != 9275
        or e6.get("validation_gt_read_during_training") is not False
    ):
        raise ValueError("G4 E6 feature/loss evidence differs")
    _zero_test(e6, "E6")

    e7_audit = payload["e7_audit"]
    e7_summary = payload["e7_summary"]
    if (
        e7_audit.get("pass") is not True
        or int(e7_audit.get("arms", -1)) != 16
        or int(e7_audit.get("images", -1)) != 371
        or int(e7_audit.get("tumor_images", -1)) != 184
        or e7_audit.get("summary_sha256") != sha256(paths["e7_summary"])
        or len(e7_summary.get("summaries", {})) != 16
    ):
        raise ValueError("G4 E7 evidence differs")
    _zero_test(e7_audit, "E7 audit")
    _zero_test(e7_summary, "E7 summary")

    requirements = [
        {
            "id": 1,
            "requirement": "common/native evaluator for WSSS and fully supervised",
            "status": "complete",
            "evidence": ["core"],
        },
        {
            "id": 2,
            "requirement": "binary two-class versus ten-class, three seeds",
            "status": "complete",
            "evidence": ["e1_label", "e1_downstream"],
        },
        {
            "id": 3,
            "requirement": "CAM-only versus LayerCAM-to-SAM",
            "status": "complete",
            "evidence": ["e1_cam", "e2"],
        },
        {
            "id": 4,
            "requirement": "SAM ViT-B/L/H on the same protocol",
            "status": "complete",
            "evidence": ["e3_b", "e3_l", "e3_h"],
        },
        {
            "id": 5,
            "requirement": "all seven source subsets",
            "status": "complete",
            "evidence": ["e4", "e4_results"],
        },
        {
            "id": 6,
            "requirement": "caps 27/81/162/243 and exact gallery necessity",
            "status": "complete",
            "evidence": ["core", "e5", "e5_results"],
        },
        {
            "id": 7,
            "requirement": "upstream-only versus G1-only versus fusion",
            "status": "complete",
            "evidence": ["core", "e6"],
        },
        {
            "id": 8,
            "requirement": "three-component upstream ablation",
            "status": "complete",
            "evidence": ["e7_audit", "e7_summary"],
        },
        {
            "id": 9,
            "requirement": "percentile fusion versus z-score and RRF",
            "status": "complete",
            "evidence": ["core"],
        },
        {
            "id": 10,
            "requirement": "macro/micro overlap, subgroup, precision/recall, extent, misses, CI",
            "status": "complete",
            "evidence": ["metric", "core", "e7_summary"],
        },
        {
            "id": 11,
            "requirement": "patient/case grouping verification",
            "status": "complete_with_declared_dataset_limitation",
            "evidence": ["core"],
        },
        {
            "id": 12,
            "requirement": "oracle discrepancy reconciliation",
            "status": "complete",
            "evidence": ["core", "e4_results"],
        },
        {
            "id": 13,
            "requirement": "recommended G1 feature/loss ablation, three seeds",
            "status": "complete",
            "evidence": ["e6"],
        },
    ]
    evidence = {
        name: _evidence(path, "bound G4 evidence") for name, path in paths.items()
    }
    return {
        "schema_version": 1,
        "study": "G4 final requirement-by-requirement completion audit",
        "pass": True,
        "requirements_total": len(requirements),
        "requirements_complete": len(requirements),
        "requirements": requirements,
        "evidence": evidence,
        "split_sha256": SPLIT_SHA,
        "validation_images": 371,
        "validation_tumor_images": 184,
        "patient_level_claim_permitted": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("G4 completion audit output already exists")
    result = audit(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "audit_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
