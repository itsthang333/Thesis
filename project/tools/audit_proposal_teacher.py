from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from audit_wsl_gt_pair import (
    SIZE_ORDER,
    audit_population,
    lesion_size,
    means_by_size,
    paired_bootstrap,
    read_csv,
    sha256_file,
)


EXPECTED = {
    "split_sha256": "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c",
    "classifier_sha256": "f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5",
    "sam_sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "teacher_sha256": "02d3af8feede3c3e650cb76d664185c59092697c1c8306ea67613b89f8407fb4",
    "teacher_pseudo_manifest_sha256": "7b0b133e7bbff8fecb102159b1be41801b6c51199de549a3420978b13ea7c7e6",
    "baseline_per_image_sha256": "fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540",
    "baseline_prompt_quality_sha256": "d1b570ae3a6287fdaf7fc5c28aea864d6883e5c57037542b39b17c4c6ea995e4",
    "source_commit": "ef4cd71290e9aa40f6f66983e8f0aba05d8fd4a8",
}
FROZEN_GENERATION_METADATA = {
    "split": "val",
    "target_columns": ["tumor"],
    "cam_target_class": "ground_truth",
    "image_size": 320,
    "sam_image_size": 512,
    "cam_tta_flip": True,
    "cam_percentile": 90.0,
    "cam_percentile_ensemble": True,
    "cam_percentile_values": [85.0, 90.0, 95.0],
    "morphology_fusion_mode": "components",
    "sam_prompt_mode": "box_point",
    "sam_prompt_ensemble": True,
    "max_components": 3,
    "all_cam_components": True,
    "points_per_component": 5,
    "selection_method": "coverage_mass_sam",
    "best_per_component": True,
    "fusion_topk": 1,
    "component_topk": 3,
    "support_clip_kernel": 5,
    "proposal_teacher_threshold": 0.85,
    "proposal_teacher_min_component_area": 20,
    "proposal_teacher_max_components": 3,
    "proposal_teacher_semantics": (
        "proposal_components_only; CAM scoring and support clipping unchanged"
    ),
}
CAM_SUPPORT_FIELDS = (
    "foreground_iou",
    "foreground_recall",
    "foreground_precision",
)
MECHANISM_FIELDS = (
    "point_hit_rate",
    "negative_rejection_rate",
    "box_recall",
    "box_precision",
    "oracle_best_single_dice",
    "oracle_best_single_dice_clipped",
    "selected_dice",
    "oracle_gap_dice",
    "support_loss_dice",
    "selection_loss_dice",
    "final_dice",
    "postprocess_delta_dice",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(
    rows: list[dict[str, str]],
    field: str = "image_name",
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(field, ""))
        if not key or key in result:
            raise ValueError(f"Missing or duplicate {field}: {key!r}")
        result[key] = row
    return result


def _finite_mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Missing/non-finite values for {field}")
    return statistics.fmean(values)


def mechanism_means(
    prompt_rows: list[dict[str, str]],
    metric_index: dict[str, dict[str, str]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for subgroup in ("overall", *SIZE_ORDER):
        selected = [
            row
            for row in prompt_rows
            if subgroup == "overall"
            or lesion_size(float(metric_index[row["image_name"]]["gt_area_ratio"]))
            == subgroup
        ]
        result[subgroup] = {
            "images": len(selected),
            **{
                field: _finite_mean(selected, field)
                for field in CAM_SUPPORT_FIELDS + MECHANISM_FIELDS
            },
        }
    return result


def recompute_decision(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    iterations: int,
    seed: int,
) -> tuple[dict[str, dict[str, float | int | bool]], str]:
    paired = paired_bootstrap(
        baseline_rows,
        candidate_rows,
        iterations=iterations,
        seed=seed,
    )
    promote = (
        float(paired["overall"]["paired_group_bootstrap_ci95_low"]) > 0.0
        and float(
            paired["small_lt_1pct"]["signed_gap_candidate_minus_reference"]
        )
        >= 0.0
    )
    return paired, "PROMOTE" if promote else "REJECT"


def audit_proposal_teacher(
    root: Path,
    split_manifest: Path,
    baseline_per_image: Path,
    baseline_prompt_quality: Path,
    *,
    expected_candidate_sha256: str,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    root = root.resolve()
    split_manifest = split_manifest.resolve()
    evaluation = root / "ground_truth" / "evaluation"
    pseudo = root / "ground_truth" / "pseudo_masks"
    paths = {
        "comparison": root / "comparison.json",
        "manifest": root / "run_manifest.json",
        "summary": evaluation / "summary.json",
        "candidate": evaluation / "per_image.csv",
        "prompt": pseudo / "prompt_quality.csv",
        "pseudo_manifest": pseudo / "pseudo_mask_manifest.csv",
        "pseudo_summary": pseudo / "pseudo_mask_summary.json",
        "metadata": pseudo / "run_metadata.json",
    }
    required = [
        *paths.values(),
        split_manifest,
        baseline_per_image,
        baseline_prompt_quality,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Proposal-teacher evidence is incomplete: {missing}")
    expected_candidate_sha256 = expected_candidate_sha256.strip().lower()
    if len(expected_candidate_sha256) != 64:
        raise ValueError("Expected candidate SHA-256 must contain 64 characters")

    cloud_comparison = _read_json(paths["comparison"])
    manifest = _read_json(paths["manifest"])
    metadata = _read_json(paths["metadata"])
    summary = _read_json(paths["summary"])
    pseudo_summary = _read_json(paths["pseudo_summary"])
    if manifest.get("test_evaluated") is not False:
        raise ValueError("Run manifest does not keep test locked")
    if manifest.get("source_commit") != EXPECTED["source_commit"]:
        raise ValueError("Source commit mismatch")
    if sha256_file(split_manifest) != EXPECTED["split_sha256"]:
        raise ValueError("Frozen split SHA-256 mismatch")
    if sha256_file(baseline_per_image) != EXPECTED["baseline_per_image_sha256"]:
        raise ValueError("Promoted baseline SHA-256 mismatch")
    if (
        sha256_file(baseline_prompt_quality)
        != EXPECTED["baseline_prompt_quality_sha256"]
    ):
        raise ValueError("Promoted baseline prompt-quality SHA-256 mismatch")
    if sha256_file(paths["candidate"]) != expected_candidate_sha256:
        raise ValueError("Candidate per-image SHA-256 mismatch")
    manifest_hashes = {
        "candidate": manifest.get("candidate_per_image_sha256"),
        "metadata": manifest.get("run_metadata_sha256"),
        "pseudo_manifest": manifest.get("pseudo_manifest_sha256"),
    }
    downloaded_hashes = {
        "candidate": sha256_file(paths["candidate"]),
        "metadata": sha256_file(paths["metadata"]),
        "pseudo_manifest": sha256_file(paths["pseudo_manifest"]),
    }
    if manifest_hashes != downloaded_hashes:
        raise ValueError("Downloaded artifacts differ from cloud manifest hashes")

    teacher = metadata.get("proposal_teacher")
    if not isinstance(teacher, dict):
        raise ValueError("Proposal-teacher provenance is absent")
    expected_teacher = {
        "checkpoint_sha256": EXPECTED["teacher_sha256"],
        "architecture": "resnet18_unet",
        "image_size": 448,
        "split_manifest_sha256": EXPECTED["split_sha256"],
        "train_pseudo_mask_manifest_sha256": EXPECTED[
            "teacher_pseudo_manifest_sha256"
        ],
        "supervision": "image_labels_via_pseudo_masks_only",
        "test_evaluated": False,
        "proposal_role": "add_prompt_components_only",
    }
    for field, expected in expected_teacher.items():
        if teacher.get(field) != expected:
            raise ValueError(
                f"Teacher provenance mismatch for {field}: "
                f"{teacher.get(field)!r} != {expected!r}"
            )
    for field, expected in FROZEN_GENERATION_METADATA.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Frozen generation metadata mismatch for {field}: "
                f"{metadata.get(field)!r} != {expected!r}"
            )
    if metadata.get("classifier_checkpoint_sha256") != EXPECTED["classifier_sha256"]:
        raise ValueError("Classifier SHA-256 mismatch")
    if metadata.get("sam_checkpoint_sha256") != EXPECTED["sam_sha256"]:
        raise ValueError("SAM SHA-256 mismatch")
    if metadata.get("split_manifest_sha256") != EXPECTED["split_sha256"]:
        raise ValueError("Generation split SHA-256 mismatch")

    split_rows = read_csv(split_manifest)
    baseline_rows = read_csv(baseline_per_image)
    candidate_rows = read_csv(paths["candidate"])
    population = audit_population(split_rows, candidate_rows)
    if population != audit_population(split_rows, baseline_rows):
        raise ValueError("Candidate and baseline validation populations differ")
    baseline_index = _index(baseline_rows)
    candidate_index = _index(candidate_rows)
    for name, candidate in candidate_index.items():
        baseline = baseline_index[name]
        if candidate["group_id"] != baseline["group_id"]:
            raise ValueError(f"Frozen group mismatch for {name}")
        if not math.isclose(
            float(candidate["gt_area_ratio"]),
            float(baseline["gt_area_ratio"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Frozen GT-area mismatch for {name}")
    candidate_means = means_by_size(candidate_rows)
    if not math.isclose(
        candidate_means["overall"],
        float(summary["mean_tumor_dice"]),
        rel_tol=0.0,
        abs_tol=5e-12,
    ):
        raise ValueError("Summary mean tumor Dice disagrees with per-image rows")
    if int(pseudo_summary.get("expected_images", -1)) != 371:
        raise ValueError("Pseudo-mask summary does not cover 371 validation images")
    pseudo_rows = read_csv(paths["pseudo_manifest"])
    if len(pseudo_rows) != 371:
        raise ValueError("Pseudo-mask manifest does not cover 371 validation images")

    tumor_names = {
        row["image_name"]
        for row in candidate_rows
        if str(row.get("gt_positive", "")).casefold() in {"1", "true"}
    }
    candidate_prompt_rows = read_csv(paths["prompt"])
    baseline_prompt_rows = read_csv(baseline_prompt_quality)
    candidate_prompt_index = _index(candidate_prompt_rows)
    baseline_prompt_index = _index(baseline_prompt_rows)
    if set(candidate_prompt_index) != tumor_names or set(baseline_prompt_index) != tumor_names:
        raise ValueError("Prompt diagnostics do not cover exactly all 184 tumors")
    max_cam_support_delta = 0.0
    for name in tumor_names:
        for field in CAM_SUPPORT_FIELDS:
            max_cam_support_delta = max(
                max_cam_support_delta,
                abs(
                    float(candidate_prompt_index[name][field])
                    - float(baseline_prompt_index[name][field])
                ),
            )
    if max_cam_support_delta > 1e-12:
        raise ValueError("CAM/support diagnostics changed from the promoted baseline")

    paired, decision = recompute_decision(
        baseline_rows,
        candidate_rows,
        iterations=iterations,
        seed=seed,
    )
    if decision != cloud_comparison.get("decision"):
        raise ValueError("Independent promotion decision disagrees with cloud output")
    cloud_candidate = float(cloud_comparison["candidate_mean_tumor_dice"])
    if not math.isclose(
        cloud_candidate,
        candidate_means["overall"],
        rel_tol=0.0,
        abs_tol=5e-12,
    ):
        raise ValueError("Cloud candidate mean disagrees with independent mean")

    candidate_mechanisms = mechanism_means(candidate_prompt_rows, candidate_index)
    baseline_mechanisms = mechanism_means(baseline_prompt_rows, candidate_index)
    pseudo_tumor_rows = [
        row for row in pseudo_rows if int(float(row.get("true_tumor", "0") or 0)) == 1
    ]
    teacher_components = [
        int(float(row.get("proposal_teacher_components", "0") or 0))
        for row in pseudo_tumor_rows
    ]
    teacher_support = [
        float(row.get("proposal_teacher_support_area_ratio", "0") or 0)
        for row in pseudo_tumor_rows
    ]
    return {
        "status": "PASS",
        "decision": decision,
        "test_evaluated": False,
        "population": population,
        "provenance": {
            "source_commit": EXPECTED["source_commit"],
            "split_sha256": EXPECTED["split_sha256"],
            "classifier_sha256": EXPECTED["classifier_sha256"],
            "sam_sha256": EXPECTED["sam_sha256"],
            "teacher_sha256": EXPECTED["teacher_sha256"],
            "teacher_pseudo_manifest_sha256": EXPECTED[
                "teacher_pseudo_manifest_sha256"
            ],
            "downloaded_artifact_sha256": downloaded_hashes,
        },
        "candidate_mean_tumor_dice": candidate_means,
        "paired_vs_promoted_baseline": paired,
        "proposal_usage": {
            "tumor_images": len(pseudo_tumor_rows),
            "tumor_images_with_teacher_components": sum(
                count > 0 for count in teacher_components
            ),
            "mean_teacher_components": statistics.fmean(teacher_components),
            "mean_teacher_support_area_ratio": statistics.fmean(teacher_support),
        },
        "protocol_invariance": {
            "max_cam_support_metric_abs_delta_vs_baseline": max_cam_support_delta,
            "selector": metadata["selection_method"],
            "support_clip_kernel": metadata["support_clip_kernel"],
            "teacher_semantics": metadata["proposal_teacher_semantics"],
        },
        "candidate_mechanism": candidate_mechanisms,
        "baseline_mechanism": baseline_mechanisms,
        "mechanism_delta": {
            subgroup: {
                field: float(candidate_mechanisms[subgroup][field])
                - float(baseline_mechanisms[subgroup][field])
                for field in MECHANISM_FIELDS
            }
            for subgroup in ("overall", *SIZE_ORDER)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--baseline-prompt-quality", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    result = audit_proposal_teacher(
        args.root,
        args.split_manifest,
        args.baseline_per_image,
        args.baseline_prompt_quality,
        expected_candidate_sha256=args.expected_candidate_sha256,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
