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
    paired_bootstrap,
    read_csv,
    sha256_file,
)


EXPECTED = {
    "split_sha256": "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c",
    "classifier_sha256": "7da19e9c0537501c4c919200ed65b2bf6992383af70aa91c18b312a5d6204043",
    "sam_sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "baseline_per_image_sha256": "fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540",
    "cpm_per_image_sha256": "9124c162ff1454cbcb5643358f763ab2c255f67620dae455c0ee211c97b91a52",
    "source_commit": "ab5f7cca1036b60a8b225288f14e20a70097234a",
    "candidate_per_image_sha256": "9a0e90d48177ba7d9ed3b4579fd74807519a9be0231ac1c493a8181f9b284955",
}
DIRECT_CAM_FIELDS = (
    "foreground_iou",
    "foreground_recall",
    "foreground_precision",
    "point_hit_rate",
    "negative_rejection_rate",
    "box_recall",
    "box_precision",
)
MECHANISM_FIELDS = (
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


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Missing or non-finite values for {field}")
    return statistics.fmean(values)


def _index(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[field]
        if not key or key in indexed:
            raise ValueError(f"Missing or duplicate {field}: {key!r}")
        indexed[key] = row
    return indexed


def _subgroup_mechanisms(
    prompt_rows: list[dict[str, str]],
    metric_rows_by_name: dict[str, dict[str, str]],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for subgroup in SIZE_ORDER:
        selected = [
            row
            for row in prompt_rows
            if lesion_size(
                float(metric_rows_by_name[row["image_name"]]["gt_area_ratio"])
            )
            == subgroup
        ]
        values: dict[str, float | int] = {"images": len(selected)}
        for field in DIRECT_CAM_FIELDS + MECHANISM_FIELDS:
            values[field] = _mean(selected, field)
        output[subgroup] = values
    return output


def audit_grid_gallery(
    root: Path,
    split_manifest: Path,
    baseline_per_image: Path,
    cpm_per_image: Path,
    cpm_prompt_quality: Path,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    root = root.resolve()
    split_manifest = split_manifest.resolve()
    comparison_path = root / "grid_gallery_comparison.json"
    run_manifest_path = root / "run_manifest.json"
    evaluation = root / "ground_truth" / "evaluation"
    pseudo = root / "ground_truth" / "pseudo_masks"
    candidate_per_image = evaluation / "per_image.csv"
    candidate_prompt_quality = pseudo / "prompt_quality.csv"
    pseudo_manifest = pseudo / "pseudo_mask_manifest.csv"
    run_metadata_path = pseudo / "run_metadata.json"
    required = [
        comparison_path,
        run_manifest_path,
        evaluation / "summary.json",
        candidate_per_image,
        candidate_prompt_quality,
        pseudo_manifest,
        run_metadata_path,
        baseline_per_image,
        cpm_per_image,
        cpm_prompt_quality,
        split_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Grid-gallery evidence is incomplete: {missing}")

    comparison = _read_json(comparison_path)
    manifest = _read_json(run_manifest_path)
    metadata = _read_json(run_metadata_path)
    if comparison.get("status") != "complete":
        raise ValueError("Grid-gallery comparison is not complete")
    if any(
        payload.get("test_evaluated") is not False
        for payload in (comparison, manifest)
    ):
        raise ValueError("Grid-gallery evidence does not keep test locked")
    if manifest["comparison_sha256"] != sha256_file(comparison_path):
        raise ValueError("Comparison SHA-256 mismatch")
    if manifest["split_manifest_sha256"] != EXPECTED["split_sha256"]:
        raise ValueError("Run split SHA-256 mismatch")
    if sha256_file(split_manifest) != EXPECTED["split_sha256"]:
        raise ValueError("Local frozen split SHA-256 mismatch")
    if manifest["classifier_checkpoint_sha256"] != EXPECTED["classifier_sha256"]:
        raise ValueError("Classifier checkpoint SHA-256 mismatch")
    if manifest["sam_checkpoint_sha256"] != EXPECTED["sam_sha256"]:
        raise ValueError("SAM checkpoint SHA-256 mismatch")
    if manifest["source_commit"] != EXPECTED["source_commit"]:
        raise ValueError("Source commit mismatch")
    if sha256_file(baseline_per_image) != EXPECTED["baseline_per_image_sha256"]:
        raise ValueError("Promoted baseline per-image SHA-256 mismatch")
    if sha256_file(cpm_per_image) != EXPECTED["cpm_per_image_sha256"]:
        raise ValueError("CPM control per-image SHA-256 mismatch")
    if sha256_file(candidate_per_image) != EXPECTED["candidate_per_image_sha256"]:
        raise ValueError("Grid candidate per-image SHA-256 mismatch")
    protocol_result = comparison["protocol_results"]["ground_truth"]
    artifact_hashes = {
        "run_metadata": sha256_file(run_metadata_path),
        "pseudo_mask_manifest": sha256_file(pseudo_manifest),
        "evaluation_per_image": sha256_file(candidate_per_image),
    }
    if artifact_hashes != {
        "run_metadata": protocol_result["run_metadata_sha256"],
        "pseudo_mask_manifest": protocol_result["pseudo_mask_manifest_sha256"],
        "evaluation_per_image": protocol_result["evaluation_per_image_sha256"],
    }:
        raise ValueError("Downloaded grid-gallery artifact hash mismatch")

    required_metadata = {
        "split": "val",
        "cam_target_class": "ground_truth",
        "sam_grid_gallery": True,
        "sam_grid_points_per_side": 32,
        "sam_grid_points_per_batch": 64,
        "sam_grid_pred_iou_thresh": 0.88,
        "sam_grid_stability_thresh": 0.95,
        "sam_grid_box_nms_thresh": 0.7,
        "selection_method": "coverage_mass_sam",
        "best_per_component": False,
        "fusion_topk": 1,
        "support_clip_kernel": 5,
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Grid-gallery metadata mismatch for {field}: "
                f"{metadata.get(field)!r} != {expected!r}"
            )
    if metadata.get("split_manifest_sha256") != EXPECTED["split_sha256"]:
        raise ValueError("Generation split SHA-256 mismatch")

    split_rows = read_csv(split_manifest)
    candidate_rows = read_csv(candidate_per_image)
    baseline_rows = read_csv(baseline_per_image)
    cpm_rows = read_csv(cpm_per_image)
    population = audit_population(split_rows, candidate_rows)
    if population != audit_population(split_rows, baseline_rows):
        raise ValueError("Candidate and promoted-baseline cohorts differ")
    if population != audit_population(split_rows, cpm_rows):
        raise ValueError("Candidate and CPM-control cohorts differ")
    candidate_index = _index(candidate_rows, "image_name")
    for control_name, control_rows in (
        ("promoted baseline", baseline_rows),
        ("CPM control", cpm_rows),
    ):
        for row in control_rows:
            candidate = candidate_index[row["image_name"]]
            if candidate["group_id"] != row["group_id"]:
                raise ValueError(
                    f"{control_name} group mismatch for {row['image_name']}"
                )
            if not math.isclose(
                float(candidate["gt_area_ratio"]),
                float(row["gt_area_ratio"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"{control_name} GT-area mismatch for {row['image_name']}"
                )

    candidate_prompt_rows = read_csv(candidate_prompt_quality)
    cpm_prompt_rows = read_csv(cpm_prompt_quality)
    candidate_prompt_index = _index(candidate_prompt_rows, "image_name")
    cpm_prompt_index = _index(cpm_prompt_rows, "image_name")
    tumor_names = {
        row["image_name"]
        for row in candidate_rows
        if str(row["gt_positive"]).casefold() in {"1", "true"}
    }
    if set(candidate_prompt_index) != tumor_names or set(cpm_prompt_index) != tumor_names:
        raise ValueError("Prompt-quality rows do not cover exactly 184 tumors")
    max_direct_cam_abs_delta = 0.0
    for name in sorted(tumor_names):
        for field in DIRECT_CAM_FIELDS:
            delta = abs(
                float(candidate_prompt_index[name][field])
                - float(cpm_prompt_index[name][field])
            )
            max_direct_cam_abs_delta = max(max_direct_cam_abs_delta, delta)
    if max_direct_cam_abs_delta > 1e-12:
        raise ValueError("Direct CAM/prompt diagnostics changed from CPM control")

    pseudo_rows = read_csv(pseudo_manifest)
    if len(pseudo_rows) != 371:
        raise ValueError(f"Pseudo-mask manifest has {len(pseudo_rows)} rows")
    candidate_mechanisms = _subgroup_mechanisms(
        candidate_prompt_rows, candidate_index
    )
    cpm_mechanisms = _subgroup_mechanisms(cpm_prompt_rows, candidate_index)
    mechanism_delta_vs_cpm = {
        subgroup: {
            field: float(candidate_mechanisms[subgroup][field])
            - float(cpm_mechanisms[subgroup][field])
            for field in MECHANISM_FIELDS
        }
        for subgroup in SIZE_ORDER
    }
    overall_candidate_mechanism = {
        field: _mean(candidate_prompt_rows, field)
        for field in DIRECT_CAM_FIELDS + MECHANISM_FIELDS
    }
    overall_cpm_mechanism = {
        field: _mean(cpm_prompt_rows, field)
        for field in DIRECT_CAM_FIELDS + MECHANISM_FIELDS
    }
    paired_vs_baseline = paired_bootstrap(
        baseline_rows,
        candidate_rows,
        iterations=iterations,
        seed=seed,
    )
    paired_vs_cpm = paired_bootstrap(
        cpm_rows,
        candidate_rows,
        iterations=iterations,
        seed=seed,
    )
    independently_promoted = (
        paired_vs_baseline["overall"]["signed_gap_candidate_minus_reference"] > 0
        and paired_vs_baseline["overall"]["paired_group_bootstrap_ci95_low"] > 0
        and paired_vs_baseline["small_lt_1pct"][
            "signed_gap_candidate_minus_reference"
        ]
        >= 0
    )
    if independently_promoted != bool(comparison["promoted"]):
        raise ValueError("Independent promotion decision disagrees with cloud output")

    return {
        "status": "PASS",
        "decision": "PROMOTE" if independently_promoted else "REJECT",
        "test_evaluated": False,
        "population": population,
        "provenance": {
            "split_sha256": EXPECTED["split_sha256"],
            "classifier_sha256": EXPECTED["classifier_sha256"],
            "sam_sha256": EXPECTED["sam_sha256"],
            "source_commit": EXPECTED["source_commit"],
            "downloaded_artifact_sha256": artifact_hashes,
        },
        "protocol": {
            "proposal_gallery": "official SAM automatic-mask 32x32 point grid",
            "selector": "frozen coverage_mass_sam with global top-1",
            "structural_limitation": (
                "dense masks have no CAM-component identity, so the predeclared "
                "global top-1 aggregation differs structurally from component prompts"
            ),
            "max_direct_cam_prompt_metric_abs_delta_vs_cpm": (
                max_direct_cam_abs_delta
            ),
        },
        "overall_candidate_mechanism": overall_candidate_mechanism,
        "overall_cpm_control_mechanism": overall_cpm_mechanism,
        "candidate_mechanism_by_size": candidate_mechanisms,
        "cpm_control_mechanism_by_size": cpm_mechanisms,
        "mechanism_delta_vs_cpm_by_size": mechanism_delta_vs_cpm,
        "paired_vs_promoted_baseline": paired_vs_baseline,
        "paired_vs_same_cpm_component_prompt_control": paired_vs_cpm,
        "promotion_rule_recomputed": independently_promoted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--cpm-per-image", type=Path, required=True)
    parser.add_argument("--cpm-prompt-quality", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    result = audit_grid_gallery(
        args.root,
        args.split_manifest,
        args.baseline_per_image,
        args.cpm_per_image,
        args.cpm_prompt_quality,
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
