from __future__ import annotations

"""Validation-only diagnosis of what G1/upstream reweighting can and cannot fix.

This is deliberately an evaluator, not a selector.  It opens validation masks
only after the immutable G1 and upstream candidate scores have been loaded.  A
post-hoc alpha sweep is reported as an explanatory upper bound and is never a
promotable validation result.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_g2_objective import (
    average_percentile_rank,
    rank_fusion_scores,
    stable_select,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPECTED_BASELINE_DICE = 0.28872948670665205
SUBGROUPS = ("overall", "small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--expected-score-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def pareto_frontier(g1: np.ndarray, upstream: np.ndarray) -> np.ndarray:
    """Return candidates not strictly dominated in both higher-is-better scores."""

    left = np.asarray(g1, dtype=np.float64)
    right = np.asarray(upstream, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or not len(left):
        raise ValueError("score vectors must be non-empty aligned vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("score vectors must be finite")
    ge_left = left[:, None] >= left[None, :]
    ge_right = right[:, None] >= right[None, :]
    strict = (left[:, None] > left[None, :]) | (right[:, None] > right[None, :])
    # Row j dominates column i when both scores are no worse and one is better.
    dominated = np.any(ge_left & ge_right & strict, axis=0)
    return ~dominated


def alpha_grid_selected_quality(
    g1: np.ndarray,
    upstream: np.ndarray,
    quality: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fixed rank-fusion family; tie-breaking stays identical to G1."""

    g1 = np.asarray(g1, dtype=np.float64)
    upstream = np.asarray(upstream, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if not (g1.shape == upstream.shape == quality.shape) or g1.ndim != 1:
        raise ValueError("candidate vectors must align")
    g1_rank = average_percentile_rank(g1)
    upstream_rank = average_percentile_rank(upstream)
    selected = np.empty(len(alphas), dtype=np.int64)
    selected_quality = np.empty(len(alphas), dtype=np.float64)
    for offset, alpha in enumerate(alphas):
        score = alpha * g1_rank + (1.0 - alpha) * upstream_rank
        local = stable_select(score, g1)
        selected[offset] = local
        selected_quality[offset] = quality[local]
    return selected, selected_quality


def _subset(rows: Iterable[dict[str, Any]], subgroup: str) -> list[dict[str, Any]]:
    return [row for row in rows if subgroup == "overall" or row["size_group"] == subgroup]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = _mean(rows, "baseline_dice")
    eligible = _mean(rows, "eligible_oracle_dice")
    frontier = _mean(rows, "pareto_frontier_oracle_dice")
    linear = _mean(rows, "per_image_alpha_oracle_dice")
    return {
        "n": len(rows),
        "baseline_dice": baseline,
        "eligible_oracle_dice": eligible,
        "selector_regret": eligible - baseline,
        "pareto_frontier_oracle_dice": frontier,
        "per_image_alpha_oracle_dice": linear,
        "exact_regret_decomposition": {
            "score_dominance_identifiability_gap": eligible - frontier,
            "nonlinear_monotone_frontier_gap": frontier - linear,
            "per_image_weight_gap": linear - baseline,
            "sum": eligible - baseline,
        },
        "oracle_not_represented_on_frontier": int(
            sum(float(row["score_dominance_gap"]) > 1.0e-12 for row in rows)
        ),
        "dominance_gap_ge_0_05": int(
            sum(float(row["score_dominance_gap"]) >= 0.05 for row in rows)
        ),
        "per_image_reweight_gain_ge_0_05": int(
            sum(float(row["per_image_weight_gain"]) >= 0.05 for row in rows)
        ),
        "baseline_complete_misses": int(sum(int(row["baseline_complete_miss"]) for row in rows)),
        "frontier_recoverable_misses": int(
            sum(
                int(row["baseline_complete_miss"])
                and float(row["pareto_frontier_oracle_dice"]) > 0.0
                for row in rows
            )
        ),
        "linear_reweight_recoverable_misses": int(
            sum(
                int(row["baseline_complete_miss"])
                and float(row["per_image_alpha_oracle_dice"]) > 0.0
                for row in rows
            )
        ),
        "frontier_size": {
            "mean": _mean(rows, "pareto_frontier_count"),
            "median": float(np.median([float(row["pareto_frontier_count"]) for row in rows])),
        },
    }


def _render_dossier(summary: dict[str, Any]) -> str:
    lines = [
        "# G1 two-score identifiability dossier",
        "",
        "## Question",
        "",
        "Can any reweighting of the existing G1 and upstream percentile ranks close",
        "the selector gap, or are correct candidates jointly ranked below wrong masks?",
        "This is a post-hoc validation diagnosis only; no alpha found here is promotable.",
        "",
        "## Exact decomposition",
        "",
        "For each image, `eligible oracle - baseline` is decomposed as:",
        "",
        "1. `eligible oracle - Pareto-frontier oracle`: candidates invisible to every",
        "   monotone combination because another mask scores no worse on both inputs;",
        "2. `Pareto-frontier oracle - per-image alpha oracle`: nonlinear frontier choice",
        "   not reachable by the nonnegative linear rank-fusion family;",
        "3. `per-image alpha oracle - baseline`: theoretical gain from an image-specific",
        "   G1/upstream weight, unavailable without a new routing signal.",
        "",
        "| Group | Baseline | Eligible oracle | Pareto oracle | Per-image alpha oracle | Dominance gap | Frontier gap | Weight gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in SUBGROUPS:
        row = summary["subgroups"][group]
        dec = row["exact_regret_decomposition"]
        lines.append(
            f"| {group} | {row['baseline_dice']:.6f} | {row['eligible_oracle_dice']:.6f} | "
            f"{row['pareto_frontier_oracle_dice']:.6f} | {row['per_image_alpha_oracle_dice']:.6f} | "
            f"{dec['score_dominance_identifiability_gap']:.6f} | "
            f"{dec['nonlinear_monotone_frontier_gap']:.6f} | {dec['per_image_weight_gap']:.6f} |"
        )
    sweep = summary["global_alpha_sweep_exploratory"]
    lines.extend(
        [
            "",
            "## Shared-weight falsification check",
            "",
            f"- Frozen equal-rank alpha 0.5 Dice: `{sweep['alpha_0_5_dice']:.6f}`.",
            f"- Post-hoc best shared alpha: `{sweep['best_alpha']:.3f}` with Dice "
            f"`{sweep['best_overall_dice']:.6f}` (diagnostic only).",
            f"- Gain over the frozen baseline: `{sweep['best_overall_gain']:.6f}`.",
            "",
            "## Decision",
            "",
            summary["mechanistic_conclusion"],
            "",
            "BTXRD test remains locked. No selector or alpha discovered from validation",
            "polygons is authorized for test or for a confirmatory claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("canonical validation cohort mismatch")
    row_by_id = {str(row["image_id"]): row for row in val_rows}
    freeze_path = args.score_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_score_freeze_sha256:
        raise ValueError("score freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("score freeze contract mismatch")
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[str(row["image_id"]) for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("complete validation candidate gallery is required")

    # Annotation boundary: immutable scores and candidate identities are verified above.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    alphas = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    per_image: list[dict[str, Any]] = []
    alpha_quality: list[np.ndarray] = []
    alpha_group: list[str] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_id_value = dataset[index]
        image_id = str(image_id_value)
        split_row = row_by_id[image_id]
        if split_row["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        gt_pixels = int(target.sum())
        subgroup = size_group(float(target.mean()))
        stem = Path(image_id).stem
        score_path = args.score_root / "stage_a_scores" / f"{stem}.npz"
        candidate_row = candidate_rows[stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(score_path, allow_pickle=False) as payload:
            kept = payload["candidate_indices"].astype(np.int64)
            g1 = payload["g1_frozen_candidate_logits"].astype(np.float64)
            upstream = payload["upstream_scores"].astype(np.float64)
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"][kept].astype(bool)
        if not (len(proposals) == len(g1) == len(upstream)):
            raise ValueError(f"candidate alignment mismatch: {image_id}")
        proposal_pixels = proposals.sum(axis=(1, 2), dtype=np.int64)
        intersections = np.logical_and(proposals, target).sum(axis=(1, 2), dtype=np.int64)
        denominators = proposal_pixels + gt_pixels
        quality = np.divide(
            2.0 * intersections,
            denominators,
            out=np.zeros(len(proposals), dtype=np.float64),
            where=denominators > 0,
        )
        baseline_score = rank_fusion_scores(g1, upstream)
        baseline_local = stable_select(baseline_score, g1)
        frontier = pareto_frontier(g1, upstream)
        frontier_quality = quality[frontier]
        selected_by_alpha, quality_by_alpha = alpha_grid_selected_quality(
            g1, upstream, quality, alphas
        )
        eligible_oracle = float(quality.max())
        frontier_oracle = float(frontier_quality.max())
        linear_oracle = float(quality_by_alpha.max())
        baseline_dice = float(quality[baseline_local])
        if frontier_oracle + 1.0e-12 < linear_oracle:
            raise RuntimeError("linear rank fusion escaped the Pareto frontier")
        if linear_oracle + 1.0e-12 < baseline_dice:
            raise RuntimeError("alpha grid does not contain frozen alpha 0.5")
        per_image.append(
            {
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "size_group": subgroup,
                "candidate_count": len(quality),
                "pareto_frontier_count": int(frontier.sum()),
                "baseline_dice": baseline_dice,
                "baseline_complete_miss": int(intersections[baseline_local] == 0),
                "eligible_oracle_dice": eligible_oracle,
                "pareto_frontier_oracle_dice": frontier_oracle,
                "per_image_alpha_oracle_dice": linear_oracle,
                "per_image_best_alpha_first": float(alphas[int(quality_by_alpha.argmax())]),
                "score_dominance_gap": eligible_oracle - frontier_oracle,
                "nonlinear_frontier_gap": frontier_oracle - linear_oracle,
                "per_image_weight_gain": linear_oracle - baseline_dice,
                "baseline_selected_on_frontier": int(frontier[baseline_local]),
                "distinct_alpha_selected_candidates": int(len(np.unique(selected_by_alpha))),
            }
        )
        alpha_quality.append(quality_by_alpha)
        alpha_group.append(subgroup)

    if len(per_image) != 184 or Counter(row["size_group"] for row in per_image) != Counter(
        {"small": 94, "medium": 72, "large": 18}
    ):
        raise RuntimeError("tumor/subgroup cohort mismatch")
    if not all(int(row["baseline_selected_on_frontier"]) == 1 for row in per_image):
        raise RuntimeError("positive equal-rank baseline must select a Pareto-frontier candidate")

    alpha_matrix = np.stack(alpha_quality, axis=0)
    alpha_means = alpha_matrix.mean(axis=0)
    alpha_half = int(np.flatnonzero(np.isclose(alphas, 0.5, atol=1.0e-12))[0])
    baseline_mean = float(np.mean([row["baseline_dice"] for row in per_image]))
    if abs(baseline_mean - EXPECTED_BASELINE_DICE) > 1.0e-12:
        raise RuntimeError(
            f"frozen baseline mismatch: {baseline_mean:.12f} != {EXPECTED_BASELINE_DICE:.12f}"
        )
    if abs(float(alpha_means[alpha_half]) - EXPECTED_BASELINE_DICE) > 1.0e-12:
        raise RuntimeError("alpha=0.5 does not reproduce the frozen baseline")
    best_alpha_index = int(alpha_means.argmax())
    best_alpha = float(alphas[best_alpha_index])
    best_overall = float(alpha_means[best_alpha_index])
    subgroup_summaries = {
        group: _summarize(_subset(per_image, group)) for group in SUBGROUPS
    }
    best_alpha_subgroups = {}
    for group in ("small", "medium", "large"):
        indices = [offset for offset, value in enumerate(alpha_group) if value == group]
        best_alpha_subgroups[group] = float(alpha_matrix[indices, best_alpha_index].mean())

    total = subgroup_summaries["overall"]
    decomposition = total["exact_regret_decomposition"]
    global_gain = best_overall - EXPECTED_BASELINE_DICE
    if decomposition["score_dominance_identifiability_gap"] >= 0.03:
        conclusion = (
            "A material part of the oracle gap is score-identifiability failure: good masks are "
            "jointly dominated under both existing signals. Reweighting G1/upstream cannot fix "
            "that component; a genuinely independent lesion-evidence channel is required."
        )
    elif global_gain < 0.01 and decomposition["per_image_weight_gap"] >= 0.05:
        conclusion = (
            "Different images prefer incompatible G1/upstream weights while no shared weight "
            "materially improves the baseline. A new per-image causal routing signal would be "
            "required; weight sweeping alone is falsified."
        )
    else:
        conclusion = (
            "The two existing scores retain material recoverability. Interpret the three exact "
            "gap terms and matched prior failures before deciding whether calibration or a new "
            "representation is justified."
        )
    summary = {
        "stage": "rich_gallery_g1_two_score_identifiability_v1",
        "role": "post_hoc_validation_diagnostic_not_promotable",
        "baseline_formula": "0.5*percentile_rank(g1)+0.5*percentile_rank(upstream)",
        "cohort": {"validation": 371, "tumor": 184, "small": 94, "medium": 72, "large": 18},
        "subgroups": subgroup_summaries,
        "global_alpha_sweep_exploratory": {
            "grid_count": len(alphas),
            "best_alpha": best_alpha,
            "best_overall_dice": best_overall,
            "best_overall_gain": global_gain,
            "alpha_0_5_dice": float(alpha_means[alpha_half]),
            "best_alpha_subgroup_dice": best_alpha_subgroups,
            "validation_tuned_not_promotable": True,
        },
        "mechanistic_conclusion": conclusion,
        "candidate_scores_frozen_before_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dossier_path = args.output_dir / "G1_TWO_SCORE_IDENTIFIABILITY_DOSSIER.md"
    dossier_path.write_text(_render_dossier(summary), encoding="utf-8")
    audit = {
        "stage": summary["stage"],
        "split_sha256": sha256_file(args.split_manifest),
        "score_freeze_sha256": sha256_file(freeze_path),
        "candidate_manifest_sha256": candidate_audit["manifest_sha256"],
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "dossier_sha256": sha256_file(dossier_path),
        "candidate_scores_frozen_before_validation_gt": True,
        "validation_gt_evaluator_only": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
