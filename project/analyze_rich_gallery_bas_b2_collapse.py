from __future__ import annotations

"""Post-freeze diagnosis of BAS-B2 and the remaining G1-fusion bottleneck."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from mae_reconstruction_io import sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank


BASELINE = "g1_upstream_baseline"
PRIMARY = "g1_upstream_bas_three_way"
VARIANTS = (
    BASELINE,
    "bas_only",
    "g1_bas_two_way",
    "upstream_bas_two_way",
    PRIMARY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-per-image-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--identifiability-root", type=Path, required=True)
    parser.add_argument("--expected-identifiability-per-image-sha256", required=True)
    parser.add_argument("--expected-identifiability-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()) if len(array) else 0.0


def median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.median(array)) if len(array) else 0.0


def regime_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n": len(rows),
        "baseline_dice": mean(float(row["baseline_dice"]) for row in rows),
        "primary_dice": mean(float(row["primary_dice"]) for row in rows),
        "primary_delta": mean(float(row["primary_delta"]) for row in rows),
        "baseline_misses": sum(int(row["baseline_miss"]) for row in rows),
        "primary_misses": sum(int(row["primary_miss"]) for row in rows),
        "primary_area_gt_median": median(float(row["primary_area_gt"]) for row in rows),
        "dominance_gap": mean(float(row["score_dominance_gap"]) for row in rows),
    }


def render_dossier(summary: dict[str, object]) -> str:
    actual = summary["actual_binary_mask_metrics"]
    collapse = summary["training_collapse"]
    area = summary["candidate_area_proxy"]
    regimes = summary["regimes"]
    dominance = summary["two_score_identifiability"]
    lines = [
        "# G1 fixed-rank fusion after BAS-B2: causal bottleneck dossier",
        "",
        "## Immutable reference and actual endpoint",
        "",
        "The frozen baseline remains `0.5*rank(G1)+0.5*rank(upstream)`. BAS-B2",
        "adds one predeclared rank to form `(rank(G1)+rank(upstream)+rank(BAS))/3`.",
        "All choices were frozen before validation polygons and test remained locked.",
        "",
        "| Variant | Dice | IoU | <1% | 1-<5% | >=5% | Misses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = actual[variant]
        lines.append(
            f"| `{variant}` | {item['overall']['dice']:.6f} | {item['overall']['iou']:.6f} | "
            f"{item['small']['dice']:.6f} | {item['medium']['dice']:.6f} | "
            f"{item['large']['dice']:.6f} | {item['overall']['complete_misses']} |"
        )
    lines.extend(
        [
            "",
            "BAS-B2 is a strict negative result: the primary Dice delta is "
            f"`{actual[PRIMARY]['overall']['dice'] - actual[BASELINE]['overall']['dice']:+.6f}`.",
            "It helps large lesions but substantially damages both small and medium lesions.",
            "",
            "## Root cause 1: optimization collapsed before BAS was scientifically tested",
            "",
            f"- Epoch 1 train accuracy: `{collapse['epoch1_accuracy']:.6f}`.",
            f"- Epoch 2-100 accuracy range: `{collapse['tail_accuracy_min']:.6f}` to "
            f"`{collapse['tail_accuracy_max']:.6f}`.",
            f"- Epoch 2-100 full CE range: `{collapse['tail_full_ce_min']:.9f}` to "
            f"`{collapse['tail_full_ce_max']:.9f}` (binary chance is `log(2)`).",
            f"- The constant class-0 argmax accuracy is exactly "
            f"`1493/2981={collapse['constant_logit_argmax_accuracy']:.9f}`, matching "
            f"the observed tail accuracy to numerical precision.",
            f"- The tail BAS mean is `{collapse['tail_bas_mean']:.9f}`. A label-only "
            f"constant map (normal=0, tumor=1) predicts "
            f"`1.2*1488/2981={collapse['constant_label_map_bas']:.9f}`; residual "
            f"`{collapse['tail_bas_minus_constant_label_map']:+.3e}`.",
            f"- Validation image AUROC: `{collapse['validation_auroc']:.6f}`; activation-range "
            f"mean: `{collapse['activation_range_mean']:.3e}`; nondegenerate tumor maps: "
            f"`{collapse['tumor_nondegenerate_fraction']:.3f}`.",
            "",
            "The terminal ReLU class head reached a dead two-logit state: the classifier",
            "outputs became indistinguishable and the sigmoid localization branch saturated",
            "to a label prior rather than a spatial map. The exact CE, accuracy and BAS-loss",
            "identities show that this is not ordinary underfitting. Therefore B2 does not",
            "falsify the BAS paper's mechanism; it falsifies",
            "this unstable binary transfer and the absence of a fail-fast optimization gate.",
            "",
            "## Root cause 2: the frozen BAS candidate score is an area rank",
            "",
            f"Mean/median within-image Spearman(BAS score, candidate area) is "
            f"`{area['bas_area_correlation_mean']:.6f}/{area['bas_area_correlation_median']:.6f}`; "
            f"the fraction above 0.90 is `{area['fraction_above_0_9']:.3f}`.",
            f"The frozen tumor activation has mean/std/range "
            f"`{area['activation_mean_tumor']:.9f}/"
            f"{area['activation_std_mean_tumor']:.3e}/"
            f"{area['activation_range_mean_tumor']:.3e}`.",
            f"BAS-only selects mean/median candidate-area rank "
            f"`{area['selected_area_rank']['bas_only']['mean']:.6f}/"
            f"{area['selected_area_rank']['bas_only']['median']:.6f}`.",
            "",
            "For an almost constant normalized activation `A(x)=mu`, candidate `M` has",
            "`coverage=sum_M A/sum_Omega A=|M|/|Omega|` and",
            "`purity=sum_M A/|M|=mu`. Its harmonic score",
            "`2*coverage*purity/(coverage+purity)` is strictly increasing in mask area.",
            "Thus the third rank is not tumor identity; it is a disguised large-mask prior.",
            "This exactly explains BAS-only's high recall but extreme area inflation.",
            "",
            "## What BAS reveals about the baseline rather than merely failing",
            "",
            "| Regime | n | Base Dice | BAS-3way Dice | Delta | Base/primary misses | Primary area/GT median |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "all": "all tumors",
        "severe_dominance": "two-score dominance >=0.05",
        "baseline_miss": "baseline complete miss",
        "underextent": "baseline area/GT <0.5",
        "calibrated_extent": "baseline area/GT 0.5-2",
        "moderate_overextent": "baseline area/GT 2-10",
        "extreme_overextent": "baseline area/GT >=10",
    }
    for key, label in labels.items():
        item = regimes[key]
        lines.append(
            f"| {label} | {item['n']} | {item['baseline_dice']:.6f} | "
            f"{item['primary_dice']:.6f} | {item['primary_delta']:+.6f} | "
            f"{item['baseline_misses']}/{item['primary_misses']} | "
            f"{item['primary_area_gt_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The broad area signal recovers overlap in many baseline misses, but its false",
            "positive mass destroys Dice elsewhere. Fewer complete misses are therefore not",
            "evidence of a better selector. This reproduces the earlier relational/consensus",
            "failure: expansion helps under-extent and large lesions while harming tiny lesions.",
            "",
            "## Exact remaining bottleneck",
            "",
            f"Eligible oracle Dice is `{dominance['eligible_oracle_dice']:.6f}` versus baseline "
            f"`{dominance['baseline_dice']:.6f}`. Of the `{dominance['total_regret']:.6f}` gap,",
            f"`{dominance['dominance_gap']:.6f}` ({dominance['dominance_share']:.2%}) is invisible",
            "to every monotone G1/upstream fusion. Candidate truncation is negligible.",
            "The precise missing observable is candidate-level tumor identity with calibrated",
            "extent: it must distinguish tiny-lesion anatomy overreach from medium/large",
            "discriminative fragments. A source router, area prior, consensus score, SAM",
            "confidence, anomaly score, another global CAM, or another fusion-weight sweep",
            "cannot supply that observable.",
            "",
            "## Improvement boundary",
            "",
            "1. Keep the immutable rich gallery and G1/upstream baseline.",
            "2. Treat the current BAS output as a technical-collapse diagnostic, never as a",
            "   third selector score.",
            "3. Any BAS correction must first pass image-label-only fail-fast gates: non-chance",
            "   balanced classification, nondegenerate activation, and BAS-area correlation",
            "   materially below 1.0 before a full run.",
            "4. The only scientifically justified learned addition is a candidate-conditioned",
            "   positive-evidence residual with explicit inside-versus-ring/background contrast",
            "   and train-normal hard negatives. It must preserve candidate scale rather than",
            "   rewarding raw covered area.",
            "5. Freeze every candidate choice before polygons; actual binary Dice and all",
            "   94/72/18 subgroups remain the promotion endpoint.",
            "",
            "No validation-tuned BAS weight, area threshold, source route, morphology rescue,",
            "or test access is authorized from this failure.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    freeze_path = args.stage_a_root / "prediction_freeze.json"
    summary_path = args.stage_b_root / "summary.json"
    per_image_path = args.stage_b_root / "per_image.csv"
    ident_path = args.identifiability_root / "per_image.csv"
    ident_summary_path = args.identifiability_root / "summary.json"
    candidate_manifest = args.candidate_root / "candidate_diagnostics_manifest.csv"
    checks = (
        (freeze_path, args.expected_freeze_sha256),
        (summary_path, args.expected_summary_sha256),
        (per_image_path, args.expected_per_image_sha256),
        (ident_path, args.expected_identifiability_per_image_sha256),
        (ident_summary_path, args.expected_identifiability_summary_sha256),
        (candidate_manifest, args.expected_candidate_manifest_sha256),
    )
    for path, expected in checks:
        if sha256_file(path) != expected:
            raise ValueError(f"input SHA-256 mismatch: {path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    stage_b = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("test_evaluated") is not False
        or stage_b.get("test_evaluated") is not False
    ):
        raise ValueError("BAS provenance/safety contract mismatch")
    per_image = read_rows(per_image_path)
    baseline = {row["image_id"]: row for row in per_image if row["variant"] == BASELINE}
    variants = {
        variant: {row["image_id"]: row for row in per_image if row["variant"] == variant}
        for variant in VARIANTS
    }
    ident = {row["image_id"]: row for row in read_rows(ident_path)}
    ident_summary = json.loads(ident_summary_path.read_text(encoding="utf-8"))
    if (
        ident_summary.get("candidate_scores_frozen_before_validation_gt") is not True
        or ident_summary.get("test_evaluated") is not False
    ):
        raise ValueError("identifiability provenance/safety contract mismatch")
    if not (len(baseline) == len(ident) == 184):
        raise ValueError("tumor analysis cohort mismatch")
    selections = {
        (row["variant"], row["image_id"]): row
        for row in read_rows(args.stage_a_root / "selection_manifest.csv")
    }
    candidates = {row["image_name"]: row for row in read_rows(candidate_manifest)}
    merged: list[dict[str, object]] = []
    area_correlations: list[float] = []
    activation_ranges: list[float] = []
    activation_stds: list[float] = []
    activation_means: list[float] = []
    area_ranks: dict[str, list[float]] = defaultdict(list)
    for image_id, base in baseline.items():
        stem = Path(image_id).stem
        candidate_row = candidates[image_id]
        candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            masks = payload["sam_masks"].astype(bool)
        score_path = args.stage_a_root / selections[(BASELINE, image_id)]["score_path"]
        with np.load(score_path, allow_pickle=False) as payload:
            candidate_indices = payload["candidate_indices"].astype(np.int64)
            bas_scores = payload["bas_scores"].astype(np.float64)
        candidate_area = masks[candidate_indices].mean(axis=(1, 2))
        candidate_area_rank = average_percentile_rank(candidate_area)
        area_correlations.append(rank_correlation(bas_scores, candidate_area))
        for variant in VARIANTS:
            local = int(selections[(variant, image_id)]["selected_local_index"])
            area_ranks[variant].append(float(candidate_area_rank[local]))
        activation = np.load(
            args.stage_a_root / "activation_maps" / f"{stem}.npy",
            allow_pickle=False,
        ).astype(np.float64)
        activation_ranges.append(float(np.ptp(activation)))
        activation_stds.append(float(activation.std()))
        activation_means.append(float(activation.mean()))
        primary = variants[PRIMARY][image_id]
        identity = ident[image_id]
        row: dict[str, object] = {
            "image_id": image_id,
            "group_id": base["group_id"],
            "size_group": base["size_group"],
            "baseline_dice": float(base["dice"]),
            "primary_dice": float(primary["dice"]),
            "primary_delta": float(primary["dice"]) - float(base["dice"]),
            "baseline_miss": int(base["complete_miss"]),
            "primary_miss": int(primary["complete_miss"]),
            "baseline_area_gt": float(base["selected_gt_area_ratio"]),
            "primary_area_gt": float(primary["selected_gt_area_ratio"]),
            "score_dominance_gap": float(identity["score_dominance_gap"]),
            "baseline_oracle_rank": int(float(base["oracle_rank"])),
            "primary_oracle_rank": int(float(primary["oracle_rank"])),
            "bas_area_rank_correlation": area_correlations[-1],
            "activation_range": activation_ranges[-1],
            "activation_std": activation_stds[-1],
            "activation_mean": activation_means[-1],
        }
        for variant in VARIANTS:
            row[f"{variant}_dice"] = float(variants[variant][image_id]["dice"])
            row[f"{variant}_area_rank"] = area_ranks[variant][-1]
        merged.append(row)

    training = read_rows(args.stage_a_root / "training_history.csv")
    if len(training) != 100:
        raise ValueError("BAS training history must contain 100 epochs")
    tail = training[1:]
    train_images = 2981
    train_normals = 1493
    train_tumors = 1488
    constant_label_map_bas = 1.2 * train_tumors / train_images
    diagnostics = json.loads(
        (args.stage_a_root / "label_safe_diagnostics.json").read_text(encoding="utf-8")
    )
    actual = stage_b["actual_binary_mask_metrics"]
    ident_overall = ident_summary["subgroups"]["overall"]
    exact_identification = ident_overall["exact_regret_decomposition"]
    identification = {
        "baseline_dice": actual[BASELINE]["overall"]["dice"],
        "eligible_oracle_dice": float(ident_overall["eligible_oracle_dice"]),
        "total_regret": float(exact_identification["sum"]),
        "dominance_gap": float(
            exact_identification["score_dominance_identifiability_gap"]
        ),
        "dominance_share": float(
            exact_identification["score_dominance_identifiability_gap"]
        )
        / float(exact_identification["sum"]),
        "nonlinear_monotone_frontier_gap": float(
            exact_identification["nonlinear_monotone_frontier_gap"]
        ),
        "per_image_weight_gap": float(exact_identification["per_image_weight_gap"]),
        "candidate_truncation_regret": float(
            actual[BASELINE]["overall"]["candidate_truncation_regret"]
        ),
    }
    selectors = {
        "all": lambda row: True,
        "severe_dominance": lambda row: float(row["score_dominance_gap"]) >= 0.05,
        "baseline_miss": lambda row: int(row["baseline_miss"]) == 1,
        "underextent": lambda row: float(row["baseline_area_gt"]) < 0.5,
        "calibrated_extent": lambda row: 0.5 <= float(row["baseline_area_gt"]) < 2.0,
        "moderate_overextent": lambda row: 2.0 <= float(row["baseline_area_gt"]) < 10.0,
        "extreme_overextent": lambda row: float(row["baseline_area_gt"]) >= 10.0,
    }
    summary: dict[str, object] = {
        "stage": "rich_gallery_g1_post_bas_b2_bottleneck_v1",
        "actual_binary_mask_metrics": actual,
        "training_collapse": {
            "epoch1_accuracy": float(training[0]["accuracy"]),
            "tail_accuracy_min": min(float(row["accuracy"]) for row in tail),
            "tail_accuracy_max": max(float(row["accuracy"]) for row in tail),
            "tail_full_ce_min": min(float(row["full_ce"]) for row in tail),
            "tail_full_ce_max": max(float(row["full_ce"]) for row in tail),
            "tail_foreground_ce_min": min(
                float(row["foreground_ce"]) for row in tail
            ),
            "tail_foreground_ce_max": max(
                float(row["foreground_ce"]) for row in tail
            ),
            "tail_bas_mean": mean(float(row["bas"]) for row in tail),
            "constant_logit_argmax_accuracy": train_normals / train_images,
            "constant_label_map_bas": constant_label_map_bas,
            "tail_bas_minus_constant_label_map": mean(
                float(row["bas"]) for row in tail
            )
            - constant_label_map_bas,
            "validation_auroc": float(diagnostics["auroc"]),
            "activation_range_mean": float(diagnostics["activation_range_mean"]),
            "tumor_nondegenerate_fraction": float(
                diagnostics["tumor_nondegenerate_activation_fraction"]
            ),
        },
        "candidate_area_proxy": {
            "bas_area_correlation_mean": mean(area_correlations),
            "bas_area_correlation_median": median(area_correlations),
            "fraction_above_0_9": mean(value > 0.9 for value in area_correlations),
            "activation_range_mean_tumor": mean(activation_ranges),
            "activation_std_mean_tumor": mean(activation_stds),
            "activation_mean_tumor": mean(activation_means),
            "selected_area_rank": {
                variant: {"mean": mean(values), "median": median(values)}
                for variant, values in area_ranks.items()
            },
        },
        "two_score_identifiability": identification,
        "regimes": {
            name: regime_summary([row for row in merged if predicate(row)])
            for name, predicate in selectors.items()
        },
        "stage_a_prediction_freeze_sha256": args.expected_freeze_sha256,
        "stage_b_summary_sha256": args.expected_summary_sha256,
        "stage_b_per_image_sha256": args.expected_per_image_sha256,
        "identifiability_per_image_sha256": args.expected_identifiability_per_image_sha256,
        "identifiability_summary_sha256": args.expected_identifiability_summary_sha256,
        "validation_tumors": 184,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_sha = write_rows(args.output_dir / "per_image.csv", merged)
    dossier = render_dossier(summary)
    dossier_path = args.output_dir / "POST_BAS_B2_BOTTLENECK_DOSSIER.md"
    dossier_path.write_text(dossier, encoding="utf-8")
    summary["per_image_sha256"] = per_image_sha
    summary["dossier_sha256"] = sha256_file(dossier_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "pass": True,
        "summary_sha256": sha256_file(summary_path),
        "per_image_sha256": per_image_sha,
        "dossier_sha256": sha256_file(dossier_path),
        "candidate_payloads_verified": 184,
        "validation_gt_reopened": False,
        "test_evaluated": False,
    }
    (args.output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "audit": audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
