from __future__ import annotations

"""Deep validation-only decomposition of the frozen G1 rank-fusion baseline."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_g2_objective import (
    average_percentile_rank,
    rank_fusion_scores,
    stable_select,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_g2_selector_pair import canonical_source


TOP_K = (1, 3, 5, 10, 20, 50)
SUBGROUPS = ("overall", "small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--expected-g2-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--consensus-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def percentile_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def records_for(rows: Iterable[dict[str, object]], subgroup: str) -> list[dict[str, object]]:
    return [row for row in rows if subgroup == "overall" or row["size_group"] == subgroup]


def summarize_numeric(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("canonical validation cohort mismatch")
    freeze_path = args.g2_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_g2_freeze_sha256:
        raise ValueError("G2 freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G2 freeze contract mismatch")
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("analysis requires the complete validation gallery")

    selection_path = args.g2_root / "stage_a_selection_manifest.csv"
    if sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("G2 selection manifest changed")
    with selection_path.open("r", newline="", encoding="utf-8-sig") as handle:
        selection_rows = list(csv.DictReader(handle))
    frozen_baseline = {
        row["image_id"]: int(row["selected_candidate_index"])
        for row in selection_rows
        if row["variant"] == "g1_frozen__rank_fusion"
    }
    if len(frozen_baseline) != 371:
        raise ValueError("frozen G1 fusion baseline cohort mismatch")

    consensus_rows: dict[tuple[str, str], dict[str, str]] = {}
    with args.consensus_evaluation.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            consensus_rows[(row["variant"], row["image_id"])] = row

    # Annotation boundary: all baseline and diagnostic choices were frozen above.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_id = dataset[index]
        image_id = str(image_id)
        split_row = next(row for row in val_rows if row["image_id"] == image_id)
        if split_row["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        gt_pixels = int(target.sum())
        area = float(target.mean())
        subgroup = size_group(area)
        stem = Path(image_id).stem
        candidate_row = candidate_rows[stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        score_path = args.g2_root / "stage_a_scores" / f"{stem}.npz"
        with np.load(score_path, allow_pickle=False) as score:
            kept = score["candidate_indices"].astype(np.int64)
            g1 = score["g1_frozen_candidate_logits"].astype(np.float64)
            upstream = score["upstream_scores"].astype(np.float64)
        with np.load(candidate_path, allow_pickle=False) as payload:
            all_proposals = payload["sam_masks"].astype(bool)
            all_sam_scores = payload["sam_scores"].astype(np.float64)
            all_causal_scores = payload["classifier_causal_scores"].astype(np.float64)
            all_prompt_modes = payload["prompt_modes"].astype(str)
            all_source_names = np.asarray(
                [canonical_source(value) for value in payload["proposal_source_ids"]],
                dtype="U32",
            )
            proposals = all_proposals[kept]
            sam_scores = all_sam_scores[kept]
            causal_scores = all_causal_scores[kept]
            prompt_modes = all_prompt_modes[kept]
            source_names = np.asarray(
                all_source_names[kept], dtype="U32"
            )
        if not (len(proposals) == len(g1) == len(upstream) == len(source_names)):
            raise ValueError(f"candidate alignment mismatch: {image_id}")
        proposal_pixels = proposals.sum(axis=(1, 2), dtype=np.int64)
        intersections = np.logical_and(proposals, target).sum(axis=(1, 2), dtype=np.int64)
        denominators = proposal_pixels + gt_pixels
        candidate_dice = np.divide(
            2.0 * intersections,
            denominators,
            out=np.zeros(len(proposals), dtype=np.float64),
            where=denominators > 0,
        )
        all_proposal_pixels = all_proposals.sum(axis=(1, 2), dtype=np.int64)
        all_intersections = np.logical_and(all_proposals, target).sum(
            axis=(1, 2), dtype=np.int64
        )
        all_denominators = all_proposal_pixels + gt_pixels
        all_candidate_dice = np.divide(
            2.0 * all_intersections,
            all_denominators,
            out=np.zeros(len(all_proposals), dtype=np.float64),
            where=all_denominators > 0,
        )
        fusion = rank_fusion_scores(g1, upstream)
        selected = stable_select(fusion, g1)
        if int(kept[selected]) != frozen_baseline[image_id]:
            raise RuntimeError(f"baseline reproduction failed: {image_id}")
        raw_selected = stable_select(g1, g1)
        upstream_selected = stable_select(upstream, g1)
        sam_selected = stable_select(sam_scores, g1)
        causal_selected = stable_select(causal_scores, g1)
        order = np.asarray(
            sorted(
                range(len(fusion)),
                key=lambda candidate: (fusion[candidate], g1[candidate], -candidate),
                reverse=True,
            ),
            dtype=np.int64,
        )
        eligible_oracle_dice = float(candidate_dice.max())
        gallery_oracle_dice = float(all_candidate_dice.max())
        oracle_candidates = np.flatnonzero(
            np.isclose(candidate_dice, eligible_oracle_dice, atol=1e-12)
        )
        position = {int(candidate): rank for rank, candidate in enumerate(order, start=1)}
        oracle_local = min(oracle_candidates, key=lambda candidate: position[int(candidate)])
        oracle_rank = position[int(oracle_local)]
        selected_source = str(source_names[selected])
        selected_source_members = source_names == selected_source
        selected_source_best = float(candidate_dice[selected_source_members].max())
        eligible_oracle_source = str(source_names[int(oracle_local)])
        gallery_oracle_local = int(all_candidate_dice.argmax())
        gallery_oracle_source = str(all_source_names[gallery_oracle_local])
        selected_dice = float(candidate_dice[selected])
        selected_ratio = float(proposal_pixels[selected] / gt_pixels)
        oracle_ratio = float(proposal_pixels[int(oracle_local)] / gt_pixels)
        topk_values = {
            f"top{k}_oracle_dice": float(candidate_dice[order[: min(k, len(order))]].max())
            for k in TOP_K
        }
        baseline_consensus = consensus_rows[("g1_upstream_baseline", image_id)]
        equal_consensus = consensus_rows[("g1_upstream_consensus_equal", image_id)]
        product_consensus = consensus_rows[("g1_upstream_consensus_product", image_id)]
        per_image.append(
            {
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "size_group": subgroup,
                "gt_area_ratio": area,
                "candidate_count": len(proposals),
                "baseline_dice": selected_dice,
                "raw_g1_dice": float(candidate_dice[raw_selected]),
                "upstream_only_dice": float(candidate_dice[upstream_selected]),
                "sam_score_only_dice": float(candidate_dice[sam_selected]),
                "causal_score_only_dice": float(candidate_dice[causal_selected]),
                "fusion_delta_vs_raw": selected_dice - float(candidate_dice[raw_selected]),
                "fusion_delta_vs_upstream": selected_dice - float(candidate_dice[upstream_selected]),
                "oracle_dice": gallery_oracle_dice,
                "eligible_oracle_dice": eligible_oracle_dice,
                "selector_regret": gallery_oracle_dice - selected_dice,
                "truncation_regret": gallery_oracle_dice - eligible_oracle_dice,
                "cross_source_regret": eligible_oracle_dice - selected_source_best,
                "within_selected_source_regret": selected_source_best - selected_dice,
                "selected_source_best_dice": selected_source_best,
                "selected_source": selected_source,
                "selected_prompt_mode": str(prompt_modes[selected]),
                "oracle_source": gallery_oracle_source,
                "eligible_oracle_source": eligible_oracle_source,
                "eligible_oracle_prompt_mode": str(prompt_modes[int(oracle_local)]),
                "selected_source_matches_oracle": int(
                    selected_source == gallery_oracle_source
                ),
                "complete_miss": int(intersections[selected] == 0),
                "oracle_complete_miss": int(all_candidate_dice.max() == 0.0),
                "selected_gt_area_ratio": selected_ratio,
                "oracle_gt_area_ratio": oracle_ratio,
                "oracle_rank_under_fusion": oracle_rank,
                "selected_g1_rank": float(average_percentile_rank(g1)[selected]),
                "selected_upstream_rank": float(average_percentile_rank(upstream)[selected]),
                "oracle_g1_rank": float(average_percentile_rank(g1)[int(oracle_local)]),
                "oracle_upstream_rank": float(average_percentile_rank(upstream)[int(oracle_local)]),
                "g1_dice_rank_correlation": percentile_rank_correlation(g1, candidate_dice),
                "upstream_dice_rank_correlation": percentile_rank_correlation(upstream, candidate_dice),
                "fusion_dice_rank_correlation": percentile_rank_correlation(fusion, candidate_dice),
                "fusion_area_rank_correlation": percentile_rank_correlation(
                    fusion, np.log1p(proposal_pixels.astype(np.float64))
                ),
                "sam_dice_rank_correlation": percentile_rank_correlation(
                    sam_scores, candidate_dice
                ),
                "causal_dice_rank_correlation": percentile_rank_correlation(
                    causal_scores, candidate_dice
                ),
                "area_dice_rank_correlation": percentile_rank_correlation(
                    np.log1p(proposal_pixels.astype(np.float64)), candidate_dice
                ),
                "causal_nonzero_fraction": float(np.mean(causal_scores != 0.0)),
                "classifier448_oracle_dice": float(
                    candidate_dice[source_names == "classifier448"].max()
                ),
                "layercam320_oracle_dice": float(
                    candidate_dice[source_names == "layercam320"].max()
                ),
                "external_saliency_oracle_dice": float(
                    candidate_dice[source_names == "external_saliency"].max()
                ),
                "baseline_selected_consensus_iou": float(
                    baseline_consensus["selected_consensus_iou"]
                ),
                "consensus_equal_dice": float(equal_consensus["dice"]),
                "consensus_product_dice": float(product_consensus["dice"]),
                **topk_values,
            }
        )
    if len(per_image) != 184 or Counter(row["size_group"] for row in per_image) != Counter(
        {"small": 94, "medium": 72, "large": 18}
    ):
        raise RuntimeError("tumor/subgroup cohort mismatch")

    summary: dict[str, object] = {
        "stage": "rich_gallery_g1_rank_fusion_bottleneck_decomposition_v1",
        "baseline_formula": "0.5*percentile_rank(g1_logit)+0.5*percentile_rank(upstream_score)",
        "cohort": {"tumor": 184, "small": 94, "medium": 72, "large": 18},
        "subgroups": {},
        "source_decomposition": {},
        "fusion_help_harm": {},
        "miss_recoverability": {},
        "area_error_bins": {},
        "test_images_read": 0,
        "test_evaluated": False,
    }
    for subgroup in SUBGROUPS:
        rows = records_for(per_image, subgroup)
        current = {
            "n": len(rows),
            "baseline_dice": float(np.mean([row["baseline_dice"] for row in rows])),
            "raw_g1_dice": float(np.mean([row["raw_g1_dice"] for row in rows])),
            "upstream_only_dice": float(np.mean([row["upstream_only_dice"] for row in rows])),
            "sam_score_only_dice": float(
                np.mean([row["sam_score_only_dice"] for row in rows])
            ),
            "causal_score_only_dice": float(
                np.mean([row["causal_score_only_dice"] for row in rows])
            ),
            "oracle_dice": float(np.mean([row["oracle_dice"] for row in rows])),
            "selector_regret": float(np.mean([row["selector_regret"] for row in rows])),
            "truncation_regret": float(np.mean([row["truncation_regret"] for row in rows])),
            "cross_source_regret": float(np.mean([row["cross_source_regret"] for row in rows])),
            "within_selected_source_regret": float(
                np.mean([row["within_selected_source_regret"] for row in rows])
            ),
            "complete_misses": int(sum(row["complete_miss"] for row in rows)),
            "oracle_complete_misses": int(sum(row["oracle_complete_miss"] for row in rows)),
            "selected_source_match_rate": float(
                np.mean([row["selected_source_matches_oracle"] for row in rows])
            ),
            "selected_gt_area_ratio": summarize_numeric(rows, "selected_gt_area_ratio"),
            "oracle_rank_under_fusion": summarize_numeric(rows, "oracle_rank_under_fusion"),
            "rank_correlations": {
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "g1_dice_rank_correlation",
                    "upstream_dice_rank_correlation",
                    "fusion_dice_rank_correlation",
                    "fusion_area_rank_correlation",
                    "sam_dice_rank_correlation",
                    "causal_dice_rank_correlation",
                    "area_dice_rank_correlation",
                )
            },
            "source_oracle_dice": {
                source: float(np.mean([row[f"{source}_oracle_dice"] for row in rows]))
                for source in ("classifier448", "layercam320", "external_saliency")
            },
            "topk_oracle_dice": {
                str(k): float(np.mean([row[f"top{k}_oracle_dice"] for row in rows]))
                for k in TOP_K
            },
            "oracle_rank_recall": {
                str(k): float(
                    np.mean([int(int(row["oracle_rank_under_fusion"]) <= k) for row in rows])
                )
                for k in TOP_K
            },
        }
        summary["subgroups"][subgroup] = current

    for source in ("classifier448", "layercam320", "external_saliency"):
        rows = [row for row in per_image if row["selected_source"] == source]
        summary["source_decomposition"][source] = {
            "selected_n": len(rows),
            "mean_dice": float(np.mean([row["baseline_dice"] for row in rows])),
            "misses": int(sum(row["complete_miss"] for row in rows)),
            "median_selected_gt_area_ratio": float(
                np.median([row["selected_gt_area_ratio"] for row in rows])
            ),
            "oracle_source_n": int(sum(row["oracle_source"] == source for row in per_image)),
        }
    raw_delta = np.asarray([row["fusion_delta_vs_raw"] for row in per_image], dtype=np.float64)
    upstream_delta = np.asarray(
        [row["fusion_delta_vs_upstream"] for row in per_image], dtype=np.float64
    )
    summary["fusion_help_harm"] = {
        "vs_raw": {
            "wins": int((raw_delta > 1e-12).sum()),
            "losses": int((raw_delta < -1e-12).sum()),
            "ties": int((np.abs(raw_delta) <= 1e-12).sum()),
            "mean_delta": float(raw_delta.mean()),
            "positive_mass": float(raw_delta[raw_delta > 0].sum()),
            "negative_mass": float(raw_delta[raw_delta < 0].sum()),
        },
        "vs_upstream": {
            "wins": int((upstream_delta > 1e-12).sum()),
            "losses": int((upstream_delta < -1e-12).sum()),
            "ties": int((np.abs(upstream_delta) <= 1e-12).sum()),
            "mean_delta": float(upstream_delta.mean()),
        },
    }
    summary["metadata_diagnostic"] = {
        "selected_prompt_modes": dict(Counter(row["selected_prompt_mode"] for row in per_image)),
        "eligible_oracle_prompt_modes": dict(
            Counter(row["eligible_oracle_prompt_mode"] for row in per_image)
        ),
        "mean_causal_nonzero_fraction": float(
            np.mean([row["causal_nonzero_fraction"] for row in per_image])
        ),
        "sam_score_only_misses": int(
            sum(float(row["sam_score_only_dice"]) == 0.0 for row in per_image)
        ),
        "causal_score_only_misses": int(
            sum(float(row["causal_score_only_dice"]) == 0.0 for row in per_image)
        ),
    }
    misses = [row for row in per_image if row["complete_miss"] == 1]
    summary["miss_recoverability"] = {
        "baseline_misses": len(misses),
        "recoverable_oracle_overlap": int(sum(row["oracle_dice"] > 0 for row in misses)),
        "oracle_dice_ge_0_1": int(sum(row["oracle_dice"] >= 0.1 for row in misses)),
        "oracle_dice_ge_0_3": int(sum(row["oracle_dice"] >= 0.3 for row in misses)),
        "oracle_dice_ge_0_5": int(sum(row["oracle_dice"] >= 0.5 for row in misses)),
        "median_oracle_dice": float(np.median([row["oracle_dice"] for row in misses])),
        "median_oracle_rank": float(
            np.median([row["oracle_rank_under_fusion"] for row in misses])
        ),
        "subgroups": dict(Counter(row["size_group"] for row in misses)),
    }
    ratio_bins = (
        ("under_lt_0_5", 0.0, 0.5),
        ("matched_0_5_to_2", 0.5, 2.0),
        ("over_2_to_10", 2.0, 10.0),
        ("severe_over_ge_10", 10.0, float("inf")),
    )
    for name, lower, upper in ratio_bins:
        rows = [
            row
            for row in per_image
            if float(row["selected_gt_area_ratio"]) >= lower
            and float(row["selected_gt_area_ratio"]) < upper
        ]
        summary["area_error_bins"][name] = {
            "n": len(rows),
            "dice": float(np.mean([row["baseline_dice"] for row in rows])) if rows else None,
            "misses": int(sum(row["complete_miss"] for row in rows)),
            "subgroups": dict(Counter(row["size_group"] for row in rows)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "split_sha256": args.expected_split_sha256,
        "g2_freeze_sha256": args.expected_g2_freeze_sha256,
        "candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "validation_gt_read_only_after_frozen_baseline": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
