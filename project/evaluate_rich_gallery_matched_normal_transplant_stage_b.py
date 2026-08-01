from __future__ import annotations

"""Actual-Dice evaluator and layerwise failure decomposition for transplant Stage A."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.matched_normal_candidate_transplant import (
    DENSENET_DIAGNOSTIC_STAGES,
    frozen_selector_panel,
)
from models.rich_gallery_g2_objective import average_percentile_rank, stable_select
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


VARIANTS = (
    "g1_upstream_baseline",
    "transplant_only",
    "baseline_transplant_equal",
    "baseline_transplant_three_to_one",
    "baseline_random_control_three_to_one",
)
BASELINE = "g1_upstream_baseline"
PRIMARY = "baseline_transplant_three_to_one"
RANDOM_CONTROL = "baseline_random_control_three_to_one"
EXPECTED_BASELINE_DICE = 0.28872948670665205


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown rich-gallery source: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--stage-a-audit", type=Path, required=True)
    parser.add_argument("--expected-stage-a-audit-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / max(1, int(prediction.sum()) + int(target.sum())))


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    return float(intersection / max(1, int(np.logical_or(prediction, target).sum())))


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _verify_stage_a(args: argparse.Namespace):
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("prediction-freeze SHA-256 mismatch")
    if sha256_file(args.stage_a_audit) != args.expected_stage_a_audit_sha256:
        raise ValueError("Stage-A independent-audit SHA-256 mismatch")
    audit = json.loads(args.stage_a_audit.read_text(encoding="utf-8"))
    if (
        audit.get("pass") is not True
        or audit.get("validation_images_verified") != 371
        or audit.get("validation_gt_read") is not False
        or audit.get("test_evaluated") is not False
    ):
        raise ValueError("Stage-A independent audit contract mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_matched_normal_transplant_stage_a_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or tuple(freeze.get("variants", [])) != VARIANTS
        or freeze.get("validation_images") != 371
        or freeze.get("selection_rows") != 371 * len(VARIANTS)
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Stage-A freeze contract mismatch")
    manifest_path = args.prediction_root / "selection_manifest.csv"
    if sha256_file(manifest_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("selection manifest changed")
    indexed = {}
    for row in _read_csv(manifest_path):
        key = (row["variant"], row["image_id"])
        if row["variant"] not in VARIANTS or key in indexed:
            raise ValueError("selection identity mismatch")
        indexed[key] = row
    if len(indexed) != 371 * len(VARIANTS):
        raise ValueError("selection cohort incomplete")
    return freeze, indexed


def _summarize(rows: list[dict[str, object]], variant: str) -> dict[str, object]:
    result: dict[str, object] = {}
    records = [row for row in rows if row["variant"] == variant]
    for group in ("overall", "small", "medium", "large"):
        chosen = [row for row in records if group == "overall" or row["size_group"] == group]
        result[group] = {
            "n": len(chosen),
            "dice": float(np.mean([row["dice"] for row in chosen])),
            "iou": float(np.mean([row["iou"] for row in chosen])),
            "precision": float(np.mean([row["precision"] for row in chosen])),
            "recall": float(np.mean([row["recall"] for row in chosen])),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in chosen)),
            "selected_gt_area_ratio_median": float(
                np.median([row["selected_gt_area_ratio"] for row in chosen])
            ),
            "selector_regret": float(np.mean([row["selector_regret"] for row in chosen])),
            "within_selected_source_regret": float(
                np.mean([row["within_selected_source_regret"] for row in chosen])
            ),
            "cross_source_regret": float(
                np.mean([row["cross_source_regret"] for row in chosen])
            ),
            "oracle_rank_median": float(np.median([row["oracle_rank"] for row in chosen])),
            "selected_source_counts": dict(
                sorted(Counter(str(row["selected_source"]) for row in chosen).items())
            ),
        }
    return result


def _paired_bootstrap(
    rows: list[dict[str, object]],
    left: str,
    right: str,
    *,
    group: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    indexed = {(row["variant"], row["image_id"]): row for row in rows}
    selected = [
        row for row in rows
        if row["variant"] == left and (group == "overall" or row["size_group"] == group)
    ]
    by_group: defaultdict[str, list[float]] = defaultdict(list)
    for row in selected:
        paired = indexed[(right, row["image_id"])]
        by_group[str(row["group_id"])].append(float(row["dice"]) - float(paired["dice"]))
    values = np.asarray([np.mean(item) for item in by_group.values()], dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(values), size=(replicates, len(values)))
    means = values[sampled].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "groups": float(len(values)),
    }


def _layer_bottleneck(layer_images: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for stage in DENSENET_DIAGNOSTIC_STAGES:
        prefix = f"{stage}_"
        summary[stage] = {
            "oracle_percentile_median": float(
                np.median([row[prefix + "oracle_percentile"] for row in layer_images])
            ),
            "quality_rank_correlation_mean": float(
                np.mean([row[prefix + "quality_rank_correlation"] for row in layer_images])
            ),
            "area_rank_correlation_mean": float(
                np.mean([row[prefix + "area_rank_correlation"] for row in layer_images])
            ),
        }
    final_feature = float(summary["norm5"]["oracle_percentile_median"])
    early = float(summary["pool0"]["oracle_percentile_median"])
    class_inside = float(np.median([row["class_inside_oracle_percentile"] for row in layer_images]))
    logit = float(np.median([row["logit_oracle_percentile"] for row in layer_images]))
    if early < 0.55:
        branch = "candidate_content_not_discriminative_at_stem_after_sham_cancellation"
    elif final_feature < early - 0.10:
        branch = "backbone_erases_early_candidate_signal"
    elif class_inside < final_feature - 0.10:
        branch = "final_representation_change_is_not_tumor_specific"
    elif logit < class_inside - 0.10:
        branch = "global_pooling_dilutes_spatial_tumor_response"
    else:
        branch = "candidate_rank_or_fixed_fusion_remains_top1_bottleneck"
    return {
        "stages": summary,
        "class_inside_oracle_percentile_median": class_inside,
        "logit_oracle_percentile_median": logit,
        "identified_first_failure_branch": branch,
        "interpretation_thresholds_are_diagnostic_not_selector_tuning": True,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("matched-normal Stage-B output must not exist")
    if args.bootstrap_replicates != 10000 or args.bootstrap_seed != 20260802:
        raise ValueError("bootstrap controls differ from frozen protocol")
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise ValueError("canonical validation cohort mismatch")
    freeze, selections = _verify_stage_a(args)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("candidate validation cohort incomplete")

    # This is the sole annotation boundary.  Stage A and its independent audit
    # have already fixed every score and candidate choice.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    per_image: list[dict[str, object]] = []
    per_candidate: list[dict[str, object]] = []
    layer_images: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, target_tensor, image_id = dataset[index]
        image_id = str(image_id)
        if selections[(BASELINE, image_id)]["tumor"] != "1":
            continue
        target = target_tensor[0].numpy() > 0.5
        subgroup = size_group(float(target.mean()))
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed after Stage A: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as candidate:
            proposals = candidate["sam_masks"].astype(bool)
            sources = candidate["proposal_source_ids"].astype(str)
        quality = np.asarray([dice(mask, target) for mask in proposals], dtype=np.float64)
        areas = proposals.mean(axis=(1, 2)).astype(np.float64)
        global_oracle = float(quality.max())
        score_path = args.prediction_root / selections[(BASELINE, image_id)]["score_path"]
        if sha256_file(score_path) != selections[(BASELINE, image_id)]["score_sha256"]:
            raise ValueError(f"Stage-A score payload changed: {image_id}")
        with np.load(score_path, allow_pickle=False) as score:
            payload = {name: np.asarray(score[name]) for name in score.files}
        candidate_indices = payload["candidate_indices"].astype(np.int64)
        eligible_quality = quality[candidate_indices]
        eligible_area = areas[candidate_indices]
        eligible_sources = np.asarray(
            [canonical_source(value) for value in payload["proposal_sources"].astype(str)]
        )
        g1 = payload["g1_logits"].astype(np.float64)
        upstream = payload["upstream_scores"].astype(np.float64)
        baseline_fusion = 0.5 * (
            average_percentile_rank(g1) + average_percentile_rank(upstream)
        )
        panel = frozen_selector_panel(
            baseline_fusion,
            payload["matched_score"],
            payload["random_score"],
        )
        eligible_oracle_local = int(eligible_quality.argmax())
        layer_row: dict[str, object] = {
            "image_id": image_id,
            "group_id": selections[(BASELINE, image_id)]["group_id"],
            "size_group": subgroup,
        }
        for stage_index, stage in enumerate(DENSENET_DIAGNOSTIC_STAGES):
            signal = payload["matched_relative_feature_l2_contrast_mean"][:, stage_index]
            ranks = average_percentile_rank(signal)
            layer_row[f"{stage}_oracle_percentile"] = float(ranks[eligible_oracle_local])
            layer_row[f"{stage}_quality_rank_correlation"] = rank_correlation(
                signal, eligible_quality
            )
            layer_row[f"{stage}_area_rank_correlation"] = rank_correlation(
                signal, eligible_area
            )
        class_inside = payload["matched_class_response_delta_inside_mean"]
        logit_signal = payload["matched_score"]
        layer_row["class_inside_oracle_percentile"] = float(
            average_percentile_rank(class_inside)[eligible_oracle_local]
        )
        layer_row["class_inside_quality_rank_correlation"] = rank_correlation(
            class_inside, eligible_quality
        )
        layer_row["logit_oracle_percentile"] = float(
            average_percentile_rank(logit_signal)[eligible_oracle_local]
        )
        layer_row["logit_quality_rank_correlation"] = rank_correlation(
            logit_signal, eligible_quality
        )
        layer_row["logit_area_rank_correlation"] = rank_correlation(
            logit_signal, eligible_area
        )
        layer_row["matched_random_score_rank_correlation"] = rank_correlation(
            payload["matched_score"], payload["random_score"]
        )
        layer_images.append(layer_row)

        for local, candidate_index in enumerate(candidate_indices):
            candidate_record: dict[str, object] = {
                "image_id": image_id,
                "group_id": layer_row["group_id"],
                "size_group": subgroup,
                "candidate_local_index": local,
                "candidate_index": int(candidate_index),
                "source": eligible_sources[local],
                "candidate_area_ratio": float(eligible_area[local]),
                "candidate_dice": float(eligible_quality[local]),
                "g1_logit": float(g1[local]),
                "upstream_score": float(upstream[local]),
                "matched_logit_delta": float(payload["matched_score"][local]),
                "random_logit_delta": float(payload["random_score"][local]),
                "matched_recipient_std": float(payload["matched_recipient_std"][local]),
                "matched_class_inside_delta": float(class_inside[local]),
                "matched_class_ring_delta": float(
                    payload["matched_class_response_delta_ring_mean"][local]
                ),
            }
            for stage_index, stage in enumerate(DENSENET_DIAGNOSTIC_STAGES):
                candidate_record[f"{stage}_relative_l2_contrast"] = float(
                    payload["matched_relative_feature_l2_contrast_mean"][local, stage_index]
                )
                candidate_record[f"{stage}_delta_energy_inside"] = float(
                    payload["matched_delta_energy_inside_fraction_mean"][local, stage_index]
                )
            per_candidate.append(candidate_record)

        canonical_sources = np.asarray([canonical_source(value) for value in sources])
        for variant in VARIANTS:
            local = stable_select(panel[variant], g1)
            frozen = selections[(variant, image_id)]
            if (
                local != int(frozen["selected_local_index"])
                or int(candidate_indices[local]) != int(frozen["selected_candidate_index"])
            ):
                raise ValueError(f"Stage-A selection does not reproduce: {(variant, image_id)}")
            selected_index = int(candidate_indices[local])
            prediction = proposals[selected_index]
            selected_dice = dice(prediction, target)
            intersection = int(np.logical_and(prediction, target).sum())
            prediction_area = int(prediction.sum())
            target_area = int(target.sum())
            selected_source = canonical_sources[selected_index]
            source_oracle = float(eligible_quality[eligible_sources == selected_source].max())
            order = np.asarray(
                sorted(
                    range(len(panel[variant])),
                    key=lambda item: (panel[variant][item], g1[item], -item),
                    reverse=True,
                ),
                dtype=np.int64,
            )
            oracle_rank = int(np.flatnonzero(order == eligible_oracle_local)[0]) + 1
            per_image.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": frozen["group_id"],
                    "size_group": subgroup,
                    "gt_area_ratio": float(target.mean()),
                    "dice": selected_dice,
                    "iou": iou(prediction, target),
                    "precision": float(intersection / max(1, prediction_area)),
                    "recall": float(intersection / max(1, target_area)),
                    "complete_miss": int(intersection == 0),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_gt_area_ratio": float(prediction_area / max(1, target_area)),
                    "selected_source": selected_source,
                    "oracle_dice": global_oracle,
                    "eligible_oracle_dice": float(eligible_quality.max()),
                    "selector_regret": global_oracle - selected_dice,
                    "cross_source_regret": float(eligible_quality.max()) - source_oracle,
                    "within_selected_source_regret": source_oracle - selected_dice,
                    "oracle_rank": oracle_rank,
                    "score_quality_rank_correlation": rank_correlation(
                        panel[variant], eligible_quality
                    ),
                }
            )

    if len(per_image) != 184 * len(VARIANTS) or len(layer_images) != 184:
        raise RuntimeError("matched-normal Stage-B cohort mismatch")
    counts = Counter(row["size_group"] for row in layer_images)
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"subgroup mismatch: {counts}")
    summary = {variant: _summarize(per_image, variant) for variant in VARIANTS}
    for variant in VARIANTS:
        for group in ("overall", "small", "medium", "large"):
            summary[variant][group]["delta_vs_baseline"] = (
                summary[variant][group]["dice"] - summary[BASELINE][group]["dice"]
            )
    if abs(summary[BASELINE]["overall"]["dice"] - EXPECTED_BASELINE_DICE) > 1.0e-12:
        raise RuntimeError("immutable G1+upstream baseline Dice does not reproduce")
    bootstrap = {
        "primary_vs_baseline": {
            group: _paired_bootstrap(
                per_image,
                PRIMARY,
                BASELINE,
                group=group,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + offset,
            )
            for offset, group in enumerate(("overall", "small", "medium", "large"))
        },
        "primary_vs_random_control": {
            group: _paired_bootstrap(
                per_image,
                PRIMARY,
                RANDOM_CONTROL,
                group=group,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 20 + offset,
            )
            for offset, group in enumerate(("overall", "small", "medium", "large"))
        },
    }
    layerwise = _layer_bottleneck(layer_images)
    primary_metrics = summary[PRIMARY]
    baseline_metrics = summary[BASELINE]
    random_metrics = summary[RANDOM_CONTROL]
    promotion = {
        "primary_variant": PRIMARY,
        "overall_beats_baseline": primary_metrics["overall"]["dice"]
        > baseline_metrics["overall"]["dice"],
        "overall_beats_matched_random_control": primary_metrics["overall"]["dice"]
        > random_metrics["overall"]["dice"],
        "small_not_materially_harmed": primary_metrics["small"]["dice"]
        >= baseline_metrics["small"]["dice"] - 0.01,
        "medium_not_materially_harmed": primary_metrics["medium"]["dice"]
        >= baseline_metrics["medium"]["dice"] - 0.01,
    }
    promotion["pass"] = all(
        bool(promotion[key])
        for key in (
            "overall_beats_baseline",
            "overall_beats_matched_random_control",
            "small_not_materially_harmed",
            "medium_not_materially_harmed",
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_sha = _write_csv(args.output_dir / "per_image_results.csv", per_image)
    per_candidate_sha = _write_csv(
        args.output_dir / "per_candidate_layerwise.csv", per_candidate
    )
    layer_image_sha = _write_csv(
        args.output_dir / "per_image_layerwise_summary.csv", layer_images
    )
    result = {
        "stage": "rich_gallery_matched_normal_transplant_stage_b_v1",
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "stage_a_audit_sha256": args.expected_stage_a_audit_sha256,
        "split_sha256": args.expected_split_sha256,
        "variants": summary,
        "bootstrap": bootstrap,
        "layerwise_bottleneck": layerwise,
        "promotion": promotion,
        "validation_images": 371,
        "tumor_images_evaluated": 184,
        "subgroups": {"small": 94, "medium": 72, "large": 18},
        "per_image_results_sha256": per_image_sha,
        "per_candidate_layerwise_sha256": per_candidate_sha,
        "per_image_layerwise_summary_sha256": layer_image_sha,
        "candidate_scores_frozen_before_validation_gt": True,
        "validation_gt_opened_only_in_stage_b": True,
        "test_evaluated": False,
    }
    result_path = args.output_dir / "evaluation_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "summary_sha256": sha256_file(result_path)}, indent=2))


if __name__ == "__main__":
    main()
