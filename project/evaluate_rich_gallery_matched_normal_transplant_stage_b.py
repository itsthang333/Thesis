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

LAYER_METRICS = (
    "feature_l2_inside",
    "feature_l2_ring",
    "feature_l2_contrast",
    "relative_feature_l2_inside",
    "relative_feature_l2_ring",
    "relative_feature_l2_contrast",
    "cosine_inside",
    "cosine_ring",
    "delta_energy_inside_fraction",
    "mask_mass",
    "ring_mass",
)


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


def recipient_pair_sign_agreement(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Recover sign agreement for the frozen two-recipient population moments.

    With exactly two recipients and population standard deviation, the two
    observations are ``mean-std`` and ``mean+std`` (up to ordering).  This
    derives the predeclared sign-stability diagnostic without retaining donor
    identities or changing any Stage-A score.
    """

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != std.shape:
        raise ValueError("recipient mean/std shapes differ")
    return ((mean - std) * (mean + std) > 0.0).astype(np.float64)


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


def _finite_summary(values: list[object]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    return {
        "n": float(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _layer_stratum_summary(
    layer_images: list[dict[str, object]],
    *,
    selector,
) -> dict[str, object]:
    chosen = [row for row in layer_images if selector(row)]
    result: dict[str, object] = {"n": len(chosen), "stages": {}}
    for stage in DENSENET_DIAGNOSTIC_STAGES:
        prefix = f"{stage}_"
        stage_summary: dict[str, object] = {}
        for arm in ("matched", "random"):
            stage_summary[arm] = {
                "oracle_percentile": _finite_summary(
                    [row[prefix + arm + "_oracle_percentile"] for row in chosen]
                ),
                "quality_rank_correlation": _finite_summary(
                    [row[prefix + arm + "_quality_rank_correlation"] for row in chosen]
                ),
                "area_rank_correlation": _finite_summary(
                    [row[prefix + arm + "_area_rank_correlation"] for row in chosen]
                ),
                "oracle_relative_l2_contrast": _finite_summary(
                    [row[prefix + arm + "_oracle_relative_l2_contrast"] for row in chosen]
                ),
                "oracle_recipient_cv": _finite_summary(
                    [row[prefix + arm + "_oracle_recipient_cv"] for row in chosen]
                ),
            }
        stage_summary["matched_minus_random"] = {
            "oracle_percentile": _finite_summary(
                [row[prefix + "oracle_percentile_gain"] for row in chosen]
            ),
            "quality_rank_correlation": _finite_summary(
                [row[prefix + "quality_rank_correlation_gain"] for row in chosen]
            ),
        }
        result["stages"][stage] = stage_summary
    result["terminal"] = {
        "matched_class_inside_oracle_percentile": _finite_summary(
            [row["matched_class_inside_oracle_percentile"] for row in chosen]
        ),
        "random_class_inside_oracle_percentile": _finite_summary(
            [row["random_class_inside_oracle_percentile"] for row in chosen]
        ),
        "matched_logit_oracle_percentile": _finite_summary(
            [row["matched_logit_oracle_percentile"] for row in chosen]
        ),
        "random_logit_oracle_percentile": _finite_summary(
            [row["random_logit_oracle_percentile"] for row in chosen]
        ),
        "matched_random_score_rank_correlation": _finite_summary(
            [row["matched_random_score_rank_correlation"] for row in chosen]
        ),
        "matched_recipient_sign_agreement_fraction": _finite_summary(
            [row["matched_recipient_sign_agreement_fraction"] for row in chosen]
        ),
    }
    return result


def _layer_bottleneck(layer_images: list[dict[str, object]]) -> dict[str, object]:
    # Backward-compatible compact branch logic is retained for synthetic tests,
    # while real Stage-B rows receive a matched-vs-random, subgroup and
    # baseline-failure decomposition below.
    deep_rows = "pool0_matched_oracle_percentile" in layer_images[0]
    if deep_rows:
        overall = _layer_stratum_summary(layer_images, selector=lambda _row: True)
        strata = {
            "small": _layer_stratum_summary(
                layer_images, selector=lambda row: row["size_group"] == "small"
            ),
            "medium": _layer_stratum_summary(
                layer_images, selector=lambda row: row["size_group"] == "medium"
            ),
            "large": _layer_stratum_summary(
                layer_images, selector=lambda row: row["size_group"] == "large"
            ),
            "baseline_complete_miss": _layer_stratum_summary(
                layer_images, selector=lambda row: bool(row["baseline_complete_miss"])
            ),
            "baseline_overlap": _layer_stratum_summary(
                layer_images, selector=lambda row: not bool(row["baseline_complete_miss"])
            ),
            "baseline_wrong_source": _layer_stratum_summary(
                layer_images, selector=lambda row: bool(row["baseline_wrong_source"])
            ),
            "baseline_correct_source": _layer_stratum_summary(
                layer_images, selector=lambda row: not bool(row["baseline_wrong_source"])
            ),
        }
        stages = overall["stages"]
        early = float(stages["pool0"]["matched"]["oracle_percentile"]["median"])
        final_feature = float(stages["norm5"]["matched"]["oracle_percentile"]["median"])
        matched_random_gain = float(
            stages["norm5"]["matched_minus_random"]["oracle_percentile"]["median"]
        )
        class_inside = float(
            overall["terminal"]["matched_class_inside_oracle_percentile"]["median"]
        )
        logit = float(overall["terminal"]["matched_logit_oracle_percentile"]["median"])
        if early < 0.55:
            branch = "candidate_content_not_discriminative_at_stem_after_sham_cancellation"
        elif final_feature < early - 0.10:
            branch = "backbone_erases_early_candidate_signal"
        elif class_inside < final_feature - 0.10:
            branch = "final_representation_change_is_not_tumor_specific"
        elif logit < class_inside - 0.10:
            branch = "global_pooling_dilutes_spatial_tumor_response"
        elif matched_random_gain < 0.03:
            branch = "matched_recipients_do_not_add_tumor_specific_candidate_identity"
        else:
            branch = "candidate_rank_or_fixed_fusion_remains_top1_bottleneck"
        return {
            "overall": overall,
            "strata": strata,
            "identified_first_failure_branch": branch,
            "matched_random_norm5_oracle_percentile_gain_median": matched_random_gain,
            "interpretation_thresholds_are_diagnostic_not_selector_tuning": True,
        }

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


def _baseline_failure_decomposition(
    per_image: list[dict[str, object]],
) -> dict[str, object]:
    rows = [row for row in per_image if row["variant"] == BASELINE]
    if len(rows) != 184:
        raise ValueError("baseline failure decomposition requires 184 tumor images")

    def summarize(chosen: list[dict[str, object]]) -> dict[str, object]:
        if not chosen:
            return {"n": 0}
        oracle = np.asarray([row["oracle_dice"] for row in chosen], dtype=np.float64)
        eligible = np.asarray(
            [row["eligible_oracle_dice"] for row in chosen], dtype=np.float64
        )
        selected = np.asarray([row["dice"] for row in chosen], dtype=np.float64)
        return {
            "n": len(chosen),
            "actual_dice": float(selected.mean()),
            "gallery_oracle_dice": float(oracle.mean()),
            "eligible_oracle_dice": float(eligible.mean()),
            "proposal_supply_regret": float((oracle - eligible).mean()),
            "selector_regret_within_eligible_gallery": float((eligible - selected).mean()),
            "within_selected_source_regret": float(
                np.mean([row["within_selected_source_regret"] for row in chosen])
            ),
            "cross_source_regret": float(
                np.mean([row["cross_source_regret"] for row in chosen])
            ),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in chosen)),
            "wrong_source": int(
                sum(not bool(row["selected_source_matches_oracle"]) for row in chosen)
            ),
            "extent_counts": dict(sorted(Counter(str(row["extent_class"]) for row in chosen).items())),
            "median_selected_gt_area_ratio": float(
                np.median([row["selected_gt_area_ratio"] for row in chosen])
            ),
            "mean_precision": float(np.mean([row["precision"] for row in chosen])),
            "mean_recall": float(np.mean([row["recall"] for row in chosen])),
        }

    result = {"overall": summarize(rows), "subgroups": {}}
    for group in ("small", "medium", "large"):
        result["subgroups"][group] = summarize(
            [row for row in rows if row["size_group"] == group]
        )
    result["failure_strata"] = {
        "complete_miss": summarize([row for row in rows if row["complete_miss"]]),
        "over_extent": summarize([row for row in rows if row["extent_class"] == "over"]),
        "under_extent": summarize([row for row in rows if row["extent_class"] == "under"]),
        "wrong_source": summarize(
            [row for row in rows if not bool(row["selected_source_matches_oracle"])]
        ),
        "correct_source": summarize(
            [row for row in rows if bool(row["selected_source_matches_oracle"])]
        ),
    }
    return result


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
        baseline_local = stable_select(panel[BASELINE], g1)
        baseline_prediction = proposals[int(candidate_indices[baseline_local])]
        baseline_intersection = int(np.logical_and(baseline_prediction, target).sum())
        baseline_area_ratio = float(
            baseline_prediction.sum() / max(1, int(target.sum()))
        )
        oracle_source = str(eligible_sources[eligible_oracle_local])
        layer_row: dict[str, object] = {
            "image_id": image_id,
            "group_id": selections[(BASELINE, image_id)]["group_id"],
            "size_group": subgroup,
            "baseline_dice": float(eligible_quality[baseline_local]),
            "baseline_complete_miss": int(baseline_intersection == 0),
            "baseline_wrong_source": int(
                str(eligible_sources[baseline_local]) != oracle_source
            ),
            "baseline_extent_class": (
                "over" if baseline_area_ratio > 2.0
                else "under" if baseline_area_ratio < 0.5
                else "near"
            ),
            "baseline_selected_gt_area_ratio": baseline_area_ratio,
            "eligible_oracle_source": oracle_source,
        }
        for stage_index, stage in enumerate(DENSENET_DIAGNOSTIC_STAGES):
            arm_values: dict[str, dict[str, float]] = {}
            for arm in ("matched", "random"):
                signal = payload[f"{arm}_relative_feature_l2_contrast_mean"][:, stage_index]
                signal_std = payload[
                    f"{arm}_relative_feature_l2_contrast_recipient_std"
                ][:, stage_index]
                ranks = average_percentile_rank(signal)
                arm_values[arm] = {
                    "oracle_percentile": float(ranks[eligible_oracle_local]),
                    "quality_rank_correlation": rank_correlation(signal, eligible_quality),
                    "area_rank_correlation": rank_correlation(signal, eligible_area),
                    "oracle_relative_l2_contrast": float(signal[eligible_oracle_local]),
                    "oracle_recipient_cv": float(
                        signal_std[eligible_oracle_local]
                        / max(1.0e-8, abs(float(signal[eligible_oracle_local])))
                    ),
                }
                for name, value in arm_values[arm].items():
                    layer_row[f"{stage}_{arm}_{name}"] = value
            layer_row[f"{stage}_oracle_percentile_gain"] = (
                arm_values["matched"]["oracle_percentile"]
                - arm_values["random"]["oracle_percentile"]
            )
            layer_row[f"{stage}_quality_rank_correlation_gain"] = (
                arm_values["matched"]["quality_rank_correlation"]
                - arm_values["random"]["quality_rank_correlation"]
            )
            # Preserve the original compact columns as aliases for the matched arm.
            layer_row[f"{stage}_oracle_percentile"] = arm_values["matched"][
                "oracle_percentile"
            ]
            layer_row[f"{stage}_quality_rank_correlation"] = arm_values["matched"][
                "quality_rank_correlation"
            ]
            layer_row[f"{stage}_area_rank_correlation"] = arm_values["matched"][
                "area_rank_correlation"
            ]
        for arm in ("matched", "random"):
            class_inside_arm = payload[f"{arm}_class_response_delta_inside_mean"]
            logit_arm = payload[f"{arm}_score"]
            layer_row[f"{arm}_class_inside_oracle_percentile"] = float(
                average_percentile_rank(class_inside_arm)[eligible_oracle_local]
            )
            layer_row[f"{arm}_class_inside_quality_rank_correlation"] = rank_correlation(
                class_inside_arm, eligible_quality
            )
            layer_row[f"{arm}_logit_oracle_percentile"] = float(
                average_percentile_rank(logit_arm)[eligible_oracle_local]
            )
            layer_row[f"{arm}_logit_quality_rank_correlation"] = rank_correlation(
                logit_arm, eligible_quality
            )
            layer_row[f"{arm}_logit_area_rank_correlation"] = rank_correlation(
                logit_arm, eligible_area
            )
            layer_row[f"{arm}_recipient_sign_agreement_fraction"] = float(
                recipient_pair_sign_agreement(
                    logit_arm, payload[f"{arm}_recipient_std"]
                ).mean()
            )
        class_inside = payload["matched_class_response_delta_inside_mean"]
        layer_row["class_inside_oracle_percentile"] = layer_row[
            "matched_class_inside_oracle_percentile"
        ]
        layer_row["class_inside_quality_rank_correlation"] = layer_row[
            "matched_class_inside_quality_rank_correlation"
        ]
        layer_row["logit_oracle_percentile"] = layer_row[
            "matched_logit_oracle_percentile"
        ]
        layer_row["logit_quality_rank_correlation"] = layer_row[
            "matched_logit_quality_rank_correlation"
        ]
        layer_row["logit_area_rank_correlation"] = layer_row[
            "matched_logit_area_rank_correlation"
        ]
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
                "is_eligible_oracle": int(local == eligible_oracle_local),
                "is_baseline_selected": int(local == baseline_local),
                "g1_logit": float(g1[local]),
                "upstream_score": float(upstream[local]),
                "matched_logit_delta": float(payload["matched_score"][local]),
                "random_logit_delta": float(payload["random_score"][local]),
                "matched_recipient_std": float(payload["matched_recipient_std"][local]),
                "random_recipient_std": float(payload["random_recipient_std"][local]),
                "matched_recipient_sign_agreement": float(
                    recipient_pair_sign_agreement(
                        payload["matched_score"], payload["matched_recipient_std"]
                    )[local]
                ),
                "random_recipient_sign_agreement": float(
                    recipient_pair_sign_agreement(
                        payload["random_score"], payload["random_recipient_std"]
                    )[local]
                ),
                "matched_class_inside_delta": float(class_inside[local]),
                "matched_class_ring_delta": float(
                    payload["matched_class_response_delta_ring_mean"][local]
                ),
                "matched_class_contrast_delta": float(
                    payload["matched_class_response_delta_contrast_mean"][local]
                ),
                "matched_class_global_delta": float(
                    payload["matched_class_response_delta_global_mean"][local]
                ),
                "random_class_inside_delta": float(
                    payload["random_class_response_delta_inside_mean"][local]
                ),
                "random_class_ring_delta": float(
                    payload["random_class_response_delta_ring_mean"][local]
                ),
                "random_class_contrast_delta": float(
                    payload["random_class_response_delta_contrast_mean"][local]
                ),
                "random_class_global_delta": float(
                    payload["random_class_response_delta_global_mean"][local]
                ),
            }
            for stage_index, stage in enumerate(DENSENET_DIAGNOSTIC_STAGES):
                for arm in ("matched", "random"):
                    for metric in LAYER_METRICS:
                        candidate_record[f"{stage}_{arm}_{metric}"] = float(
                            payload[f"{arm}_{metric}_mean"][local, stage_index]
                        )
                        candidate_record[f"{stage}_{arm}_{metric}_recipient_std"] = float(
                            payload[f"{arm}_{metric}_recipient_std"][local, stage_index]
                        )
                # Preserve original compact aliases for downstream compatibility.
                candidate_record[f"{stage}_relative_l2_contrast"] = candidate_record[
                    f"{stage}_matched_relative_feature_l2_contrast"
                ]
                candidate_record[f"{stage}_delta_energy_inside"] = candidate_record[
                    f"{stage}_matched_delta_energy_inside_fraction"
                ]
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
                    "eligible_candidate_count": len(candidate_indices),
                    "selected_local_index": int(local),
                    "selected_candidate_index": selected_index,
                    "eligible_oracle_local_index": eligible_oracle_local,
                    "eligible_oracle_candidate_index": int(
                        candidate_indices[eligible_oracle_local]
                    ),
                    "selected_variant_score": float(panel[variant][local]),
                    "selected_g1_logit": float(g1[local]),
                    "selected_upstream_score": float(upstream[local]),
                    "selected_matched_logit_delta": float(
                        payload["matched_score"][local]
                    ),
                    "selected_random_logit_delta": float(
                        payload["random_score"][local]
                    ),
                    "dice": selected_dice,
                    "iou": iou(prediction, target),
                    "precision": float(intersection / max(1, prediction_area)),
                    "recall": float(intersection / max(1, target_area)),
                    "complete_miss": int(intersection == 0),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_gt_area_ratio": float(prediction_area / max(1, target_area)),
                    "extent_class": (
                        "over" if prediction_area / max(1, target_area) > 2.0
                        else "under" if prediction_area / max(1, target_area) < 0.5
                        else "near"
                    ),
                    "selected_source": selected_source,
                    "eligible_oracle_source": oracle_source,
                    "selected_source_matches_oracle": int(selected_source == oracle_source),
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
    baseline_failure = _baseline_failure_decomposition(per_image)
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
        "stage": "rich_gallery_matched_normal_transplant_stage_b_v2",
        "stage_b_evaluator_sha256": sha256_file(Path(__file__)),
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "stage_a_audit_sha256": args.expected_stage_a_audit_sha256,
        "split_sha256": args.expected_split_sha256,
        "variants": summary,
        "bootstrap": bootstrap,
        "baseline_failure_decomposition": baseline_failure,
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
