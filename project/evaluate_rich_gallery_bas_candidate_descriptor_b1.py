from __future__ import annotations

"""Post-freeze spatial evaluator and failure decomposition for rich BAS-B1."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank, stable_select
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_bas_candidate_descriptor_b1 import VARIANTS, canonical_source


BASELINE = "g1_upstream_baseline"
PRIMARY = "g1_upstream_bas_three_way"
EXPECTED_BASELINE_DICE = 0.28872948670665205


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / max(1, int(prediction.sum()) + int(target.sum())))


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / max(1, union))


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 0 or np.std(right_rank) <= 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_stage_a(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, str]]]:
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("rich BAS prediction-freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_bas_b1_stage_a_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
        or freeze.get("validation_images") != 371
        or freeze.get("selection_rows") != 371 * len(VARIANTS)
        or tuple(freeze.get("variants", [])) != VARIANTS
        or freeze.get("baseline_reproduction_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("rich BAS prediction-freeze contract mismatch")
    manifest_path = args.prediction_root / "selection_manifest.csv"
    if sha256_file(manifest_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("rich BAS selection manifest changed")
    cohort = {row["image_id"] for row in val_rows}
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    score_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in _read_rows(manifest_path):
        key = (row["variant"], row["image_id"])
        if row["variant"] not in VARIANTS or row["image_id"] not in cohort or key in indexed:
            raise ValueError("rich BAS selection identity mismatch")
        score_path = args.prediction_root / row["score_path"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"rich BAS score payload changed: {key}")
        if row["score_path"] not in score_cache:
            with np.load(score_path, allow_pickle=False) as payload:
                score_cache[row["score_path"]] = {name: payload[name] for name in payload.files}
        payload = score_cache[row["score_path"]]
        candidate_indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        g1_logits = np.asarray(payload["g1_logits"], dtype=np.float64)
        scores = np.asarray(payload[row["variant"]], dtype=np.float64)
        if not (len(candidate_indices) == len(g1_logits) == len(scores)):
            raise ValueError("rich BAS frozen arrays are misaligned")
        local = stable_select(scores, g1_logits)
        if (
            local != int(row["selected_local_index"])
            or int(candidate_indices[local]) != int(row["selected_candidate_index"])
        ):
            raise ValueError(f"rich BAS frozen choice does not reproduce: {key}")
        indexed[key] = row
    if len(indexed) != 371 * len(VARIANTS):
        raise ValueError("rich BAS frozen selection cohort is incomplete")
    return freeze, indexed


def _paired_bootstrap(
    rows: list[dict[str, object]],
    *,
    group: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    selected = [row for row in rows if group == "overall" or row["size_group"] == group]
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        by_group[str(row["group_id"])].append(float(row["primary_delta"]))
    group_values = np.asarray([np.mean(values) for values in by_group.values()], dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(group_values), size=(replicates, len(group_values)))
    means = group_values[sampled].mean(axis=1)
    return {
        "mean_delta": float(group_values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "groups": float(len(group_values)),
    }


def _summarize_variant(rows: list[dict[str, object]], variant: str) -> dict[str, Any]:
    records = [row for row in rows if row["variant"] == variant]
    result: dict[str, Any] = {}
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
            "oracle_rank_median": float(np.median([row["oracle_rank"] for row in chosen])),
            "oracle_rank_p90": float(np.quantile([row["oracle_rank"] for row in chosen], 0.90)),
            "score_quality_rank_correlation_mean": float(
                np.mean([row["score_quality_rank_correlation"] for row in chosen])
            ),
            "selected_source_counts": dict(
                sorted(Counter(str(row["selected_source"]) for row in chosen).items())
            ),
        }
    return result


def _failure_decomposition(
    per_image: list[dict[str, object]],
    summary: Mapping[str, Any],
    label_safe: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = {
        str(row["image_id"]): row for row in per_image if row["variant"] == BASELINE
    }
    primary = {
        str(row["image_id"]): row for row in per_image if row["variant"] == PRIMARY
    }
    paired: list[dict[str, object]] = []
    transitions: Counter[str] = Counter()
    transition_mass: defaultdict[str, float] = defaultdict(float)
    for image_id in sorted(baseline):
        before = baseline[image_id]
        after = primary[image_id]
        delta = float(after["dice"]) - float(before["dice"])
        transition = f"{before['selected_source']}->{after['selected_source']}"
        transitions[transition] += 1
        transition_mass[transition] += delta
        paired.append(
            {
                "image_id": image_id,
                "group_id": before["group_id"],
                "size_group": before["size_group"],
                "primary_delta": delta,
                "baseline_miss": int(before["complete_miss"]),
                "primary_miss": int(after["complete_miss"]),
            }
        )
    deltas = np.asarray([row["primary_delta"] for row in paired], dtype=np.float64)
    hit_transitions = Counter(
        f"{row['baseline_miss']}->{row['primary_miss']}" for row in paired
    )
    baseline_metrics = summary[BASELINE]
    primary_metrics = summary[PRIMARY]
    bas_only_metrics = summary["bas_only"]
    branches: list[str] = []
    if bas_only_metrics["overall"]["dice"] <= baseline_metrics["overall"]["dice"]:
        branches.append("bas_map_lacks_sufficient_standalone_candidate_identity")
    if float(label_safe.get("mean_bas_upstream_rank_correlation", 0.0)) > 0.80:
        branches.append("bas_evidence_duplicates_upstream_extent")
    if (
        primary_metrics["overall"]["complete_misses"] < baseline_metrics["overall"]["complete_misses"]
        and primary_metrics["small"]["selected_gt_area_ratio_median"]
        > baseline_metrics["small"]["selected_gt_area_ratio_median"]
        and primary_metrics["small"]["precision"] < baseline_metrics["small"]["precision"]
    ):
        branches.append("bas_recovers_hits_but_expands_background_on_small_lesions")
    if (
        primary_metrics["large"]["recall"] < baseline_metrics["large"]["recall"]
        and primary_metrics["large"]["selected_gt_area_ratio_median"]
        < baseline_metrics["large"]["selected_gt_area_ratio_median"]
    ):
        branches.append("bas_remains_discriminative_fragment_underextent_on_large_lesions")
    diagnostic_best = max(
        summary[name]["overall"]["dice"]
        for name in ("g1_bas_two_way", "upstream_bas_two_way", "bas_only")
    )
    if (
        diagnostic_best > baseline_metrics["overall"]["dice"]
        and primary_metrics["overall"]["dice"] <= baseline_metrics["overall"]["dice"]
    ):
        branches.append("equal_three_way_fusion_dilutes_complementary_bas_signal")
    return {
        "primary_delta": {
            group: primary_metrics[group]["dice"] - baseline_metrics[group]["dice"]
            for group in ("overall", "small", "medium", "large")
        },
        "wins": int((deltas > 1.0e-12).sum()),
        "losses": int((deltas < -1.0e-12).sum()),
        "ties": int((np.abs(deltas) <= 1.0e-12).sum()),
        "positive_dice_mass": float(deltas[deltas > 0].sum()),
        "negative_dice_mass": float(deltas[deltas < 0].sum()),
        "hit_miss_transitions": dict(sorted(hit_transitions.items())),
        "source_transition_counts": dict(sorted(transitions.items())),
        "source_transition_dice_mass": dict(sorted(transition_mass.items())),
        "candidate_supply": {
            "gallery_oracle_dice": baseline_metrics["overall"]["oracle_dice"],
            "baseline_selector_regret": baseline_metrics["overall"]["selector_regret"],
            "primary_selector_regret": primary_metrics["overall"]["selector_regret"],
        },
        "identified_failure_branches": branches,
        "no_next_gpu_run_before_manual_dossier_review": True,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10000 or args.bootstrap_seed != 20260801:
        raise ValueError("rich BAS bootstrap controls differ from design")
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("canonical validation cohort mismatch")
    freeze, selections = verify_stage_a(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("rich BAS evaluator requires complete validation candidates")

    # Annotation boundary: every candidate vector and choice was physically
    # verified above.  Only this separate evaluator imports segmentation data.
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
        _image, target_tensor, image_id = dataset[index]
        image_id = str(image_id)
        if selections[(BASELINE, image_id)]["tumor"] != "1":
            continue
        target = target_tensor[0].numpy() > 0.5
        target_area = int(target.sum())
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed after freeze: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
        candidate_dice = np.asarray([dice(mask, target) for mask in proposals])
        oracle_index = int(candidate_dice.argmax())
        oracle_dice = float(candidate_dice[oracle_index])
        subgroup = size_group(float(target.mean()))
        score_path = args.prediction_root / selections[(BASELINE, image_id)]["score_path"]
        with np.load(score_path, allow_pickle=False) as score_payload:
            candidate_indices = score_payload["candidate_indices"].astype(np.int64)
            variant_scores = {
                variant: score_payload[variant].astype(np.float64) for variant in VARIANTS
            }
        eligible_quality = candidate_dice[candidate_indices]
        for variant in VARIANTS:
            selection = selections[(variant, image_id)]
            selected_index = int(selection["selected_candidate_index"])
            prediction = proposals[selected_index]
            intersection = int(np.logical_and(prediction, target).sum())
            prediction_area = int(prediction.sum())
            scores = variant_scores[variant]
            order = np.argsort(-scores, kind="stable")
            eligible_oracle_local = int(eligible_quality.argmax())
            oracle_rank = int(np.flatnonzero(order == eligible_oracle_local)[0]) + 1
            per_image.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": selection["group_id"],
                    "size_group": subgroup,
                    "gt_area_ratio": float(target.mean()),
                    "dice": dice(prediction, target),
                    "iou": iou(prediction, target),
                    "precision": float(intersection / max(1, prediction_area)),
                    "recall": float(intersection / max(1, target_area)),
                    "complete_miss": int(intersection == 0),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_gt_area_ratio": float(prediction_area / max(1, target_area)),
                    "selected_source": canonical_source(sources[selected_index]),
                    "oracle_dice": oracle_dice,
                    "selector_regret": oracle_dice - dice(prediction, target),
                    "oracle_rank": oracle_rank,
                    "score_quality_rank_correlation": rank_correlation(scores, eligible_quality),
                }
            )
    if len(per_image) != 184 * len(VARIANTS):
        raise RuntimeError("rich BAS tumor evaluation count mismatch")
    counts = Counter(
        row["size_group"] for row in per_image if row["variant"] == BASELINE
    )
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"rich BAS subgroup mismatch: {counts}")
    summary = {variant: _summarize_variant(per_image, variant) for variant in VARIANTS}
    for variant in VARIANTS:
        for group in ("overall", "small", "medium", "large"):
            summary[variant][group]["delta_vs_baseline"] = (
                summary[variant][group]["dice"] - summary[BASELINE][group]["dice"]
            )
            summary[variant][group]["oracle_dice"] = float(
                np.mean([
                    row["oracle_dice"]
                    for row in per_image
                    if row["variant"] == variant
                    and (group == "overall" or row["size_group"] == group)
                ])
            )
    if abs(summary[BASELINE]["overall"]["dice"] - EXPECTED_BASELINE_DICE) > 1.0e-12:
        raise RuntimeError("rich BAS evaluator does not reproduce baseline Dice")
    paired_rows: list[dict[str, object]] = []
    baseline_by_id = {
        str(row["image_id"]): row for row in per_image if row["variant"] == BASELINE
    }
    primary_by_id = {
        str(row["image_id"]): row for row in per_image if row["variant"] == PRIMARY
    }
    for image_id in sorted(baseline_by_id):
        baseline = baseline_by_id[image_id]
        primary = primary_by_id[image_id]
        paired_rows.append(
            {
                "image_id": image_id,
                "group_id": baseline["group_id"],
                "size_group": baseline["size_group"],
                "primary_delta": float(primary["dice"]) - float(baseline["dice"]),
                "baseline_miss": baseline["complete_miss"],
                "primary_miss": primary["complete_miss"],
            }
        )
    bootstrap = {
        group: _paired_bootstrap(
            paired_rows,
            group=group,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + offset,
        )
        for offset, group in enumerate(("overall", "small", "medium", "large"))
    }
    label_safe = json.loads(
        (args.prediction_root / "label_safe_diagnostics.json").read_text(encoding="utf-8")
    )
    failure = _failure_decomposition(per_image, summary, label_safe)
    promotion = {
        "overall_improved": summary[PRIMARY]["overall"]["dice"] > EXPECTED_BASELINE_DICE,
        "overall_ci95_low_above_zero": bootstrap["overall"]["ci95_low"] > 0.0,
        "no_subgroup_mean_decrease": all(
            summary[PRIMARY][group]["dice"] >= summary[BASELINE][group]["dice"]
            for group in ("small", "medium", "large")
        ),
        "no_complete_miss_increase": summary[PRIMARY]["overall"]["complete_misses"]
        <= summary[BASELINE]["overall"]["complete_misses"],
    }
    promotion["pass"] = bool(all(promotion.values()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_image)
    result = {
        "stage": "rich_gallery_bas_b1_post_freeze_evaluation_v1",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187, "small": 94, "medium": 72, "large": 18},
        "actual_binary_mask_metrics": summary,
        "paired_bootstrap_primary_vs_baseline": bootstrap,
        "promotion": promotion,
        "failure_decomposition": failure,
        "label_safe_diagnostics": label_safe,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "spatial_ground_truth_used_for_training": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
