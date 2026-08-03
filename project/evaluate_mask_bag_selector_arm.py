from __future__ import annotations

"""Prediction-first evaluator for hash-frozen mask-bag selector arms."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    sha256_file,
)
from models.mask_bag_ranking_diagnostics import (
    candidate_ranking_diagnostics,
    summarize_ranking_diagnostics,
)
from models.mask_bag_score_evidence import validate_candidate_score_manifest
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.mask_bag_selector_cache_io import load_selector_cache_record


SUBGROUPS = ("overall", "small", "medium", "large")
TOP_K = (1, 3, 5, 10)
OPERATIONAL_GOALS = {
    "overall": 0.34024039,
    "small": 0.17895493,
    "medium": 0.51244178,
    "large": 0.49370336,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--expected-arm-freeze-sha256", required=True)
    parser.add_argument("--expected-arm-source-commit", required=True)
    parser.add_argument("--expected-arm-protocol-sha256", required=True)
    parser.add_argument("--expected-score-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-candidate-logit-provenance-field",
        choices=("candidate_logit_tta", "candidate_logit_recipe"),
        required=True,
    )
    parser.add_argument(
        "--expected-candidate-logit-provenance-value",
        required=True,
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-baseline-per-image-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    return parser.parse_args()


def _size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman(values_a: list[float], values_b: list[float]) -> float | None:
    first = np.asarray(values_a, dtype=np.float64)
    second = np.asarray(values_b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("Spearman inputs must be aligned vectors")
    ranks_a = _average_ranks(first)
    ranks_b = _average_ranks(second)
    if np.ptp(ranks_a) == 0.0 or np.ptp(ranks_b) == 0.0:
        return None
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _candidate_logit_provenance_matches(
    prediction: dict[str, str],
    *,
    expected_field: str,
    expected_value: str,
) -> bool:
    allowed = {
        ("candidate_logit_tta", "mean_original_aligned_horizontal_flip"),
        ("candidate_logit_recipe", "within_image_equal_percentile_rank_no_tta"),
    }
    if (expected_field, expected_value) not in allowed:
        return False
    present = {
        field
        for field in ("candidate_logit_tta", "candidate_logit_recipe")
        if field in prediction
    }
    return present == {expected_field} and prediction[expected_field] == expected_value


def _paired_group_bootstrap(
    arm: list[float],
    baseline: list[float],
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if len(arm) != len(baseline) or len(arm) != len(groups):
        raise ValueError("paired bootstrap vectors must have equal length")
    delta = np.asarray(arm, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    by_group: dict[str, list[float]] = {}
    for value, group in zip(delta, groups):
        by_group.setdefault(group, []).append(float(value))
    unique = sorted(by_group)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.integers(0, len(unique), size=len(unique))
        values = [value for position in sampled for value in by_group[unique[position]]]
        bootstrap[index] = float(np.mean(values))
    return {
        "delta_arm_minus_baseline": float(delta.mean()),
        "ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_images": len(delta),
        "n_groups": len(unique),
    }


def _verify_prediction_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str,
    val_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    manifest_path = root / "predictions" / "prediction_manifest.csv"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("prediction manifest SHA-256 mismatch")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    expected = {row["image_id"]: row for row in val_rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != set(expected):
        raise ValueError("prediction manifest cohort mismatch")
    ordered: list[dict[str, str]] = []
    for image_id in expected:
        row = indexed[image_id]
        if (
            row["group_id"] != expected[image_id]["group_id"]
            or row["tumor"] != expected[image_id]["tumor"]
        ):
            raise ValueError(f"prediction identity mismatch: {image_id}")
        map_path = root / "predictions" / row["map_path"]
        if not map_path.is_file() or sha256_file(map_path) != row["map_sha256"]:
            raise ValueError(f"prediction map hash mismatch: {image_id}")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"prediction map content mismatch: {image_id}")
        ordered.append(row)
    return ordered


def _verify_selector_cache(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    freeze_path = args.selector_cache_root / "selector_cache_freeze.json"
    if sha256_file(freeze_path) != args.expected_selector_cache_freeze_sha256:
        raise ValueError("selector cache freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("baseline_prediction_freeze_sha256")
        != args.expected_baseline_freeze_sha256
        or freeze.get("cohort") != {"train": 2981, "validation": 371}
        or freeze.get("validation_selected_indices_reproduced") != 371
        or freeze.get("validation_map_hashes_reproduced") != 371
        or freeze.get("train_masks_discarded") is not True
        or freeze.get("validation_masks_bitpacked") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("selector cache freeze provenance/safety mismatch")
    manifest_path = args.selector_cache_root / "selector_cache_manifest.csv"
    if sha256_file(manifest_path) != freeze["selector_cache_manifest_sha256"]:
        raise ValueError("selector cache manifest differs from freeze")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    val_manifest = {row["image_id"]: row for row in rows if row["split"] == "val"}
    expected = {row["image_id"]: row for row in val_rows}
    if len(val_manifest) != 371 or set(val_manifest) != set(expected):
        raise ValueError("selector validation-cache cohort mismatch")
    for image_id in expected:
        row = val_manifest[image_id]
        if (
            row["group_id"] != expected[image_id]["group_id"]
            or row["tumor"] != expected[image_id]["tumor"]
            or int(row["packed_masks_included"]) != 1
        ):
            raise ValueError(f"selector validation-cache identity mismatch: {image_id}")
        load_selector_cache_record(
            args.selector_cache_root / row["cache_path"],
            expected_sha256=row["cache_sha256"],
            require_packed_masks=True,
        )
    return freeze, val_manifest


def _verify_arm(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
    cache_freeze: dict[str, object],
    cache_rows: dict[str, dict[str, str]],
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    freeze_path = args.arm_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_arm_freeze_sha256:
        raise ValueError("arm prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("source_commit") != args.expected_arm_source_commit
        or freeze.get("protocol_sha256") != args.expected_arm_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("selector_cache_freeze_sha256")
        != args.expected_selector_cache_freeze_sha256
        or freeze.get("selector_cache_manifest_sha256")
        != cache_freeze["selector_cache_manifest_sha256"]
        or freeze.get("candidate_score_manifest_sha256")
        != args.expected_score_manifest_sha256
        or freeze.get("validation_predictions") != 371
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("arm prediction-freeze provenance/safety mismatch")
    predictions = _verify_prediction_manifest(
        args.arm_root,
        expected_manifest_sha256=str(freeze["prediction_manifest_sha256"]),
        val_rows=val_rows,
    )
    expected_scores = {
        image_id: {
            "group_id": row["group_id"],
            "tumor": row["tumor"],
            "candidate_payload_sha256": row["candidate_payload_sha256"],
            "candidate_count": int(row["candidate_count"]),
        }
        for image_id, row in cache_rows.items()
    }
    score_root = args.arm_root / "candidate_scores"
    score_rows = validate_candidate_score_manifest(
        score_root,
        expected_manifest_sha256=args.expected_score_manifest_sha256,
        expected_images=expected_scores,
    )
    score_by_id = {row["image_id"]: row for row in score_rows}
    prediction_by_id = {row["image_id"]: row for row in predictions}
    for image_id, cache_row in cache_rows.items():
        cache_payload = load_selector_cache_record(
            args.selector_cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=True,
        )
        score_row = score_by_id[image_id]
        with np.load(
            score_root / score_row["score_path"], allow_pickle=False
        ) as score_payload:
            score_indices = score_payload["candidate_indices"]
        if not np.array_equal(
            score_indices,
            np.asarray(cache_payload["candidate_indices"], dtype=np.int64),
        ):
            raise ValueError(f"arm score/cache candidate indices differ: {image_id}")
        prediction = prediction_by_id[image_id]
        if (
            prediction["candidate_payload_sha256"]
            != cache_row["candidate_payload_sha256"]
            or int(prediction["candidate_count"]) != int(cache_row["candidate_count"])
            or int(prediction["selected_candidate_index"])
            != int(score_row["selected_candidate_index"])
            or abs(
                float(prediction["selected_candidate_logit"])
                - float(score_row["selected_candidate_logit"])
            )
            > 5.0e-6
            or not _candidate_logit_provenance_matches(
                prediction,
                expected_field=args.expected_candidate_logit_provenance_field,
                expected_value=args.expected_candidate_logit_provenance_value,
            )
            or not 0.0 <= float(prediction["bag_probability"]) <= 1.0
        ):
            raise ValueError(f"arm winner/cache provenance mismatch: {image_id}")
    return freeze, predictions, score_by_id


def _verify_baseline(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    freeze_path = args.baseline_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_baseline_freeze_sha256:
        raise ValueError("baseline prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("baseline prediction freeze provenance/safety mismatch")
    predictions = _verify_prediction_manifest(
        args.baseline_root,
        expected_manifest_sha256=str(freeze["prediction_manifest_sha256"]),
        val_rows=val_rows,
    )
    if sha256_file(args.baseline_per_image) != args.expected_baseline_per_image_sha256:
        raise ValueError("baseline per-image SHA-256 mismatch")
    return freeze, predictions


def _write_validation_projection(
    path: Path,
    val_rows: list[dict[str, str]],
) -> str:
    """Project the hash-verified full split onto validation before GT loading.

    The canonical segmentation loader verifies every row present in its manifest
    before selecting a split. Passing the full frozen manifest would therefore
    read train and locked-test image/annotation bytes even for ``split="val"``.
    The full manifest has already been hash-verified by
    ``load_split_rows_without_annotations``; this projection contains only its
    371 eligible validation rows and prevents any locked-test byte access.
    """
    if len(val_rows) != 371 or any(
        row.get("split") != "val" or row.get("eligible") != "1"
        for row in val_rows
    ):
        raise ValueError("validation-only split projection cohort mismatch")
    fieldnames = list(val_rows[0])
    if any(list(row) != fieldnames for row in val_rows):
        raise ValueError("validation-only split projection schema mismatch")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(val_rows)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10000:
        raise ValueError("selector evaluator requires exactly 10,000 replicates")
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("frozen validation cohort mismatch")
    cache_freeze, cache_rows = _verify_selector_cache(args, val_rows)
    arm_freeze, arm_predictions, score_rows = _verify_arm(
        args, val_rows, cache_freeze, cache_rows
    )
    baseline_freeze, baseline_predictions = _verify_baseline(args, val_rows)

    # Protocol boundary: all current-arm/cache/baseline freezes, manifests,
    # scores and physical maps above are verified before any validation target
    # or previous GT-derived per-image value is opened.
    from datasets.factory import build_segmentation_dataset

    with args.baseline_per_image.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        baseline_per_image_rows = list(csv.DictReader(handle))
    baseline_by_id = {row["image_id"]: row for row in baseline_per_image_rows}
    arm_prediction_by_id = {row["image_id"]: row for row in arm_predictions}
    baseline_prediction_by_id = {
        row["image_id"]: row for row in baseline_predictions
    }
    with TemporaryDirectory(prefix="btxrd_val_projection_") as temporary_dir:
        validation_projection = Path(temporary_dir) / "split_manifest_val_only.csv"
        validation_projection_sha256 = _write_validation_projection(
            validation_projection,
            val_rows,
        )
        dataset = build_segmentation_dataset(
            root=args.dataset_root,
            split="val",
            image_size=320,
            augment=False,
            split_manifest=validation_projection,
        )
    per_image: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_name = dataset[index]
        image_id = str(image_name)
        prediction = arm_prediction_by_id[image_id]
        if prediction["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        cache_row = cache_rows[image_id]
        cache_payload = load_selector_cache_record(
            args.selector_cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=True,
        )
        packed_masks = cache_payload["packed_masks"]
        masks = unpack_candidate_masks(packed_masks).astype(bool)
        candidate_indices = np.asarray(cache_payload["candidate_indices"])
        score_row = score_rows[image_id]
        with np.load(
            args.arm_root / "candidate_scores" / score_row["score_path"],
            allow_pickle=False,
        ) as score_payload:
            candidate_scores = score_payload["candidate_logits"].astype(np.float64)
        candidate_quality = np.asarray(
            [_dice(mask, target) for mask in masks], dtype=np.float64
        )
        ranking = candidate_ranking_diagnostics(
            candidate_scores,
            candidate_quality,
            top_k=TOP_K,
        )
        local_selected = int(ranking["selected_candidate_index"])
        local_oracle = int(ranking["oracle_candidate_index"])
        selected_original_index = int(candidate_indices[local_selected])
        oracle_original_index = int(candidate_indices[local_oracle])
        if selected_original_index != int(prediction["selected_candidate_index"]):
            raise RuntimeError("post-freeze ranking winner differs from prediction")
        winner_quality = float(candidate_quality[local_selected])
        ranking["selected_candidate_index"] = selected_original_index
        ranking["oracle_candidate_index"] = oracle_original_index
        map_path = args.arm_root / "predictions" / prediction["map_path"]
        map_values = np.load(map_path, allow_pickle=False).astype(np.float32)
        selected_map = map_values > 0.0
        selected_dice = _dice(selected_map, target)
        baseline = baseline_by_id[image_id]
        baseline_prediction = baseline_prediction_by_id[image_id]
        area_ratio = float(target.mean())
        size_group = _size_group(area_ratio)
        baseline_oracle = float(baseline["oracle_best_single_dice"])
        if abs(float(ranking["oracle_quality"]) - baseline_oracle) > 1.0e-7:
            raise RuntimeError("cache candidate oracle differs from frozen baseline")
        ranking_row = {
            **ranking,
            "image_id": image_id,
            "group_id": prediction["group_id"],
            "size_group": size_group,
            "selected_complete_miss": int(
                not np.logical_and(selected_map, target).any()
            ),
            "baseline_complete_miss": int(baseline["complete_miss"]),
        }
        ranking_rows.append(ranking_row)
        flat: dict[str, Any] = {
            "image_id": image_id,
            "group_id": prediction["group_id"],
            "gt_area_ratio": area_ratio,
            "size_group": size_group,
            "candidate_count": int(prediction["candidate_count"]),
            "selected_candidate_index": selected_original_index,
            "oracle_candidate_index": ranking["oracle_candidate_index"],
            "oracle_best_rank": ranking["oracle_best_rank"],
            "dice": selected_dice,
            "winner_candidate_dice": winner_quality,
            "prediction_map_minus_winner_dice": selected_dice - winner_quality,
            "oracle_best_single_dice": ranking["oracle_quality"],
            "selected_to_oracle_regret": ranking["selected_to_oracle_regret"],
            "score_quality_spearman": ranking["score_quality_spearman"],
            "complete_miss": ranking_row["selected_complete_miss"],
            "baseline_dice": float(baseline["dice"]),
            "baseline_oracle_best_single_dice": baseline_oracle,
            "baseline_complete_miss": ranking_row["baseline_complete_miss"],
            "baseline_candidate_count": int(
                baseline_prediction["candidate_count"]
            ),
        }
        for k in TOP_K:
            flat[f"top_{k}_oracle_reach"] = int(
                ranking["top_k_oracle_reach"][str(k)]
            )
            flat[f"top_{k}_best_dice"] = ranking["top_k_best_quality"][str(k)]
            flat[f"top_{k}_regret"] = ranking["top_k_regret"][str(k)]
        per_image.append(flat)
    if len(per_image) != 184:
        raise RuntimeError("post-freeze tumor cohort must contain 184 images")
    subgroup_counts = {
        subgroup: sum(row["size_group"] == subgroup for row in per_image)
        for subgroup in ("small", "medium", "large")
    }
    if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"frozen subgroup cohort mismatch: {subgroup_counts}")
    if set(baseline_by_id) != {row["image_id"] for row in per_image}:
        raise ValueError("baseline per-image tumor cohort mismatch")

    ranking_summary = summarize_ranking_diagnostics(
        ranking_rows, subgroup_names=SUBGROUPS, top_k=TOP_K
    )
    subgroup_metrics: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, object]] = {}
    regret_improvements: dict[str, float] = {}
    for subgroup_index, subgroup in enumerate(SUBGROUPS):
        rows = [
            row
            for row in per_image
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        arm_dice = [float(row["dice"]) for row in rows]
        baseline_dice = [float(row["baseline_dice"]) for row in rows]
        arm_regret = [
            float(row["selected_to_oracle_regret"]) for row in rows
        ]
        baseline_regret = [
            float(row["baseline_oracle_best_single_dice"])
            - float(row["baseline_dice"])
            for row in rows
        ]
        count_miss = _spearman(
            [float(row["candidate_count"]) for row in rows],
            [float(row["complete_miss"]) for row in rows],
        )
        baseline_count_miss = _spearman(
            [float(row["baseline_candidate_count"]) for row in rows],
            [float(row["baseline_complete_miss"]) for row in rows],
        )
        subgroup_metrics[subgroup] = {
            "n": len(rows),
            "dice": float(np.mean(arm_dice)),
            "baseline_dice": float(np.mean(baseline_dice)),
            "candidate_oracle": float(
                np.mean([float(row["oracle_best_single_dice"]) for row in rows])
            ),
            "selected_to_oracle_regret": float(np.mean(arm_regret)),
            "baseline_selected_to_oracle_regret": float(
                np.mean(baseline_regret)
            ),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in rows)),
            "baseline_complete_misses": int(
                sum(int(row["baseline_complete_miss"]) for row in rows)
            ),
            "candidate_count_vs_miss_spearman": count_miss,
            "baseline_candidate_count_vs_miss_spearman": baseline_count_miss,
        }
        regret_improvements[subgroup] = float(
            np.mean(baseline_regret) - np.mean(arm_regret)
        )
        paired[subgroup] = _paired_group_bootstrap(
            arm_dice,
            baseline_dice,
            [str(row["group_id"]) for row in rows],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + subgroup_index,
        )

    labels = np.asarray([int(row["tumor"]) for row in arm_predictions])
    bag_probabilities = np.asarray(
        [float(row["bag_probability"]) for row in arm_predictions],
        dtype=np.float64,
    )
    image_auroc = float(roc_auc_score(labels, bag_probabilities))
    operational_checks = {
        subgroup: {
            "observed": subgroup_metrics[subgroup]["dice"],
            "minimum": OPERATIONAL_GOALS[subgroup],
            "pass": subgroup_metrics[subgroup]["dice"]
            >= OPERATIONAL_GOALS[subgroup],
        }
        for subgroup in SUBGROUPS
    }
    oracle_checks = {
        subgroup: {
            "observed": subgroup_metrics[subgroup]["candidate_oracle"],
            "minimum": OPERATIONAL_GOALS[subgroup],
            "pass": subgroup_metrics[subgroup]["candidate_oracle"]
            >= OPERATIONAL_GOALS[subgroup],
        }
        for subgroup in SUBGROUPS
    }
    improved_subgroups = [
        subgroup
        for subgroup in ("small", "medium", "large")
        if regret_improvements[subgroup] > 0.0
    ]
    arm_count_association = subgroup_metrics["overall"][
        "candidate_count_vs_miss_spearman"
    ]
    baseline_count_association = subgroup_metrics["overall"][
        "baseline_candidate_count_vs_miss_spearman"
    ]
    arm_count_magnitude = (
        0.0 if arm_count_association is None else abs(float(arm_count_association))
    )
    baseline_count_magnitude = (
        0.0
        if baseline_count_association is None
        else abs(float(baseline_count_association))
    )
    mechanism_checks = {
        "regret_reduced_in_at_least_two_tumor_subgroups": {
            "observed_subgroups": improved_subgroups,
            "improvements": regret_improvements,
            "minimum_count": 2,
            "pass": len(improved_subgroups) >= 2,
        },
        "overall_selected_dice_no_regression": {
            "delta": paired["overall"]["delta_arm_minus_baseline"],
            "minimum": 0.0,
            "pass": paired["overall"]["delta_arm_minus_baseline"] >= 0.0,
        },
        "absolute_count_miss_association_no_increase": {
            "arm": arm_count_magnitude,
            "baseline": baseline_count_magnitude,
            "pass": arm_count_magnitude <= baseline_count_magnitude + 1.0e-12,
        },
    }
    final_safety_checks = {
        "overall_ci95_low_above_zero": paired["overall"]["ci95"][0] > 0.0,
        "no_tumor_subgroup_mean_decrease": all(
            paired[subgroup]["delta_arm_minus_baseline"] >= 0.0
            for subgroup in ("small", "medium", "large")
        ),
        "no_complete_miss_increase": all(
            subgroup_metrics[subgroup]["complete_misses"]
            <= subgroup_metrics[subgroup]["baseline_complete_misses"]
            for subgroup in SUBGROUPS
        ),
        "image_auroc_at_least_0_75": image_auroc >= 0.75,
    }
    mechanism_pass = all(check["pass"] for check in mechanism_checks.values())
    operational_pass = (
        mechanism_pass
        and all(check["pass"] for check in operational_checks.values())
        and all(check["pass"] for check in oracle_checks.values())
        and all(final_safety_checks.values())
    )
    status = (
        "OPERATIONAL_PASS"
        if operational_pass
        else "MECHANISM_PASS"
        if mechanism_pass
        else "FAIL"
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary = {
        "arm_source_commit": args.expected_arm_source_commit,
        "arm_protocol_sha256": args.expected_arm_protocol_sha256,
        "cohort": {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            **subgroup_counts,
        },
        "image_level_auroc": image_auroc,
        "subgroups": subgroup_metrics,
        "ranking": ranking_summary,
        "complete_misses_included": True,
        "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    paired_payload = {
        "comparison": "selector arm minus accepted geometry-v3 baseline",
        "method": "paired complete-group bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed_family": args.bootstrap_seed,
        "metrics": {"dice": paired},
        "consumer_trained": False,
        "test_evaluated": False,
    }
    gate = {
        "gate_id": "mask_bag_selector_arm_gate_v1",
        "status": status,
        "mechanism_checks": mechanism_checks,
        "operational_goal_checks": operational_checks,
        "candidate_oracle_goal_checks": oracle_checks,
        "final_safety_checks": final_safety_checks,
        "consumer_authorized": operational_pass,
        "on_mechanism_pass_only": "retain as a candidate for predeclared complementary composition; do not train a consumer",
        "on_fail": "reject this mechanism and advance to the next finite selector row without changing the selector bottleneck",
        "consumer_trained": False,
        "test_evaluated": False,
    }
    output_payloads = {
        "summary.json": summary,
        "paired_comparison.json": paired_payload,
        "gate_decision.json": gate,
    }
    output_hashes: dict[str, str] = {"per_image.csv": sha256_file(per_image_path)}
    for name, payload in output_payloads.items():
        path = args.output_dir / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_hashes[name] = sha256_file(path)
    audit = {
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "arm_prediction_freeze_sha256": args.expected_arm_freeze_sha256,
        "arm_prediction_manifest_sha256": arm_freeze["prediction_manifest_sha256"],
        "candidate_score_manifest_sha256": args.expected_score_manifest_sha256,
        "candidate_logit_provenance": {
            "field": args.expected_candidate_logit_provenance_field,
            "value": args.expected_candidate_logit_provenance_value,
        },
        "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
        "baseline_prediction_manifest_sha256": baseline_freeze[
            "prediction_manifest_sha256"
        ],
        "baseline_per_image_sha256": args.expected_baseline_per_image_sha256,
        "validation_only_split_projection": {
            "rows": 371,
            "sha256": validation_projection_sha256,
            "locked_test_bytes_read": False,
        },
        "output_hashes": output_hashes,
        "cohort": summary["cohort"],
        "bootstrap_replicates": args.bootstrap_replicates,
        "complete_misses_included": True,
        "validation_gt_read_only_after_all_predictions_frozen_and_verified": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
