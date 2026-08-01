"""Post-freeze GT evaluator for the trusted rich-gallery control and B2 arm."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations
from models.rich_gallery_bas_residual import canonical_source
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from pseudo.manifest import sha256_file


EXPERIMENT_ID = "EXP-20260801-codex-b2-rich-gallery-bas-residual-v1"
CONTROL_ARM = "g1_upstream_control"
SEMANTIC_ARM = "g1_upstream_bas_semantic"
TRUSTED_CONTROL = {
    "overall": 0.28872948670665205,
    "small": 0.15772329637374455,
    "medium": 0.4352293348209193,
    "large": 0.38687353265476676,
}
GOALS = {
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
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-pair-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--independent-no-gt-audit", type=Path, required=True)
    parser.add_argument("--expected-independent-no-gt-audit-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = int(np.logical_or(prediction, target).sum())
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("B2 rank input must be finite and nonempty")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / max(len(array) - 1, 1)


def rank_correlation(scores: np.ndarray, quality: np.ndarray) -> float:
    score_rank = average_percentile_rank(scores)
    quality_rank = average_percentile_rank(quality)
    if float(np.std(score_rank)) <= 1.0e-12 or float(np.std(quality_rank)) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(score_rank, quality_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("B2 score/quality rank correlation is non-finite")
    return value


def unpack_prediction_payload(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"packed_mask", "shape"}:
            raise ValueError("B2 prediction payload schema mismatch")
        shape = tuple(int(value) for value in payload["shape"].reshape(-1))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("B2 prediction shape is invalid")
        bits = np.unpackbits(
            np.asarray(payload["packed_mask"], dtype=np.uint8),
            bitorder="little",
        )
    required = int(np.prod(shape))
    if len(bits) < required:
        raise ValueError("B2 prediction payload is truncated")
    return bits[:required].reshape(shape).astype(bool)


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("absolute B2 prediction path")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError("B2 prediction path escapes its arm root")
    return resolved


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write empty B2 evaluation CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def paired_group_bootstrap(
    arm: list[float],
    baseline: list[float],
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if not (len(arm) == len(baseline) == len(groups)) or not arm:
        raise ValueError("B2 bootstrap inputs must be aligned and nonempty")
    delta = np.asarray(arm, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    by_group: dict[str, list[float]] = {}
    for index in range(len(delta)):
        by_group.setdefault(groups[index], []).append(float(delta[index]))
    unique = sorted(by_group)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.integers(0, len(unique), size=len(unique))
        values = [value for position in sampled for value in by_group[unique[position]]]
        bootstrap[index] = float(np.mean(values))
    return {
        "delta_semantic_minus_control": float(delta.mean()),
        "ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_images": len(delta),
        "n_groups": len(unique),
        "replicates": replicates,
        "seed": seed,
    }


def verify_prediction_pair(
    args: argparse.Namespace,
    validation_rows: list[dict[str, str]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, str]]],
    dict[str, Any],
]:
    if sha256_file(args.independent_no_gt_audit) != args.expected_independent_no_gt_audit_sha256:
        raise ValueError("B2 independent no-GT audit SHA-256 mismatch")
    no_gt = json.loads(args.independent_no_gt_audit.read_text(encoding="utf-8"))
    if (
        no_gt.get("audit_pass") is not True
        or no_gt.get("experiment_id") != EXPERIMENT_ID
        or no_gt.get("source_commit") != args.expected_source_commit
        or no_gt.get("protocol_sha256") != args.expected_protocol_sha256
        or no_gt.get("pair_freeze_sha256") != args.expected_pair_freeze_sha256
        or no_gt.get("physical_predictions_reproduced") != 742
        or no_gt.get("validation_gt_read") is not False
        or no_gt.get("consumer_trained") is not False
        or no_gt.get("test_images_read") != 0
        or no_gt.get("test_evaluated") is not False
    ):
        raise ValueError("B2 independent no-GT audit contract mismatch")

    pair_path = args.prediction_root / "prediction_pair_freeze.json"
    if sha256_file(pair_path) != args.expected_pair_freeze_sha256:
        raise ValueError("B2 pair-freeze SHA-256 mismatch")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("source_commit") != args.expected_source_commit
        or pair.get("protocol_sha256") != args.expected_protocol_sha256
        or pair.get("split_sha256") != args.expected_split_sha256
        or set(pair.get("arm_freezes", {})) != {CONTROL_ARM, SEMANTIC_ARM}
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("validation_gt_read") is not False
        or pair.get("consumer_trained") is not False
        or pair.get("test_images_read") != 0
        or pair.get("test_evaluated") is not False
    ):
        raise ValueError("B2 pair-freeze safety/provenance mismatch")

    expected = {row["image_id"] for row in validation_rows}
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for arm in (CONTROL_ARM, SEMANTIC_ARM):
        arm_root = args.prediction_root / arm
        freeze_path = arm_root / "prediction_freeze.json"
        if sha256_file(freeze_path) != pair["arm_freezes"][arm]:
            raise ValueError(f"B2 {arm} freeze hash mismatch")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        manifest_path = arm_root / "prediction_manifest.csv"
        if (
            freeze.get("arm") != arm
            or freeze.get("validation_predictions") != 371
            or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
            or freeze.get("validation_gt_read") is not False
            or sha256_file(manifest_path) != freeze.get("prediction_manifest_sha256")
        ):
            raise ValueError(f"B2 {arm} frozen prediction contract mismatch")
        rows = _load_csv(manifest_path)
        current = {row["image_id"]: row for row in rows}
        if len(rows) != 371 or len(current) != 371 or set(current) != expected:
            raise ValueError(f"B2 {arm} prediction cohort mismatch")
        for image_id, row in current.items():
            path = safe_path(arm_root, row["prediction_path"])
            if sha256_file(path) != row["prediction_sha256"]:
                raise ValueError(f"B2 prediction hash mismatch: {arm}/{image_id}")
            prediction = unpack_prediction_payload(path)
            if prediction.shape != (320, 320):
                raise ValueError(f"B2 prediction shape mismatch: {arm}/{image_id}")
        indexed[arm] = current
    return pair, indexed, no_gt


def _summarize(
    records: list[dict[str, object]],
    arm: str,
) -> dict[str, dict[str, object]]:
    selected = [row for row in records if row["arm"] == arm]
    result: dict[str, dict[str, object]] = {}
    for subgroup in ("overall", "small", "medium", "large"):
        rows = [
            row
            for row in selected
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        result[subgroup] = {
            "n": len(rows),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "iou": float(np.mean([row["iou"] for row in rows])),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in rows)),
            "precision": float(np.mean([row["precision"] for row in rows])),
            "recall": float(np.mean([row["recall"] for row in rows])),
            "selected_area_mean": float(
                np.mean([row["selected_area_ratio"] for row in rows])
            ),
            "selected_area_median": float(
                np.median([row["selected_area_ratio"] for row in rows])
            ),
            "selected_to_gt_area_ratio_median": float(
                np.median([row["selected_to_gt_area_ratio"] for row in rows])
            ),
            "gallery_oracle_dice": float(
                np.mean([row["gallery_oracle_dice"] for row in rows])
            ),
            "eligible_oracle_dice": float(
                np.mean([row["eligible_oracle_dice"] for row in rows])
            ),
            "selector_regret": float(
                np.mean([row["selector_regret"] for row in rows])
            ),
            "wrong_source_regret": float(
                np.mean([row["wrong_source_regret"] for row in rows])
            ),
            "within_selected_source_regret": float(
                np.mean([row["within_selected_source_regret"] for row in rows])
            ),
            "candidate_truncation_regret": float(
                np.mean([row["candidate_truncation_regret"] for row in rows])
            ),
            "eligible_oracle_rank_median": float(
                np.median([row["eligible_oracle_rank"] for row in rows])
            ),
            "eligible_oracle_rank_p90": float(
                np.quantile([row["eligible_oracle_rank"] for row in rows], 0.90)
            ),
            "score_quality_rank_correlation_mean": float(
                np.mean([row["score_quality_rank_correlation"] for row in rows])
            ),
            "selected_source_matches_eligible_oracle_fraction": float(
                np.mean(
                    [row["selected_source_matches_eligible_oracle"] for row in rows]
                )
            ),
            "topk_restricted_oracle_dice": {
                str(depth): float(
                    np.mean([row[f"top{depth}_oracle_dice"] for row in rows])
                )
                for depth in (1, 3, 5, 10, 20, 50)
            },
            "recoverable_complete_misses": {
                str(threshold): int(
                    sum(
                        int(row["complete_miss"])
                        and float(row["gallery_oracle_dice"]) >= threshold
                        for row in rows
                    )
                )
                for threshold in (0.1, 0.3, 0.5)
            },
        }
    return result


def mechanism_diagnostics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Describe only paired B2 changes; the trusted control is not re-researched."""

    by_arm = {
        arm: {
            str(row["image_id"]): row
            for row in records
            if row["arm"] == arm
        }
        for arm in (CONTROL_ARM, SEMANTIC_ARM)
    }
    if (
        len(by_arm[CONTROL_ARM]) != 184
        or set(by_arm[CONTROL_ARM]) != set(by_arm[SEMANTIC_ARM])
    ):
        raise ValueError("B2 mechanism diagnostics require 184 aligned tumor pairs")

    changed = 0
    improved = 0
    worsened = 0
    tied = 0
    recovered_misses = 0
    lost_hits = 0
    positive_mass = 0.0
    negative_mass = 0.0
    transitions: Counter[str] = Counter()
    area_ratios: dict[str, dict[str, list[float]]] = {
        subgroup: {CONTROL_ARM: [], SEMANTIC_ARM: []}
        for subgroup in ("overall", "small", "medium", "large")
    }
    for image_id in sorted(by_arm[CONTROL_ARM]):
        control = by_arm[CONTROL_ARM][image_id]
        semantic = by_arm[SEMANTIC_ARM][image_id]
        if (
            control["group_id"] != semantic["group_id"]
            or control["size_group"] != semantic["size_group"]
        ):
            raise ValueError(f"B2 paired metadata differs: {image_id}")
        delta = float(semantic["dice"]) - float(control["dice"])
        if delta > 1.0e-12:
            improved += 1
            positive_mass += delta
        elif delta < -1.0e-12:
            worsened += 1
            negative_mass += delta
        else:
            tied += 1
        recovered_misses += int(
            int(control["complete_miss"]) == 1
            and int(semantic["complete_miss"]) == 0
        )
        lost_hits += int(
            int(control["complete_miss"]) == 0
            and int(semantic["complete_miss"]) == 1
        )
        choice_changed = int(control["selected_candidate_index"]) != int(
            semantic["selected_candidate_index"]
        )
        changed += int(choice_changed)
        if choice_changed:
            transitions[
                f'{control["selected_source"]}->{semantic["selected_source"]}'
            ] += 1
        subgroup = str(control["size_group"])
        for arm, row in ((CONTROL_ARM, control), (SEMANTIC_ARM, semantic)):
            selected_area = float(row["selected_area_ratio"])
            gt_area = float(row["gt_area_ratio"])
            ratio = selected_area / gt_area
            area_ratios["overall"][arm].append(ratio)
            area_ratios[subgroup][arm].append(ratio)

    return {
        "changed_positive_choices": changed,
        "changed_positive_choice_fraction": changed / 184.0,
        "paired_images_improved": improved,
        "paired_images_worsened": worsened,
        "paired_images_tied": tied,
        "complete_misses_recovered": recovered_misses,
        "complete_hits_lost": lost_hits,
        "positive_dice_mass": positive_mass,
        "negative_dice_mass": negative_mass,
        "changed_choice_source_transitions": dict(sorted(transitions.items())),
        "median_selected_to_gt_area_ratio": {
            subgroup: {
                "control": float(np.median(values[CONTROL_ARM])),
                "semantic": float(np.median(values[SEMANTIC_ARM])),
            }
            for subgroup, values in area_ratios.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10000 or args.bootstrap_seed != 42:
        raise ValueError("B2 evaluator differs from frozen bootstrap contract")
    if args.output_dir.exists():
        raise FileExistsError("B2 evaluation output already exists")
    validation_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(validation_rows) != 371:
        raise ValueError("B2 validation split cohort mismatch")
    pair, predictions, no_gt = verify_prediction_pair(args, validation_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in validation_rows],
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("B2 evaluator requires the complete rich gallery")
    score_manifest_path = args.prediction_root / "candidate_score_manifest.csv"
    if sha256_file(score_manifest_path) != pair["candidate_score_manifest_sha256"]:
        raise ValueError("B2 candidate-score manifest changed after pair freeze")
    score_rows = {
        row["image_id"]: row for row in _load_csv(score_manifest_path)
    }
    if len(score_rows) != 371:
        raise ValueError("B2 candidate-score cohort mismatch")

    # Annotation boundary: pair freeze, all 742 physical masks and the
    # independent no-GT audit have passed before this import and dataset build.
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
        label = int(predictions[CONTROL_ARM][image_id]["tumor"])
        if label == 0:
            continue
        target = mask_tensor[0].numpy() > 0.5
        target_area = int(target.sum())
        subgroup = size_group(float(target.mean()))
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"B2 candidate payload changed after freeze: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            candidate_masks = payload["sam_masks"].astype(bool)
            source_names = np.asarray(payload["proposal_source_ids"]).reshape(-1)
        if len(candidate_masks) != len(source_names):
            raise ValueError(f"B2 candidate/source mismatch: {image_id}")
        candidate_quality = np.asarray(
            [dice(mask, target) for mask in candidate_masks], dtype=np.float64
        )
        gallery_oracle_index = int(candidate_quality.argmax())
        gallery_oracle_dice = float(candidate_quality[gallery_oracle_index])
        score_row = score_rows[image_id]
        score_path = safe_path(args.prediction_root, score_row["score_path"])
        if sha256_file(score_path) != score_row["score_sha256"]:
            raise ValueError(f"B2 candidate scores changed after freeze: {image_id}")
        with np.load(score_path, allow_pickle=False) as score_payload:
            candidate_indices = score_payload["candidate_indices"].astype(np.int64)
            arm_scores = {
                CONTROL_ARM: score_payload["baseline_rank"].astype(np.float64),
                SEMANTIC_ARM: score_payload["semantic_rank"].astype(np.float64),
            }
        if (
            not len(candidate_indices)
            or len(np.unique(candidate_indices)) != len(candidate_indices)
            or int(candidate_indices.min()) < 0
            or int(candidate_indices.max()) >= len(candidate_masks)
            or any(len(scores) != len(candidate_indices) for scores in arm_scores.values())
        ):
            raise ValueError(f"B2 candidate-score alignment mismatch: {image_id}")
        eligible_quality = candidate_quality[candidate_indices]
        eligible_sources = np.asarray(
            [canonical_source(source_names[index]) for index in candidate_indices]
        )
        eligible_oracle_local = int(eligible_quality.argmax())
        eligible_oracle_dice = float(eligible_quality[eligible_oracle_local])
        eligible_oracle_source = str(eligible_sources[eligible_oracle_local])
        for arm in (CONTROL_ARM, SEMANTIC_ARM):
            row = predictions[arm][image_id]
            arm_root = args.prediction_root / arm
            prediction = unpack_prediction_payload(
                safe_path(arm_root, row["prediction_path"])
            )
            selected_local = int(row["selected_local_index"])
            selected_index = int(row["selected_candidate_index"])
            if (
                selected_local < 0
                or selected_local >= len(candidate_indices)
                or selected_index != int(candidate_indices[selected_local])
                or not np.array_equal(prediction, candidate_masks[selected_index])
            ):
                raise ValueError(f"B2 evaluated choice differs from gallery: {arm}/{image_id}")
            selected_source = canonical_source(source_names[selected_index])
            source_quality = eligible_quality[eligible_sources == selected_source]
            selected_source_oracle = float(source_quality.max())
            scores = arm_scores[arm]
            order = np.argsort(-scores, kind="stable")
            oracle_rank = int(np.flatnonzero(order == eligible_oracle_local)[0]) + 1
            intersection = int(np.logical_and(prediction, target).sum())
            prediction_area = int(prediction.sum())
            selected_dice = dice(prediction, target)
            candidate_truncation_regret = gallery_oracle_dice - eligible_oracle_dice
            wrong_source_regret = eligible_oracle_dice - selected_source_oracle
            within_selected_source_regret = selected_source_oracle - selected_dice
            regret_components = (
                candidate_truncation_regret,
                wrong_source_regret,
                within_selected_source_regret,
            )
            if min(regret_components) < -1.0e-12 or not np.isclose(
                sum(regret_components),
                gallery_oracle_dice - selected_dice,
                rtol=0,
                atol=1.0e-12,
            ):
                raise ValueError(f"B2 selector-regret decomposition failed: {arm}/{image_id}")
            topk = {
                depth: float(eligible_quality[order[: min(depth, len(order))]].max())
                for depth in (1, 3, 5, 10, 20, 50)
            }
            per_image.append(
                {
                    "arm": arm,
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "size_group": subgroup,
                    "gt_area_ratio": float(target.mean()),
                    "dice": selected_dice,
                    "iou": iou(prediction, target),
                    "precision": float(intersection / max(1, prediction_area)),
                    "recall": float(intersection / max(1, target_area)),
                    "complete_miss": int(intersection == 0),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_to_gt_area_ratio": float(
                        prediction_area / max(1, target_area)
                    ),
                    "selected_source": selected_source,
                    "selected_candidate_index": selected_index,
                    "gallery_oracle_index": gallery_oracle_index,
                    "gallery_oracle_source": canonical_source(
                        source_names[gallery_oracle_index]
                    ),
                    "gallery_oracle_dice": gallery_oracle_dice,
                    "eligible_oracle_candidate_index": int(
                        candidate_indices[eligible_oracle_local]
                    ),
                    "eligible_oracle_source": eligible_oracle_source,
                    "eligible_oracle_dice": eligible_oracle_dice,
                    "selected_source_oracle_dice": selected_source_oracle,
                    "selector_regret": gallery_oracle_dice - selected_dice,
                    "candidate_truncation_regret": candidate_truncation_regret,
                    "wrong_source_regret": wrong_source_regret,
                    "within_selected_source_regret": within_selected_source_regret,
                    "eligible_oracle_rank": oracle_rank,
                    "score_quality_rank_correlation": rank_correlation(
                        scores, eligible_quality
                    ),
                    "selected_source_matches_eligible_oracle": int(
                        selected_source == eligible_oracle_source
                    ),
                    **{
                        f"top{depth}_oracle_dice": value
                        for depth, value in topk.items()
                    },
                }
            )
    if len(per_image) != 368:
        raise RuntimeError("B2 evaluation tumor-pair cohort mismatch")
    subgroup_counts = Counter(
        row["size_group"] for row in per_image if row["arm"] == CONTROL_ARM
    )
    if subgroup_counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"B2 evaluation subgroup mismatch: {subgroup_counts}")

    metrics = {
        arm: _summarize(per_image, arm) for arm in (CONTROL_ARM, SEMANTIC_ARM)
    }
    for subgroup, expected in TRUSTED_CONTROL.items():
        if abs(float(metrics[CONTROL_ARM][subgroup]["dice"]) - expected) > 1.0e-8:
            raise ValueError(
                f"trusted collaborator control drifted for {subgroup}; this is an integrity failure"
            )

    by_arm = {
        arm: {
            str(row["image_id"]): row
            for row in per_image
            if row["arm"] == arm
        }
        for arm in (CONTROL_ARM, SEMANTIC_ARM)
    }
    ordered_ids = sorted(by_arm[CONTROL_ARM])
    bootstrap = paired_group_bootstrap(
        [float(by_arm[SEMANTIC_ARM][image_id]["dice"]) for image_id in ordered_ids],
        [float(by_arm[CONTROL_ARM][image_id]["dice"]) for image_id in ordered_ids],
        [str(by_arm[CONTROL_ARM][image_id]["group_id"]) for image_id in ordered_ids],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    subgroup_deltas = {
        subgroup: float(metrics[SEMANTIC_ARM][subgroup]["dice"])
        - float(metrics[CONTROL_ARM][subgroup]["dice"])
        for subgroup in ("small", "medium", "large")
    }
    mechanism = mechanism_diagnostics(per_image)
    mechanism_pass = bool(
        float(bootstrap["ci95"][0]) > 0.0
        and all(delta >= 0.0 for delta in subgroup_deltas.values())
        and int(metrics[SEMANTIC_ARM]["overall"]["complete_misses"])
        <= int(metrics[CONTROL_ARM]["overall"]["complete_misses"])
    )
    goal_pass = bool(
        all(
            float(metrics[SEMANTIC_ARM][subgroup]["dice"]) >= target
            for subgroup, target in GOALS.items()
        )
    )
    result = {
        "stage": "rich_gallery_bas_semantic_b2_postfreeze_evaluation_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "pair_freeze_sha256": args.expected_pair_freeze_sha256,
        "independent_no_gt_audit_sha256": args.expected_independent_no_gt_audit_sha256,
        "cohort": {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            "small": 94,
            "medium": 72,
            "large": 18,
        },
        "actual_binary_mask_metrics": metrics,
        "paired_complete_group_bootstrap": bootstrap,
        "subgroup_deltas_semantic_minus_control": subgroup_deltas,
        "paired_mechanism_diagnostics": mechanism,
        "failure_analysis_contract": {
            "per_image_candidate_supply_selector_extent_and_rank_depth_preserved": True,
            "regret_decomposition": (
                "candidate_truncation + wrong_source + within_selected_source"
            ),
            "topk_depths": [1, 3, 5, 10, 20, 50],
            "recoverable_miss_thresholds": [0.1, 0.3, 0.5],
            "no_next_gpu_run_before_manual_failure_dossier_if_rejected": True,
        },
        "mechanism_promotion_gate_pass": mechanism_pass,
        "operational_goal_gate_pass": goal_pass,
        "consumer_authorized": bool(mechanism_pass and goal_pass),
        "trusted_control_used_as_integrity_anchor_not_retrained": True,
        "pair_physically_frozen_before_validation_gt": pair[
            "pair_physically_frozen_before_validation_gt"
        ],
        "independent_no_gt_audit_pass": no_gt["audit_pass"],
        "validation_gt_read_only_after_pair_freeze_and_no_gt_audit": True,
        "spatial_ground_truth_used_for_training": False,
        "complete_misses_included": True,
        "test_images_read": 0,
        "test_evaluated": False,
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "validation_candidate_manifest_sha256": candidate_audit["manifest_sha256"],
        "validation_candidate_pseudo_manifest_sha256": candidate_audit[
            "pseudo_manifest_sha256"
        ],
        "validation_candidate_summary_sha256": candidate_audit["summary_sha256"],
    }
    args.output_dir.mkdir(parents=True)
    per_image_path = args.output_dir / "per_image.csv"
    per_image_sha256 = _write_csv(per_image_path, per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation_freeze = {
        "stage": "rich_gallery_bas_semantic_b2_evaluation_freeze_v1",
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "pair_freeze_sha256": args.expected_pair_freeze_sha256,
        "independent_no_gt_audit_sha256": args.expected_independent_no_gt_audit_sha256,
        "per_image_sha256": per_image_sha256,
        "summary_sha256": sha256_file(summary_path),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "prediction_frozen_before_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "evaluation_freeze.json"
    freeze_path.write_text(
        json.dumps(evaluation_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**result, "evaluation_freeze_sha256": sha256_file(freeze_path)},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
