from __future__ import annotations

"""Evaluate frozen G4 ablation choices on validation spatial annotations."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import _decode_labelme_polygon_mask, resolve_btxrd_root
from evaluation.segmentation_metrics import (
    bootstrap_group_confidence_intervals,
    json_safe,
    paired_group_bootstrap_deltas,
    segmentation_metrics,
    summarize_segmentation_rows,
)
from frozen_io import load_split_rows_without_annotations, sha256_file
BASELINE_ARM = "E8__R7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--choice-root", type=Path, required=True)
    parser.add_argument("--expected-choice-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-grid", choices=("native", "320", "448"), default="native")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _resize(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape == shape:
        return np.asarray(mask, dtype=bool)
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0


def _size_group(native_area_ratio: float) -> str:
    if native_area_ratio < 0.01:
        return "small_lt_1pct"
    if native_area_ratio < 0.05:
        return "medium_1_to_5pct"
    return "large_ge_5pct"


def _candidate_dice(masks: np.ndarray, target_320: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    intersections = np.logical_and(masks, target_320[None]).sum(axis=(1, 2))
    denominators = masks.sum(axis=(1, 2)) + int(target_320.sum())
    return np.divide(
        2.0 * intersections,
        denominators,
        out=np.zeros(len(masks), dtype=np.float64),
        where=denominators > 0,
    )


def _oracle_from_dice(
    candidate_dice: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, int]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return 0.0, -1
    # Lower frozen candidate index resolves exact oracle ties deterministically.
    selected = max(indices.tolist(), key=lambda i: (float(candidate_dice[i]), -int(i)))
    return float(candidate_dice[selected]), int(selected)


def _summarize_arm(rows: list[dict[str, object]]) -> dict[str, object]:
    summary = summarize_segmentation_rows(rows)
    tumor = [row for row in rows if bool(row["gt_positive"])]
    subgroup = {}
    for label in ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct"):
        subgroup_rows = [row for row in tumor if row["native_size_group"] == label]
        subgroup[label] = summarize_segmentation_rows(subgroup_rows)
    summary.update(
        {
            "native_subgroups": subgroup,
            "candidate_oracle_dice_common320": float(
                np.mean([float(row["oracle_dice_common320"]) for row in tumor])
            ),
            "selected_dice_common320": float(
                np.mean([float(row["selected_dice_common320"]) for row in tumor])
            ),
            "selector_regret_common320": float(
                np.mean([float(row["selector_regret_common320"]) for row in tumor])
            ),
            "within_source_regret_common320": float(
                np.mean([float(row["within_source_regret_common320"]) for row in tumor])
            ),
            "cross_source_regret_common320": float(
                np.mean([float(row["cross_source_regret_common320"]) for row in tumor])
            ),
            "oracle_source_differs_from_selected_rate": float(
                np.mean([bool(row["oracle_source_differs_from_selected"]) for row in tumor])
            ),
            "candidate_recall_at_0_10": float(
                np.mean([float(row["oracle_dice_common320"]) >= 0.10 for row in tumor])
            ),
            "candidate_recall_at_0_30": float(
                np.mean([float(row["oracle_dice_common320"]) >= 0.30 for row in tumor])
            ),
            "candidate_recall_at_0_50": float(
                np.mean([float(row["oracle_dice_common320"]) >= 0.50 for row in tumor])
            ),
            "selected_source_counts": dict(
                sorted(Counter(str(row["selected_source"]) for row in tumor).items())
            ),
            "oracle_source_counts": dict(
                sorted(Counter(str(row["oracle_source"]) for row in tumor).items())
            ),
            "eligible_candidate_count_median": float(
                np.median([int(row["eligible_candidate_count"]) for row in tumor])
            ),
            "eligible_candidate_count_iqr": [
                float(np.percentile([int(row["eligible_candidate_count"]) for row in tumor], 25)),
                float(np.percentile([int(row["eligible_candidate_count"]) for row in tumor], 75)),
            ],
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    split_by_id = {str(row["image_id"]): row for row in split_rows}
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("G4 evaluator requires the canonical 371/184 validation cohort")

    freeze_path = args.choice_root / "g4_choice_freeze.json"
    if sha256_file(freeze_path) != args.expected_choice_freeze_sha256:
        raise ValueError("G4 choice freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "g4_offline_ablation_choice_freeze_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G4 choice freeze violates the validation-only boundary")
    choices_path = args.choice_root / "g4_choices.csv"
    if sha256_file(choices_path) != freeze["choices_sha256"]:
        raise ValueError("G4 choices changed after freezing")
    choices = _read_csv(choices_path)
    arms = [str(arm) for arm in freeze["arms"]]
    by_image: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in choices:
        image_id, arm = row["image_id"], row["arm"]
        if arm in by_image[image_id]:
            raise ValueError(f"duplicate G4 choice: {image_id}/{arm}")
        by_image[image_id][arm] = row
    if set(by_image) != set(split_by_id) or any(set(items) != set(arms) for items in by_image.values()):
        raise ValueError("G4 choice matrix is incomplete")

    # Integrity boundary: verify all candidate payloads before opening one polygon.
    candidate_paths: dict[str, Path] = {}
    for image_id in sorted(split_by_id):
        path = args.candidate_root / "candidate_diagnostics" / f"{Path(image_id).stem}.npz"
        expected = {row["candidate_payload_sha256"] for row in by_image[image_id].values()}
        if len(expected) != 1 or sha256_file(path) not in expected:
            raise ValueError(f"candidate payload hash differs: {image_id}")
        candidate_paths[image_id] = path

    # Annotation boundary: all choices and all candidate bytes are immutable above.
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    rows_by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    opened_annotations = 0
    for split_row in split_rows:
        image_id = str(split_row["image_id"])
        tumor = int(split_row["tumor"]) == 1
        image_path = btxrd_root / "images" / image_id
        with Image.open(image_path) as image_handle:
            native_width, native_height = image_handle.size
        native_shape = (native_height, native_width)
        if tumor:
            annotation_path = btxrd_root / "Annotations" / f"{Path(image_id).stem}.json"
            native_target = _decode_labelme_polygon_mask(
                annotation_path, height=native_height, width=native_width
            ).astype(bool)
            opened_annotations += 1
        else:
            native_target = np.zeros(native_shape, dtype=bool)
        native_area_ratio = float(native_target.mean())
        size_group = _size_group(native_area_ratio) if tumor else "normal"
        target_320 = _resize(native_target, (320, 320))
        primary_shape = native_shape if args.primary_grid == "native" else (int(args.primary_grid),) * 2
        primary_target = _resize(native_target, primary_shape)

        with np.load(candidate_paths[image_id], allow_pickle=False) as payload:
            masks = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
            prompt_modes = payload["prompt_modes"].astype(str)
            upstream = payload["selection_scores"].astype(np.float64)
            if not (len(masks) == len(sources) == len(prompt_modes) == len(upstream)):
                raise ValueError(f"candidate arrays differ: {image_id}")
            candidate_dice_320 = (
                _candidate_dice(masks, target_320)
                if tumor
                else np.zeros(len(masks), dtype=np.float64)
            )
            full_oracle_dice, full_oracle_index = (
                _oracle_from_dice(
                    candidate_dice_320, np.arange(len(masks), dtype=np.int64)
                )
                if tumor
                else (0.0, -1)
            )
            for arm in arms:
                choice = by_image[image_id][arm]
                selected_index = int(choice["selected_candidate_index"])
                eligible_text = str(choice["eligible_candidate_indices"]).strip()
                eligible = np.asarray(
                    [int(value) for value in eligible_text.split(";") if value != ""],
                    dtype=np.int64,
                )
                if len(eligible) != int(choice["eligible_candidate_count"]):
                    raise ValueError(f"eligible count differs: {image_id}/{arm}")
                if (
                    len(np.unique(eligible)) != len(eligible)
                    or (len(eligible) and (eligible.min() < 0 or eligible.max() >= len(masks)))
                    or (selected_index >= 0 and selected_index not in set(eligible.tolist()))
                ):
                    raise ValueError(f"invalid frozen eligibility: {image_id}/{arm}")
                if tumor and selected_index >= 0:
                    selected_320 = masks[selected_index]
                else:
                    selected_320 = np.zeros((320, 320), dtype=bool)
                primary_prediction = _resize(selected_320, primary_shape)
                metrics = segmentation_metrics(
                    primary_prediction,
                    primary_target,
                    compute_boundary=arm == BASELINE_ARM,
                )
                oracle_dice, oracle_index = (
                    _oracle_from_dice(candidate_dice_320, eligible)
                    if tumor
                    else (0.0, -1)
                )
                selected_source = (
                    str(sources[selected_index]) if selected_index >= 0 else "empty_no_eligible_candidate"
                )
                same_source = np.flatnonzero(sources == selected_source)
                same_source_oracle, _ = (
                    _oracle_from_dice(candidate_dice_320, same_source)
                    if tumor and len(same_source)
                    else (0.0, -1)
                )
                selected_dice_320 = (
                    float(candidate_dice_320[selected_index])
                    if tumor and selected_index >= 0
                    else 0.0
                )
                row = {
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "tumor": int(tumor),
                    "arm": arm,
                    "primary_grid": args.primary_grid,
                    "native_height": native_height,
                    "native_width": native_width,
                    "native_gt_area_ratio": native_area_ratio,
                    "native_size_group": size_group,
                    "selected_candidate_index": selected_index,
                    "selected_source": selected_source,
                    "eligible_candidate_count": len(eligible),
                    "selected_dice_common320": selected_dice_320,
                    "oracle_dice_common320": oracle_dice,
                    "oracle_candidate_index": oracle_index,
                    "oracle_source": str(sources[oracle_index]) if oracle_index >= 0 else "none",
                    "selector_regret_common320": oracle_dice - selected_dice_320,
                    "within_source_oracle_dice_common320": same_source_oracle,
                    "within_source_regret_common320": same_source_oracle - selected_dice_320,
                    "cross_source_regret_common320": oracle_dice - same_source_oracle,
                    "oracle_source_differs_from_selected": bool(
                        oracle_index >= 0 and str(sources[oracle_index]) != selected_source
                    ),
                    "full_gallery_oracle_dice_common320": full_oracle_dice,
                    "full_gallery_oracle_candidate_index": full_oracle_index,
                    **metrics,
                }
                rows_by_arm[arm].append(row)

    if opened_annotations != 184:
        raise ValueError(f"opened {opened_annotations} annotations instead of 184")
    all_rows = [row for arm in arms for row in rows_by_arm[arm]]
    per_image_path = args.output_dir / "per_image_all_arms.csv"
    per_image_sha = _write_csv(per_image_path, all_rows)

    summaries: dict[str, object] = {}
    baseline_tumor = [row for row in rows_by_arm[BASELINE_ARM] if bool(row["gt_positive"])]
    for arm in arms:
        arm_rows = rows_by_arm[arm]
        tumor_rows = [row for row in arm_rows if bool(row["gt_positive"])]
        summary = _summarize_arm(arm_rows)
        summary["group_bootstrap_ci95"] = bootstrap_group_confidence_intervals(
            arm_rows,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        summary["paired_delta_vs_R7"] = paired_group_bootstrap_deltas(
            baseline_tumor,
            tumor_rows,
            metrics=("dice", "iou", "precision", "recall"),
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        summaries[arm] = summary

    report = {
        "schema_version": 1,
        "study": "G4 replayable offline ablations E4/E5-partial/E6-selector/E8",
        "primary_grid": args.primary_grid,
        "primary_endpoint": "macro Dice over 184 validation tumor images",
        "oracle_grid": "common 320x320 candidate grid",
        "native_subgroup_definition": "native polygon area / native image area",
        "baseline_arm": BASELINE_ARM,
        "choice_freeze_sha256": args.expected_choice_freeze_sha256,
        "split_sha256": args.expected_split_sha256,
        "images": len(split_rows),
        "tumor_images": 184,
        "spatial_annotations_opened": opened_annotations,
        "test_images_read": 0,
        "test_evaluated": False,
        "summaries": summaries,
        "limitations": freeze.get("limitations"),
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "pass": True,
        "choices_frozen_before_annotations": True,
        "choice_freeze_sha256": args.expected_choice_freeze_sha256,
        "per_image_sha256": per_image_sha,
        "summary_sha256": sha256_file(report_path),
        "images": len(split_rows),
        "tumor_images": 184,
        "arms": len(arms),
        "per_image_rows": len(all_rows),
        "validation_annotations_opened": opened_annotations,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    audit_path = args.output_dir / "evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "evaluation_audit_sha256": sha256_file(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
