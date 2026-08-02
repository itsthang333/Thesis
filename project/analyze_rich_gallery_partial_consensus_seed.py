from __future__ import annotations

"""Diagnose whether the frozen rich gallery can supervise a robust consumer.

The rules in :data:`RULES` are fixed before validation polygons are opened.
They are not candidate selectors and cannot be reported as deployable Dice.
Their purpose is to measure foreground seed precision, lesion coverage and
false-negative leakage in putatively safe background pixels.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from merge_frozen_candidate_galleries import merge_payloads
from models.rich_gallery_g2_objective import average_percentile_rank, stable_select


RULES = (
    "baseline_hard",
    "source_union_hard",
    "source_majority_partial",
    "source_intersection_partial",
    "baseline_cross_source_partial",
    "top3_majority_partial",
)
SOURCES = ("layercam320", "classifier448", "external_saliency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--addition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown source {value!r}")


def _payload_index(root: Path) -> dict[str, Path]:
    paths = sorted(root.rglob("*.npz"))
    result = {path.stem: path for path in paths}
    if len(paths) != 371 or len(result) != 371:
        raise ValueError(f"expected exactly 371 validation payloads under {root}")
    return result


def _read_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _top_order(scores: np.ndarray, logits: np.ndarray) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: (float(scores[index]), float(logits[index]), -index),
        reverse=True,
    )


def build_fixed_rule_masks(
    masks: np.ndarray,
    source_names: np.ndarray,
    logits: np.ndarray,
    upstream: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Return frozen foreground/background rules and selection diagnostics."""

    values = np.asarray(masks, dtype=bool)
    source_names = np.asarray(source_names).astype(str)
    logits = np.asarray(logits, dtype=np.float64)
    upstream = np.asarray(upstream, dtype=np.float64)
    if values.ndim != 3 or len(values) != len(source_names):
        raise ValueError("candidate masks/sources are not aligned")
    fusion = 0.5 * (
        average_percentile_rank(logits) + average_percentile_rank(upstream)
    )
    baseline_local = stable_select(fusion, logits)
    source_top: list[int] = []
    for source in SOURCES:
        eligible = np.flatnonzero(source_names == source)
        if not len(eligible):
            raise ValueError(f"missing source {source}")
        chosen = stable_select(fusion[eligible], logits[eligible])
        source_top.append(int(eligible[chosen]))
    source_stack = values[np.asarray(source_top)]
    source_votes = source_stack.sum(axis=0)
    source_union = source_votes >= 1
    source_majority = source_votes >= 2
    source_intersection = source_votes == len(SOURCES)

    baseline_source = source_names[baseline_local]
    other = [
        source_stack[index]
        for index, source in enumerate(SOURCES)
        if source != baseline_source
    ]
    baseline_confirmed = values[baseline_local] & np.logical_or.reduce(other)

    top3 = _top_order(fusion, logits)[:3]
    top3_stack = values[np.asarray(top3)]
    top3_votes = top3_stack.sum(axis=0)
    top3_union = top3_votes >= 1

    rules = {
        "baseline_hard": (values[baseline_local], ~values[baseline_local]),
        "source_union_hard": (source_union, ~source_union),
        "source_majority_partial": (source_majority, ~source_union),
        "source_intersection_partial": (source_intersection, ~source_union),
        "baseline_cross_source_partial": (baseline_confirmed, ~source_union),
        "top3_majority_partial": (top3_votes >= 2, ~top3_union),
    }
    return rules, {
        "baseline_local_index": int(baseline_local),
        "baseline_source": str(baseline_source),
        "source_top_local_indices": [int(value) for value in source_top],
        "top3_local_indices": [int(value) for value in top3],
    }


def score_partial_labels(
    foreground: np.ndarray, background: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    fg = np.asarray(foreground, dtype=bool)
    bg = np.asarray(background, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if fg.shape != truth.shape or bg.shape != truth.shape or np.any(fg & bg):
        raise ValueError("partial-label masks are invalid")
    tp = int(np.logical_and(fg, truth).sum())
    fp = int(np.logical_and(fg, ~truth).sum())
    bg_true = int(np.logical_and(bg, ~truth).sum())
    bg_false = int(np.logical_and(bg, truth).sum())
    fg_count = int(fg.sum())
    target_count = int(truth.sum())
    union = int(np.logical_or(fg, truth).sum())
    labeled = fg | bg
    labeled_count = int(labeled.sum())
    pixels = int(truth.size)
    return {
        "tp": tp,
        "fp": fp,
        "bg_true": bg_true,
        "bg_false": bg_false,
        "fg_count": fg_count,
        "target_count": target_count,
        "labeled_count": labeled_count,
        "pixels": pixels,
        "precision": float(tp / fg_count) if fg_count else 0.0,
        "recall": float(tp / target_count),
        "dice": float(2 * tp / (fg_count + target_count)),
        "iou": float(tp / union),
        "background_target_leakage": float(bg_false / target_count),
        "labeled_fraction": float(labeled_count / pixels),
        "unknown_fraction": float(1.0 - labeled_count / pixels),
        "partial_accuracy": float((tp + bg_true) / labeled_count),
        "complete_miss": int(tp == 0),
    }


def summarize(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    values = list(rows)
    if not values:
        return {"n": 0}
    sums = {
        name: int(sum(int(row[name]) for row in values))
        for name in (
            "tp",
            "fp",
            "bg_true",
            "bg_false",
            "fg_count",
            "target_count",
            "labeled_count",
            "pixels",
        )
    }
    return {
        "n": len(values),
        "mean_dice": float(np.mean([float(row["dice"]) for row in values])),
        "mean_iou": float(np.mean([float(row["iou"]) for row in values])),
        "mean_precision": float(
            np.mean([float(row["precision"]) for row in values])
        ),
        "mean_recall": float(np.mean([float(row["recall"]) for row in values])),
        "micro_precision": float(sums["tp"] / max(1, sums["fg_count"])),
        "micro_recall": float(sums["tp"] / max(1, sums["target_count"])),
        "mean_background_target_leakage": float(
            np.mean([float(row["background_target_leakage"]) for row in values])
        ),
        "mean_labeled_fraction": float(
            np.mean([float(row["labeled_fraction"]) for row in values])
        ),
        "mean_unknown_fraction": float(
            np.mean([float(row["unknown_fraction"]) for row in values])
        ),
        "complete_misses": int(sum(int(row["complete_miss"]) for row in values)),
        "micro_counts": sums,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output directory must not already exist")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split hash mismatch")
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(split_rows) != 371:
        raise ValueError("validation cohort mismatch")
    expected_ids = {str(row["image_id"]) for row in split_rows}
    anchor_paths = _payload_index(args.anchor_root)
    addition_paths = _payload_index(args.addition_root)

    table = pd.read_csv(args.candidate_table)
    if len(table) != 32519 or table["image_id"].nunique() != 184:
        raise ValueError("candidate table cohort mismatch")
    if not set(table["image_id"]).issubset(expected_ids):
        raise ValueError("candidate table escaped validation split")

    # Spatial annotations are opened only after all fixed rules above exist.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    targets = {
        str(image_id): target[0].numpy() > 0.5
        for _image, target, image_id in dataset
    }
    if set(targets) != expected_ids:
        raise ValueError("dataset IDs do not match canonical validation")

    output_rows: list[dict[str, object]] = []
    maximum_area_error = 0.0
    baseline_selection_exact = 0
    for image_id, frame in table.groupby("image_id", sort=True):
        stem = Path(str(image_id)).stem
        anchor = _read_payload(anchor_paths[stem])
        addition = _read_payload(addition_paths[stem])
        merged, _stats = merge_payloads(
            anchor, addition, addition_namespace="classifier448"
        )
        frame = frame.sort_values("candidate_local_index").reset_index(drop=True)
        candidate_indices = frame["candidate_index"].to_numpy(dtype=np.int64)
        masks = merged["sam_masks"][candidate_indices].astype(bool)
        area_error = np.max(
            np.abs(
                masks.mean(axis=(1, 2))
                - frame["candidate_area_ratio"].to_numpy(dtype=np.float64)
            )
        )
        maximum_area_error = max(maximum_area_error, float(area_error))
        sources = frame["source"].map(canonical_source).to_numpy()
        rules, selections = build_fixed_rule_masks(
            masks,
            sources,
            frame["g1_logit"].to_numpy(dtype=np.float64),
            frame["upstream_score"].to_numpy(dtype=np.float64),
        )
        frozen_selected = np.flatnonzero(
            frame["is_baseline_selected"].to_numpy(dtype=np.int64) == 1
        )
        if len(frozen_selected) != 1:
            raise ValueError(f"baseline selection is not unique for {image_id}")
        baseline_selection_exact += int(
            int(frozen_selected[0]) == selections["baseline_local_index"]
        )
        target = targets[str(image_id)]
        subgroup = str(frame.iloc[0]["size_group"])
        baseline_miss = int(
            np.logical_and(rules["baseline_hard"][0], target).sum() == 0
        )
        for rule in RULES:
            metrics = score_partial_labels(*rules[rule], target)
            output_rows.append(
                {
                    "image_id": str(image_id),
                    "group_id": str(frame.iloc[0]["group_id"]),
                    "size_group": subgroup,
                    "rule": rule,
                    "baseline_complete_miss": baseline_miss,
                    **metrics,
                }
            )

    if baseline_selection_exact != 184 or maximum_area_error > 1e-12:
        raise ValueError("merged candidate order does not reproduce frozen evidence")
    summary: dict[str, object] = {
        "stage": "rich_gallery_partial_consensus_seed_diagnostic_v1",
        "rules_frozen_before_validation_gt": list(RULES),
        "candidate_table_sha256": sha256_file(args.candidate_table),
        "split_sha256": args.expected_split_sha256,
        "validation_images": 371,
        "tumor_images": 184,
        "subgroups": {"small": 94, "medium": 72, "large": 18},
        "overall": {},
        "by_subgroup": {},
        "baseline_miss_rescue": {},
        "test_images_read": 0,
        "test_evaluated": False,
    }
    for rule in RULES:
        selected = [row for row in output_rows if row["rule"] == rule]
        summary["overall"][rule] = summarize(selected)  # type: ignore[index]
        summary["by_subgroup"][rule] = {  # type: ignore[index]
            subgroup: summarize(
                [row for row in selected if row["size_group"] == subgroup]
            )
            for subgroup in ("small", "medium", "large")
        }
        misses = [row for row in selected if row["baseline_complete_miss"] == 1]
        summary["baseline_miss_rescue"][rule] = {  # type: ignore[index]
            "n": len(misses),
            "rescued_any_overlap": int(
                sum(int(row["complete_miss"]) == 0 for row in misses)
            ),
            "mean_recall": float(np.mean([float(row["recall"]) for row in misses])),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    summary_path = args.output_dir / "summary.json"
    _write_csv(per_image_path, output_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    audit = {
        "pass": True,
        "candidate_rows": len(table),
        "tumor_images": 184,
        "rules": list(RULES),
        "baseline_selection_exact": baseline_selection_exact,
        "maximum_candidate_area_error": maximum_area_error,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "validation_gt_opened_only_for_diagnostic": True,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
