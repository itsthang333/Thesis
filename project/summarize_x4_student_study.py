from __future__ import annotations

"""Aggregate all matched X4 student seeds and the five frozen contrasts."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from evaluation.segmentation_metrics import (
    json_safe,
    paired_group_bootstrap_deltas,
    summarize_segmentation_rows,
)
from frozen_io import sha256_file
from x4_contract import STUDENT_ARMS, STUDENT_SEEDS, load_x4_protocol


SIZE_GROUPS = ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")
SUMMARY_METRICS = (
    "mean_tumor_dice",
    "mean_tumor_iou",
    "mean_tumor_precision",
    "mean_tumor_recall",
    "micro_dice",
    "micro_iou",
    "lesion_any_overlap_recall",
    "lesion_one_to_one_iou25_recall",
    "normal_false_positive_case_rate",
    "normal_median_pred_area_ratio",
    "tumor_zero_overlap_rate",
)
CONTRASTS = {
    "rich_gallery_vs_cam": ("cam", "rich_gallery"),
    "rich_gallery_vs_puzzlecam": ("puzzlecam", "rich_gallery"),
    "rich_gallery_vs_s2c": ("s2c", "rich_gallery"),
    "rich_gallery_vs_fully_supervised": ("fully_supervised", "rich_gallery"),
    "rich_gallery_student_vs_direct_rich_gallery": ("direct_rich_gallery", "rich_gallery"),
}


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def read_evaluation_rows(
    path: Path,
    *,
    expected_images: int = 371,
    expected_tumor_images: int = 184,
) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    numeric = (
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_px",
        "assd_px",
        "gt_area_ratio",
        "pred_area_ratio",
        "predicted_gt_area_ratio",
        "relative_area_difference",
    )
    integer = (
        "tp_pixels",
        "fp_pixels",
        "fn_pixels",
        "tn_pixels",
        "gt_lesions",
        "detected_lesions_any_overlap",
        "predicted_lesions",
        "matched_predicted_lesions_any_overlap",
        "lesion_tp_one_to_one_iou10",
        "lesion_tp_one_to_one_iou25",
        "lesion_tp_one_to_one_iou50",
    )
    for raw_row in raw:
        row: dict[str, object] = dict(raw_row)
        for key in numeric:
            text = str(raw_row.get(key, "")).strip()
            row[key] = float(text) if text else float("nan")
        for key in integer:
            text = str(raw_row.get(key, "")).strip()
            row[key] = int(float(text)) if text else 0
        if "gt_positive" in raw_row:
            row["gt_positive"] = _bool(raw_row["gt_positive"])
        elif "tumor" in raw_row:
            row["gt_positive"] = _bool(raw_row["tumor"])
        elif expected_images == expected_tumor_images:
            # A tumor-only reference table can omit a constant tumor column;
            # the caller has already bound the exact all-tumor cohort size.
            row["gt_positive"] = True
        else:
            row["gt_positive"] = int(row.get("gt_lesions", 0)) > 0
        if "predicted_positive" in raw_row:
            row["predicted_positive"] = _bool(raw_row["predicted_positive"])
        else:
            row["predicted_positive"] = int(row.get("tp_pixels", 0)) + int(
                row.get("fp_pixels", 0)
            ) > 0
        for key in ("empty_prediction", "zero_overlap"):
            if key in raw_row:
                row[key] = _bool(raw_row[key])
        rows.append(row)
    image_ids = [str(row.get("image_id", "")) for row in rows]
    if (
        len(rows) != expected_images
        or len(set(image_ids)) != expected_images
        or "" in image_ids
    ):
        raise ValueError(
            f"X4 per-image cohort is not exact {expected_images}: {path}"
        )
    if sum(bool(row["gt_positive"]) for row in rows) != expected_tumor_images:
        raise ValueError(
            f"X4 per-image tumor cohort is not exact {expected_tumor_images}: {path}"
        )
    return rows


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "sample_sd": float("nan"), "n": 0}
    return {
        "mean": mean(finite),
        "sample_sd": stdev(finite) if len(finite) > 1 else 0.0,
        "n": len(finite),
    }


def summarize_study(
    runs: dict[tuple[str, int], list[dict[str, object]]],
    direct_rows: list[dict[str, object]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    expected = {(arm, run_seed) for arm in STUDENT_ARMS for run_seed in STUDENT_SEEDS}
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(f"X4 run matrix differs; missing={missing}, extra={extra}")
    direct_tumor_cohort = {
        str(row["image_id"]) for row in direct_rows if bool(row["gt_positive"])
    }
    if len(direct_tumor_cohort) != 184:
        raise ValueError("X4 direct Rich-Gallery tumor cohort is not exact 184")
    if any(
        {
            str(row["image_id"])
            for row in rows
            if bool(row["gt_positive"])
        }
        != direct_tumor_cohort
        for rows in runs.values()
    ):
        raise ValueError("X4 student/direct tumor cohorts differ")

    per_run: dict[str, object] = {}
    arm_seed_summaries: dict[str, list[dict[str, object]]] = {arm: [] for arm in STUDENT_ARMS}
    for arm in STUDENT_ARMS:
        for run_seed in STUDENT_SEEDS:
            rows = runs[(arm, run_seed)]
            summary = summarize_segmentation_rows(rows)
            subgroups = {
                group: summarize_segmentation_rows(
                    [row for row in rows if str(row.get("native_size_group")) == group]
                )
                for group in SIZE_GROUPS
            }
            record = {"seed": run_seed, "summary": summary, "subgroups": subgroups}
            arm_seed_summaries[arm].append(record)
            per_run[f"{arm}:seed{run_seed}"] = record

    across_seeds: dict[str, object] = {}
    for arm, records in arm_seed_summaries.items():
        across_seeds[arm] = {
            "overall": {
                metric: _mean_sd([float(record["summary"][metric]) for record in records])
                for metric in SUMMARY_METRICS
            },
            "subgroups": {
                group: {
                    metric: _mean_sd(
                        [float(record["subgroups"][group][metric]) for record in records]
                    )
                    for metric in ("mean_tumor_dice", "mean_tumor_iou")
                }
                for group in SIZE_GROUPS
            },
        }

    paired: dict[str, object] = {}
    for name, (reference_arm, comparison_arm) in CONTRASTS.items():
        seed_reports = []
        for run_seed in STUDENT_SEEDS:
            reference = direct_rows if reference_arm == "direct_rich_gallery" else runs[
                (reference_arm, run_seed)
            ]
            comparison = runs[(comparison_arm, run_seed)]
            reference_tumor = [row for row in reference if bool(row["gt_positive"])]
            comparison_tumor = [row for row in comparison if bool(row["gt_positive"])]
            report = paired_group_bootstrap_deltas(
                reference_tumor,
                comparison_tumor,
                metrics=("dice", "iou", "precision", "recall"),
                iterations=iterations,
                seed=seed + run_seed,
            )
            bootstrap_seed = int(report.pop("seed"))
            seed_reports.append(
                {
                    **report,
                    "seed": run_seed,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
        paired[name] = {
            "delta_direction": f"{comparison_arm}_minus_{reference_arm}",
            "per_seed": seed_reports,
            "across_seed_point_delta": {
                metric: _mean_sd(
                    [
                        float(report["intervals"][metric]["point_delta"])
                        for report in seed_reports
                    ]
                )
                for metric in ("dice", "iou", "precision", "recall")
            },
        }
    return {
        "schema_version": 1,
        "study": "X4 matched student multi-seed aggregation",
        "student_arms": list(STUDENT_ARMS),
        "student_seeds": list(STUDENT_SEEDS),
        "images_per_run": 371,
        "tumor_images_per_run": 184,
        "normal_images_per_run": 187,
        "per_run": per_run,
        "across_seeds": across_seeds,
        "paired_contrasts": paired,
        "bootstrap_iterations": iterations,
        "bootstrap_seed_base": seed,
        "bootstrap_group_provenance": "heuristic; not patient/case identifiers",
        "test_images_read": 0,
        "test_evaluated": False,
    }


def parse_run(value: str) -> tuple[tuple[str, int], Path]:
    try:
        arm, seed_text, path_text = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run must be ARM:SEED:PER_IMAGE_CSV") from exc
    if arm not in STUDENT_ARMS or int(seed_text) not in STUDENT_SEEDS:
        raise argparse.ArgumentTypeError("run arm/seed differs from X4 contract")
    return (arm, int(seed_text)), Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--direct-rich-gallery-per-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, protocol_sha = load_x4_protocol(args.repo_root)
    run_paths = dict(args.run)
    if len(run_paths) != len(args.run):
        raise ValueError("duplicate X4 arm/seed run")
    runs = {key: read_evaluation_rows(path) for key, path in run_paths.items()}
    # The frozen Direct Rich-Gallery endpoint is defined only on the 184 tumor
    # images.  Student runs still contain all 371 validation images so their
    # normal-image specificity remains measurable.
    direct = read_evaluation_rows(
        args.direct_rich_gallery_per_image,
        expected_images=184,
        expected_tumor_images=184,
    )
    report = summarize_study(
        runs,
        direct,
        iterations=int(protocol["paired_bootstrap"]["iterations"]),
        seed=int(protocol["paired_bootstrap"]["seed"]),
    )
    report["x4_protocol_sha256"] = protocol_sha
    report["input_sha256"] = {
        f"{arm}:seed{run_seed}": sha256_file(path)
        for (arm, run_seed), path in sorted(run_paths.items())
    }
    report["direct_rich_gallery_per_image_sha256"] = sha256_file(
        args.direct_rich_gallery_per_image
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
