from __future__ import annotations

"""Resume G4 summary/bootstrap from a hash-locked completed per-image table."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import shutil

from evaluation.segmentation_metrics import (
    bootstrap_group_confidence_intervals,
    json_safe,
    paired_group_bootstrap_deltas,
)
from evaluate_g4_offline_ablations import _summarize_arm
from frozen_io import load_split_rows_without_annotations, sha256_file


INTEGER = re.compile(r"^-?[0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--choice-root", type=Path, required=True)
    parser.add_argument("--expected-choice-freeze-sha256", required=True)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--expected-per-image-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def coerce_csv_value(value: str) -> object:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return ""
    if INTEGER.fullmatch(value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _coerce_row(row: dict[str, str]) -> dict[str, object]:
    identifiers = {
        "image_id",
        "group_id",
        "arm",
        "primary_grid",
        "native_size_group",
        "selected_source",
        "oracle_source",
    }
    return {
        key: value if key in identifiers else coerce_csv_value(value)
        for key, value in row.items()
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.per_image) != args.expected_per_image_sha256:
        raise ValueError("resume per-image SHA-256 mismatch")
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    split_by_id = {str(row["image_id"]): row for row in split_rows}
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("resume requires canonical 371/184 validation")

    freeze_path = args.choice_root / "g4_choice_freeze.json"
    if sha256_file(freeze_path) != args.expected_choice_freeze_sha256:
        raise ValueError("resume choice freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "g4_e5_exact_choice_freeze_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("resume choice freeze violates E5 boundary")
    choices_path = args.choice_root / "g4_choices.csv"
    if sha256_file(choices_path) != freeze["choices_sha256"]:
        raise ValueError("resume choices changed")
    choice_rows = _read_csv(choices_path)
    choice_by_key = {(row["image_id"], row["arm"]): row for row in choice_rows}
    arms = [str(arm) for arm in freeze["arms"]]
    baseline_arm = str(freeze["baseline_arm"])

    rows = [_coerce_row(row) for row in _read_csv(args.per_image)]
    expected_keys = {(image_id, arm) for image_id in split_by_id for arm in arms}
    by_key = {(str(row["image_id"]), str(row["arm"])): row for row in rows}
    if len(rows) != len(by_key) or set(by_key) != expected_keys or set(choice_by_key) != expected_keys:
        raise ValueError("resume per-image arm matrix differs")
    rows_by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(by_key):
        image_id, arm = key
        row = by_key[key]
        split_row = split_by_id[image_id]
        if (
            str(row["group_id"]) != split_row["group_id"]
            or bool(row["gt_positive"]) != bool(int(split_row["tumor"]))
            or str(row["primary_grid"]) != "native"
            or int(row["selected_candidate_index"])
            != int(choice_by_key[key]["selected_candidate_index"])
        ):
            raise ValueError(f"resume per-image provenance differs: {image_id}/{arm}")
        rows_by_arm[arm].append(row)
    for arm in arms:
        arm_rows = rows_by_arm[arm]
        if len(arm_rows) != 371 or sum(bool(row["gt_positive"]) for row in arm_rows) != 184:
            raise ValueError(f"resume arm cohort differs: {arm}")
        subgroup = {
            label: sum(row["native_size_group"] == label for row in arm_rows)
            for label in ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")
        }
        if subgroup != {
            "small_lt_1pct": 94,
            "medium_1_to_5pct": 72,
            "large_ge_5pct": 18,
        }:
            raise ValueError(f"resume subgroup cohort differs: {arm}/{subgroup}")

    args.output_dir.mkdir(parents=True)
    per_image_path = args.output_dir / "per_image_all_arms.csv"
    shutil.copy2(args.per_image, per_image_path)
    summaries: dict[str, object] = {}
    baseline_tumor = [row for row in rows_by_arm[baseline_arm] if bool(row["gt_positive"])]
    for arm in arms:
        arm_rows = rows_by_arm[arm]
        tumor_rows = [row for row in arm_rows if bool(row["gt_positive"])]
        summary = _summarize_arm(arm_rows)
        summary["group_bootstrap_ci95"] = bootstrap_group_confidence_intervals(
            arm_rows,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        summary["paired_delta_vs_baseline"] = paired_group_bootstrap_deltas(
            baseline_tumor,
            tumor_rows,
            metrics=("dice", "iou", "precision", "recall"),
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        summaries[arm] = summary

    report = {
        "schema_version": 1,
        "study": freeze["study"],
        "primary_grid": "native",
        "primary_endpoint": "macro Dice over 184 validation tumor images",
        "oracle_grid": "common 320x320 candidate grid",
        "native_subgroup_definition": "native polygon area / native image area",
        "baseline_arm": baseline_arm,
        "choice_freeze_sha256": args.expected_choice_freeze_sha256,
        "split_sha256": args.expected_split_sha256,
        "images": 371,
        "tumor_images": 184,
        "spatial_annotations_opened": 184,
        "resume_source_per_image_sha256": args.expected_per_image_sha256,
        "annotations_opened_in_resume_process": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "summaries": summaries,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "pass": True,
        "choices_frozen_before_annotations": True,
        "choice_freeze_sha256": args.expected_choice_freeze_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(report_path),
        "resume_source_per_image_sha256": args.expected_per_image_sha256,
        "images": 371,
        "tumor_images": 184,
        "arms": len(arms),
        "per_image_rows": len(rows),
        "validation_annotations_opened": 184,
        "annotations_opened_in_resume_process": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    audit_path = args.output_dir / "evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "evaluation_audit_sha256": sha256_file(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
