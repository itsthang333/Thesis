from __future__ import annotations

"""X4 X9 deterministic, non-exclusive segmentation error taxonomy."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


DIRECT_METHOD = "direct_rich_gallery"
FAILURE_CLASSES = (
    "candidate_supply_failure",
    "selector_choice_failure",
    "complete_miss",
    "over_segmentation",
    "under_segmentation",
    "wrong_anatomical_structure",
    "fragmented_mask",
    "missing_component",
    "normal_false_positive",
    "small_lesion_specific_failure",
)
PRIMARY_PRIORITY = (
    "normal_false_positive",
    "candidate_supply_failure",
    "selector_choice_failure",
    "small_lesion_specific_failure",
    "wrong_anatomical_structure",
    "missing_component",
    "complete_miss",
    "over_segmentation",
    "under_segmentation",
    "fragmented_mask",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def classify_failure(
    row: dict[str, object],
    *,
    candidate_context: dict[str, object] | None,
) -> dict[str, bool]:
    """Return all applicable X9 flags; categories deliberately may overlap."""

    gt_positive = as_bool(row.get("gt_positive", row.get("tumor", 0)))
    predicted_positive = as_bool(row.get("predicted_positive", False))
    dice = float(row.get("dice", 0.0))
    ratio_raw = row.get("predicted_gt_area_ratio", row.get("selected_gt_area_ratio", "nan"))
    try:
        area_ratio = float(ratio_raw)
    except (TypeError, ValueError):
        area_ratio = float("nan")
    gt_lesions = int(float(row.get("gt_lesions", 0) or 0))
    detected_lesions = int(float(row.get("detected_lesions_any_overlap", 0) or 0))
    predicted_lesions = int(float(row.get("predicted_lesions", 0) or 0))
    zero_overlap = as_bool(row.get("zero_overlap", row.get("complete_miss", False)))
    size_group = str(row.get("native_size_group", row.get("size_group", ""))).lower()

    supply_failure = False
    selector_failure = False
    if candidate_context is not None and gt_positive:
        oracle = float(candidate_context["full_gallery_oracle_dice_common320"])
        regret = float(candidate_context["selector_regret_common320"])
        supply_failure = oracle < 0.10
        selector_failure = oracle >= 0.30 and regret >= 0.20
    return {
        "candidate_supply_failure": supply_failure,
        "selector_choice_failure": selector_failure,
        "complete_miss": gt_positive and zero_overlap,
        "over_segmentation": gt_positive and area_ratio > 2.0,
        "under_segmentation": gt_positive and area_ratio < 0.5,
        "wrong_anatomical_structure": gt_positive and predicted_positive and zero_overlap,
        "fragmented_mask": gt_positive and predicted_lesions >= 2 and predicted_lesions > gt_lesions,
        "missing_component": gt_positive and gt_lesions >= 2 and detected_lesions < gt_lesions,
        "normal_false_positive": (not gt_positive) and predicted_positive,
        "small_lesion_specific_failure": (
            gt_positive and ("small" in size_group or "lt_1pct" in size_group) and dice < 0.10
        ),
    }


def primary_failure(flags: dict[str, bool]) -> str:
    return next((name for name in PRIMARY_PRIORITY if flags[name]), "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-per-image", type=Path, required=True)
    parser.add_argument("--expected-direct-per-image-sha256", required=True)
    parser.add_argument("--direct-arm", default="E5_exact__cap243")
    parser.add_argument(
        "--method-table",
        action="append",
        nargs=3,
        metavar=("METHOD", "CSV", "SHA256"),
        default=[],
        help="Optional X4 student/common-evaluator table; repeat per method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_method_rows(
    rows: list[dict[str, str]],
    *,
    method: str,
    expected_ids: set[str],
) -> dict[str, dict[str, str]]:
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != expected_ids:
        raise ValueError(f"X9 method cohort differs for {method}")
    return indexed


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.direct_per_image) != args.expected_direct_per_image_sha256:
        raise ValueError("X9 direct per-image SHA-256 mismatch")
    all_direct = read_csv(args.direct_per_image)
    direct_rows = [row for row in all_direct if row.get("arm") == args.direct_arm]
    direct = {row["image_id"]: row for row in direct_rows}
    if len(direct_rows) != 371 or len(direct) != 371:
        raise ValueError("X9 direct arm must contain canonical 371 validation rows")
    if sum(int(row["tumor"]) for row in direct_rows) != 184:
        raise ValueError("X9 direct arm must contain canonical 184 tumor images")
    method_tables: dict[str, dict[str, dict[str, str]]] = {DIRECT_METHOD: direct}
    method_hashes: dict[str, str] = {DIRECT_METHOD: args.expected_direct_per_image_sha256}
    for method, path_raw, expected_sha in args.method_table:
        if method == DIRECT_METHOD or method in method_tables:
            raise ValueError(f"duplicate/reserved X9 method: {method}")
        path = Path(path_raw)
        if sha256_file(path) != expected_sha:
            raise ValueError(f"X9 method SHA-256 mismatch: {method}")
        method_tables[method] = _validate_method_rows(
            read_csv(path), method=method, expected_ids=set(direct)
        )
        method_hashes[method] = expected_sha

    output_rows: list[dict[str, object]] = []
    for method, table in method_tables.items():
        for image_id in sorted(direct):
            row = table[image_id]
            context = direct[image_id] if method == DIRECT_METHOD else None
            flags = classify_failure(row, candidate_context=context)
            output_rows.append(
                {
                    "method": method,
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "tumor": direct[image_id]["tumor"],
                    "size_group": row.get(
                        "native_size_group", row.get("size_group", direct[image_id]["native_size_group"])
                    ),
                    "dice": row["dice"],
                    "candidate_source": direct[image_id]["selected_source"],
                    "candidate_oracle_dice_common320": direct[image_id][
                        "full_gallery_oracle_dice_common320"
                    ],
                    "selector_regret_common320": direct[image_id]["selector_regret_common320"],
                    "primary_failure": primary_failure(flags),
                    "all_failure_flags": ";".join(name for name in FAILURE_CLASSES if flags[name]),
                    **{name: int(flags[name]) for name in FAILURE_CLASSES},
                }
            )
    summaries: dict[str, object] = {}
    for method in method_tables:
        selected = [row for row in output_rows if row["method"] == method]
        tumor = [row for row in selected if int(row["tumor"])]
        normal = [row for row in selected if not int(row["tumor"])]
        summaries[method] = {
            "images": len(selected),
            "tumor_images": len(tumor),
            "normal_images": len(normal),
            "candidate_based_flags_applicable": method == DIRECT_METHOD,
            "nonexclusive_flag_counts": {
                name: int(sum(int(row[name]) for row in selected)) for name in FAILURE_CLASSES
            },
            "nonexclusive_flag_rates_tumor_or_normal_denominator": {
                name: (
                    float(sum(int(row[name]) for row in selected) / len(normal))
                    if name == "normal_false_positive" and normal
                    else float(sum(int(row[name]) for row in selected) / len(tumor))
                    if name != "normal_false_positive" and tumor
                    else 0.0
                )
                for name in FAILURE_CLASSES
            },
            "primary_failure_counts": dict(
                sorted(Counter(str(row["primary_failure"]) for row in selected).items())
            ),
        }
    report = {
        "schema_version": 1,
        "study": "X4 X9 deterministic error taxonomy",
        "direct_arm": args.direct_arm,
        "methods": list(method_tables),
        "failure_classes": list(FAILURE_CLASSES),
        "classification_mode": (
            "non-exclusive flags plus one deterministic primary label; overlapping phenotypes "
            "are retained rather than forced into mutually exclusive bins"
        ),
        "frozen_rules": {
            "candidate_supply_failure": "full-gallery oracle Dice < 0.10",
            "selector_choice_failure": "oracle Dice >= 0.30 and selector regret >= 0.20",
            "complete_miss": "tumor and zero overlap",
            "over_segmentation": "predicted/GT area ratio > 2.0",
            "under_segmentation": "predicted/GT area ratio < 0.5",
            "wrong_anatomical_structure": "non-empty tumor prediction with zero overlap",
            "fragmented_mask": "predicted components >=2 and greater than GT component count",
            "missing_component": "multifocal GT with fewer overlapping detected components",
            "normal_false_positive": "normal image with any predicted positive pixel",
            "small_lesion_specific_failure": "<1% lesion and Dice < 0.10",
        },
        "primary_priority": list(PRIMARY_PRIORITY),
        "summaries": summaries,
        "input_sha256": method_hashes,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True)
    assignments_path = args.output_dir / "assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report["assignments_sha256"] = sha256_file(assignments_path)
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "pass": True,
        "stage": "x4_error_taxonomy_audit_v1",
        "summary_sha256": sha256_file(report_path),
        "assignments_sha256": report["assignments_sha256"],
        "methods": list(method_tables),
        "images_per_method": 371,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
