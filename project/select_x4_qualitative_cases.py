from __future__ import annotations

"""Freeze X4 X10 qualitative cases using deterministic metric rules."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CATEGORIES = (
    "dice_q10",
    "dice_q50",
    "dice_q90",
    "small_representative",
    "medium_representative",
    "large_representative",
    "over_segmentation",
    "under_segmentation",
    "complete_miss",
    "wrong_location",
    "normal_false_positive",
    "multifocal",
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


def choose_closest(
    rows: list[dict[str, object]],
    *,
    field: str,
    target: float,
    used: set[str],
) -> dict[str, object] | None:
    if not rows:
        return None
    ranked = sorted(
        rows,
        key=lambda row: (
            str(row["image_id"]) in used,
            abs(float(row[field]) - target),
            str(row["image_id"]),
        ),
    )
    return ranked[0]


def freeze_case_rows(
    direct_rows: list[dict[str, object]],
    taxonomy_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    direct = [row for row in direct_rows if int(row["tumor"]) == 1]
    taxonomy_by_method: dict[str, list[dict[str, object]]] = {}
    for row in taxonomy_rows:
        taxonomy_by_method.setdefault(str(row["method"]), []).append(row)
    direct_taxonomy = {
        str(row["image_id"]): row
        for row in taxonomy_by_method.get("direct_rich_gallery", [])
    }
    if len(direct) != 184 or set(direct_taxonomy) != {str(row["image_id"]) for row in direct_rows}:
        raise ValueError("X10 direct/taxonomy cohorts differ")
    used: set[str] = set()
    output: list[dict[str, object]] = []

    def add(category: str, row: dict[str, object] | None, reason: str) -> None:
        if row is None:
            output.append(
                {
                    "category": category,
                    "available": 0,
                    "image_id": "",
                    "method": "",
                    "selection_reason": reason,
                }
            )
            return
        image_id = str(row["image_id"])
        used.add(image_id)
        output.append(
            {
                "category": category,
                "available": 1,
                "image_id": image_id,
                "method": str(row.get("method", "direct_rich_gallery")),
                "selection_reason": reason,
            }
        )

    dice_values = np.asarray([float(row["dice"]) for row in direct])
    for quantile, category in ((0.10, "dice_q10"), (0.50, "dice_q50"), (0.90, "dice_q90")):
        target = float(np.quantile(dice_values, quantile))
        add(
            category,
            choose_closest(direct, field="dice", target=target, used=used),
            f"closest direct Dice to frozen q={quantile:.2f} value {target:.12g}",
        )
    for token, category in (
        ("small", "small_representative"),
        ("medium", "medium_representative"),
        ("large", "large_representative"),
    ):
        candidates = [row for row in direct if token in str(row["native_size_group"]).lower()]
        target = float(np.median([float(row["dice"]) for row in candidates]))
        add(
            category,
            choose_closest(candidates, field="dice", target=target, used=used),
            f"closest Dice to the frozen {token} subgroup median {target:.12g}",
        )

    direct_by_id = {str(row["image_id"]): row for row in direct_rows}
    for flag, category, field in (
        ("over_segmentation", "over_segmentation", "predicted_gt_area_ratio"),
        ("under_segmentation", "under_segmentation", "predicted_gt_area_ratio"),
        ("complete_miss", "complete_miss", "full_gallery_oracle_dice_common320"),
        ("wrong_anatomical_structure", "wrong_location", "predicted_gt_area_ratio"),
    ):
        candidates = [
            direct_by_id[image_id]
            for image_id, taxonomy in direct_taxonomy.items()
            if int(taxonomy[flag]) == 1
        ]
        target = float(np.median([float(row[field]) for row in candidates])) if candidates else 0.0
        add(
            category,
            choose_closest(candidates, field=field, target=target, used=used),
            f"closest {field} to the median among frozen {flag} cases",
        )

    normal_fp_candidates = [
        row
        for method, rows in sorted(taxonomy_by_method.items())
        for row in rows
        if int(row["normal_false_positive"]) == 1
    ]
    normal_fp = sorted(
        normal_fp_candidates,
        key=lambda row: (str(row["image_id"]) in used, str(row["method"]), str(row["image_id"])),
    )
    add(
        "normal_false_positive",
        normal_fp[0] if normal_fp else None,
        "first deterministic method/image normal FP after method then image_id sort",
    )
    multifocal = [row for row in direct if int(float(row["gt_lesions"])) > 1]
    target = float(np.median([float(row["dice"]) for row in multifocal])) if multifocal else 0.0
    add(
        "multifocal",
        choose_closest(multifocal, field="dice", target=target, used=used),
        f"closest Dice to multifocal median {target:.12g}",
    )
    if tuple(row["category"] for row in output) != CATEGORIES:
        raise RuntimeError("X10 category order differs")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-per-image", type=Path, required=True)
    parser.add_argument("--expected-direct-per-image-sha256", required=True)
    parser.add_argument("--direct-arm", default="E5_exact__cap243")
    parser.add_argument("--taxonomy-assignments", type=Path, required=True)
    parser.add_argument("--expected-taxonomy-assignments-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.direct_per_image) != args.expected_direct_per_image_sha256:
        raise ValueError("X10 direct per-image SHA-256 mismatch")
    if sha256_file(args.taxonomy_assignments) != args.expected_taxonomy_assignments_sha256:
        raise ValueError("X10 taxonomy assignments SHA-256 mismatch")
    all_direct = read_csv(args.direct_per_image)
    direct = [row for row in all_direct if row.get("arm") == args.direct_arm]
    if len(direct) != 371:
        raise ValueError("X10 direct arm must have 371 rows")
    taxonomy = read_csv(args.taxonomy_assignments)
    selections = freeze_case_rows(direct, taxonomy)
    args.output_dir.mkdir(parents=True)
    selection_path = args.output_dir / "selection_manifest.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selections[0]))
        writer.writeheader()
        writer.writerows(selections)
    report = {
        "schema_version": 1,
        "stage": "x4_qualitative_case_freeze_v1",
        "direct_arm": args.direct_arm,
        "categories": list(CATEGORIES),
        "available_categories": int(sum(int(row["available"]) for row in selections)),
        "selection_manifest_sha256": sha256_file(selection_path),
        "direct_per_image_sha256": args.expected_direct_per_image_sha256,
        "taxonomy_assignments_sha256": args.expected_taxonomy_assignments_sha256,
        "selection_before_image_or_gt_rendering": True,
        "selection_by_visual_appeal": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "selection_freeze.json"
    freeze_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "selection_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
