from __future__ import annotations

"""Paired accuracy/resource summary for two independently audited G4 E3 arms."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluation.segmentation_metrics import paired_group_bootstrap_deltas


METRICS = ("dice", "iou", "precision", "recall")
SUBGROUPS = ("small", "medium", "large")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(root: Path) -> list[dict[str, object]]:
    path = root / "evaluation" / "per_image.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = dict(row)
            for metric in METRICS:
                parsed[metric] = float(row[metric])
            rows.append(parsed)
    if len(rows) != 184 or len({str(row["image_id"]) for row in rows}) != 184:
        raise ValueError("E3 paired comparison requires 184 unique tumor images")
    counts = {
        group: sum(str(row["size_group"]) == group for row in rows)
        for group in SUBGROUPS
    }
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise ValueError(f"E3 subgroup cohort differs: {counts}")
    return rows


def summarize(
    reference_root: Path,
    comparison_root: Path,
    reference_audit_path: Path,
    comparison_audit_path: Path,
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    reference_audit = read_json(reference_audit_path)
    comparison_audit = read_json(comparison_audit_path)
    if reference_audit.get("pass") is not True or comparison_audit.get("pass") is not True:
        raise ValueError("both E3 arms must pass independent audit")
    if reference_audit.get("test_evaluated") is not False or comparison_audit.get("test_evaluated") is not False:
        raise ValueError("E3 pair must remain validation-only")
    if int(reference_audit.get("test_images_read", -1)) != 0 or int(comparison_audit.get("test_images_read", -1)) != 0:
        raise ValueError("E3 pair read test images")

    reference_rows = read_rows(reference_root)
    comparison_rows = read_rows(comparison_root)
    paired = {
        "overall": paired_group_bootstrap_deltas(
            reference_rows,
            comparison_rows,
            metrics=METRICS,
            iterations=iterations,
            seed=seed,
        )
    }
    for offset, group in enumerate(SUBGROUPS, start=1):
        paired[group] = paired_group_bootstrap_deltas(
            [row for row in reference_rows if row["size_group"] == group],
            [row for row in comparison_rows if row["size_group"] == group],
            metrics=METRICS,
            iterations=iterations,
            seed=seed + offset,
        )

    ref_summary = reference_audit["summary"]
    cmp_summary = comparison_audit["summary"]
    ref_resources = reference_audit["resource_metrics"]
    cmp_resources = comparison_audit["resource_metrics"]
    selected_source_changed = sum(
        str(ref["source"]) != str(cmp["source"])
        for ref, cmp in zip(
            sorted(reference_rows, key=lambda row: str(row["image_id"])),
            sorted(comparison_rows, key=lambda row: str(row["image_id"])),
        )
    )
    return {
        "schema_version": 1,
        "study": "G4 E3 paired SAM backbone accuracy-resource comparison",
        "delta_direction": "comparison_minus_reference",
        "reference_model": reference_audit["sam_model_type"],
        "comparison_model": comparison_audit["sam_model_type"],
        "reference_audit_sha256": sha256(reference_audit_path),
        "comparison_audit_sha256": sha256(comparison_audit_path),
        "reference_summary": ref_summary,
        "comparison_summary": cmp_summary,
        "paired_group_bootstrap": paired,
        "selected_source_changed_images": selected_source_changed,
        "resource_ratio_comparison_over_reference": {
            key: float(cmp_resources[key]) / float(ref_resources[key])
            for key in (
                "candidate_generation_elapsed_seconds",
                "peak_memory_allocated_bytes",
                "peak_memory_reserved_bytes",
                "merged_gallery_bytes",
                "total_arm_elapsed_seconds",
            )
        },
        "uncertainty_note": (
            "Paired nonparametric bootstrap over complete heuristic groups; "
            "group_id is not a verified patient/case identifier."
        ),
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--comparison-audit", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("paired E3 output already exists")
    report = summarize(
        args.reference_root,
        args.comparison_root,
        args.reference_audit,
        args.comparison_audit,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_sha256": sha256(args.output),
        "overall": report["paired_group_bootstrap"]["overall"]["intervals"],
        "resource_ratio": report["resource_ratio_comparison_over_reference"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
