from __future__ import annotations

"""X4 X11 risk-coverage analysis for frozen Rich-Gallery confidence scores."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


COVERAGES = (1.0, 0.8, 0.6, 0.4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with exact-tie handling."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    return float(np.sum(x_centered * y_centered) / denominator) if denominator else float("nan")


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-correct AUROC via the Mann-Whitney rank statistic."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum = float(ranks[labels == 1].sum())
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def risk_coverage_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (-float(row["confidence"]), str(row["image_id"])))
    output: list[dict[str, object]] = []
    for coverage in COVERAGES:
        count = min(len(ordered), max(1, int(math.ceil(coverage * len(ordered)))))
        kept = ordered[:count]
        output.append(
            {
                "coverage": coverage,
                "retained_images": count,
                "retained_fraction": count / len(ordered),
                "mean_dice": float(np.mean([float(row["dice"]) for row in kept])),
                "dice_lt_0_10_rate": float(
                    np.mean([float(row["dice"]) < 0.10 for row in kept])
                ),
                "complete_miss_rate": float(
                    np.mean([bool(row["complete_miss"]) for row in kept])
                ),
                "minimum_retained_confidence": float(kept[-1]["confidence"]),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--direct-per-image", type=Path, required=True)
    parser.add_argument("--expected-direct-per-image-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.selection_manifest) != args.expected_selection_manifest_sha256:
        raise ValueError("selection manifest SHA-256 mismatch")
    if sha256_file(args.direct_per_image) != args.expected_direct_per_image_sha256:
        raise ValueError("direct evaluation per-image SHA-256 mismatch")
    selection_rows = read_csv(args.selection_manifest)
    selection = {row["image_id"]: row for row in selection_rows}
    direct_rows = read_csv(args.direct_per_image)
    direct = {row["image_id"]: row for row in direct_rows}
    if len(selection_rows) != 371 or len(selection) != 371:
        raise ValueError("X11 requires the frozen 371-image selection manifest")
    if len(direct_rows) != 184 or len(direct) != 184 or not set(direct).issubset(selection):
        raise ValueError("X11 requires the canonical 184-tumor direct evaluation")
    joined: list[dict[str, object]] = []
    for image_id in sorted(direct):
        selected = selection[image_id]
        row = direct[image_id]
        if selected.get("tumor") != "1":
            raise ValueError(f"X11 direct row is not tumor-labelled: {image_id}")
        confidence = float(selected["selected_fused_rank"])
        dice = float(row["dice"])
        complete_miss = bool(int(row["complete_miss"]))
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid fused-rank confidence: {image_id}")
        joined.append(
            {
                "image_id": image_id,
                "group_id": row["group_id"],
                "size_group": row["size_group"],
                "candidate_source": selected["selected_source"],
                "confidence": confidence,
                "dice": dice,
                "dice_lt_0_10": int(dice < 0.10),
                "complete_miss": int(complete_miss),
            }
        )
    confidence = np.asarray([float(row["confidence"]) for row in joined])
    dice = np.asarray([float(row["dice"]) for row in joined])
    dice_failure = np.asarray([int(row["dice_lt_0_10"]) for row in joined])
    complete_miss = np.asarray([int(row["complete_miss"]) for row in joined])
    coverage = risk_coverage_rows(joined)
    report = {
        "schema_version": 1,
        "study": "X4 X11 Rich-Gallery risk-coverage",
        "confidence": "selected_fused_rank from the frozen G1/equal-rank choice",
        "failure_score": "1 - selected_fused_rank",
        "tumor_images": len(joined),
        "spearman_confidence_vs_dice": spearman_correlation(confidence, dice),
        "auroc_detect_dice_lt_0_10": binary_auroc(dice_failure, 1.0 - confidence),
        "auroc_detect_complete_miss": binary_auroc(complete_miss, 1.0 - confidence),
        "coverage_rule": (
            "retain ceil(coverage*n) images with highest confidence; exact ties use image_id"
        ),
        "coverage": coverage,
        "interpretation_boundary": (
            "This is a post-freeze validation diagnostic. It does not select candidates, "
            "change the deployed pipeline, or define a new routing rule."
        ),
        "selection_manifest_sha256": args.expected_selection_manifest_sha256,
        "direct_per_image_sha256": args.expected_direct_per_image_sha256,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    coverage_path = args.output_dir / "coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    report.update(
        {
            "per_image_sha256": sha256_file(per_image_path),
            "coverage_sha256": sha256_file(coverage_path),
        }
    )
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "pass": True,
        "stage": "x4_risk_coverage_audit_v1",
        "summary_sha256": sha256_file(report_path),
        "per_image_sha256": report["per_image_sha256"],
        "coverage_sha256": report["coverage_sha256"],
        "tumor_images": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
