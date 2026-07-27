from __future__ import annotations

"""Compare two already evaluated mask-bag arms without reopening GT masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import sha256_file, validate_sha256


SUBGROUPS = ("overall", "small", "medium", "large")
EXPECTED_COUNTS = {"small": 94, "medium": 72, "large": 18}
REQUIRED_FIELDS = {
    "image_id",
    "group_id",
    "gt_area_ratio",
    "size_group",
    "dice",
    "oracle_best_single_dice",
    "complete_miss",
    "selected_area_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-per-image", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-per-image", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261101)
    return parser.parse_args()


def _read(path: Path, expected_sha256: str) -> dict[str, dict[str, str]]:
    expected = validate_sha256(expected_sha256, name=f"{path.name} SHA-256")
    if sha256_file(path) != expected:
        raise ValueError(f"Per-image SHA-256 mismatch: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 184 or not rows:
        raise ValueError("Each evaluated arm must contain exactly 184 tumor rows")
    missing = sorted(REQUIRED_FIELDS - set(rows[0]))
    if missing:
        raise ValueError(f"Per-image evaluation lacks fields: {missing}")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["image_id"]
        if not image_id or image_id in indexed:
            raise ValueError("Per-image evaluation contains duplicate/empty image IDs")
        if row["size_group"] not in EXPECTED_COUNTS:
            raise ValueError("Per-image evaluation contains an invalid subgroup")
        if row["complete_miss"] not in {"0", "1"}:
            raise ValueError("Per-image evaluation contains an invalid miss flag")
        for field in ("gt_area_ratio", "dice", "oracle_best_single_dice", "selected_area_ratio"):
            value = float(row[field])
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"Invalid {field} for {image_id}")
        indexed[image_id] = row
    counts = {
        subgroup: sum(row["size_group"] == subgroup for row in rows)
        for subgroup in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Frozen subgroup counts differ: {counts}")
    return indexed


def _paired_group_bootstrap(
    deltas: np.ndarray,
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for value, group in zip(deltas, groups, strict=True):
        grouped.setdefault(group, []).append(float(value))
    unique = sorted(grouped)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.integers(0, len(unique), size=len(unique))
        rows = [value for position in sampled for value in grouped[unique[position]]]
        bootstrap[index] = float(np.mean(rows))
    return {
        "delta_candidate_minus_reference": float(deltas.mean()),
        "ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_images": int(deltas.size),
        "n_groups": len(unique),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10000 or args.bootstrap_seed != 20261101:
        raise ValueError("Comparator requires the frozen 10,000/20261101 bootstrap")
    if not args.candidate_name.strip() or not args.reference_name.strip():
        raise ValueError("Both arm names must be nonempty")
    candidate = _read(args.candidate_per_image, args.expected_candidate_sha256)
    reference = _read(args.reference_per_image, args.expected_reference_sha256)
    if set(candidate) != set(reference):
        raise ValueError("Evaluated arm image identities differ")

    paired_rows: list[dict[str, object]] = []
    for image_id in sorted(candidate):
        new = candidate[image_id]
        old = reference[image_id]
        for field in ("group_id", "gt_area_ratio", "size_group", "oracle_best_single_dice"):
            if new[field] != old[field]:
                raise ValueError(f"Frozen paired field {field} differs for {image_id}")
        new_dice = float(new["dice"])
        old_dice = float(old["dice"])
        new_miss = int(new["complete_miss"])
        old_miss = int(old["complete_miss"])
        paired_rows.append(
            {
                "image_id": image_id,
                "group_id": new["group_id"],
                "size_group": new["size_group"],
                "candidate_dice": new_dice,
                "reference_dice": old_dice,
                "delta_dice": new_dice - old_dice,
                "candidate_complete_miss": new_miss,
                "reference_complete_miss": old_miss,
                "miss_recovered": int(old_miss == 1 and new_miss == 0),
                "overlap_lost": int(old_miss == 0 and new_miss == 1),
            }
        )

    metrics: dict[str, object] = {}
    for index, subgroup in enumerate(SUBGROUPS):
        rows = [
            row
            for row in paired_rows
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        deltas = np.asarray([float(row["delta_dice"]) for row in rows])
        candidate_values = np.asarray([float(row["candidate_dice"]) for row in rows])
        reference_values = np.asarray([float(row["reference_dice"]) for row in rows])
        paired = _paired_group_bootstrap(
            deltas,
            [str(row["group_id"]) for row in rows],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
        )
        metrics[subgroup] = {
            **paired,
            "candidate_dice": float(candidate_values.mean()),
            "reference_dice": float(reference_values.mean()),
            "candidate_complete_misses": int(
                sum(int(row["candidate_complete_miss"]) for row in rows)
            ),
            "reference_complete_misses": int(
                sum(int(row["reference_complete_miss"]) for row in rows)
            ),
            "misses_recovered": int(sum(int(row["miss_recovered"]) for row in rows)),
            "overlaps_lost": int(sum(int(row["overlap_lost"]) for row in rows)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = args.output_dir / "paired_per_image.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(paired_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(paired_rows)
    payload = {
        "comparison": f"{args.candidate_name} minus {args.reference_name}",
        "candidate_per_image_sha256": args.expected_candidate_sha256,
        "reference_per_image_sha256": args.expected_reference_sha256,
        "paired_per_image_sha256": sha256_file(rows_path),
        "method": "paired complete-group bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed_family": args.bootstrap_seed,
        "cohort": {"tumor": 184, **EXPECTED_COUNTS},
        "complete_misses_included": True,
        "metrics": metrics,
        "ground_truth_reopened": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    output_path = args.output_dir / "paired_comparison.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
