from __future__ import annotations

"""Matched three-seed binary-versus-ten-class downstream WSSS comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from evaluation.segmentation_metrics import paired_group_bootstrap_deltas


SEEDS = (42, 43, 44)
METRICS = ("dice", "iou", "precision", "recall")
SUBGROUPS = ("small", "medium", "large")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(root: Path, seed: int) -> list[dict[str, object]]:
    path = root / f"seed_{seed}" / "evaluation" / "per_image.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = dict(row)
            for metric in METRICS:
                parsed[metric] = float(row[metric])
            rows.append(parsed)
    if len(rows) != 184 or len({str(row["image_id"]) for row in rows}) != 184:
        raise ValueError(f"seed {seed} must contain 184 unique tumor images")
    return rows


def mean_sd(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("mean/SD requires at least two values")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def summarize(
    binary_root: Path,
    ten_root: Path,
    binary_audit_path: Path,
    ten_audit_path: Path,
    *,
    iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    binary_audit = read_json(binary_audit_path)
    ten_audit = read_json(ten_audit_path)
    for expected, payload in (("binary", binary_audit), ("ten_class", ten_audit)):
        if (
            payload.get("pass") is not True
            or payload.get("arm") != expected
            or payload.get("test_evaluated") is not False
            or int(payload.get("test_images_read", -1)) != 0
        ):
            raise ValueError(f"{expected} audit is invalid")

    seed_reports: list[dict[str, object]] = []
    binary_dice: list[float] = []
    ten_dice: list[float] = []
    subgroup_binary = {group: [] for group in SUBGROUPS}
    subgroup_ten = {group: [] for group in SUBGROUPS}
    for seed_index, seed in enumerate(SEEDS):
        binary_rows = read_rows(binary_root, seed)
        ten_rows = read_rows(ten_root, seed)
        paired: dict[str, object] = {}
        for group_index, group in enumerate(("overall", *SUBGROUPS)):
            binary_subset = (
                binary_rows
                if group == "overall"
                else [row for row in binary_rows if row["size_group"] == group]
            )
            ten_subset = (
                ten_rows
                if group == "overall"
                else [row for row in ten_rows if row["size_group"] == group]
            )
            paired[group] = paired_group_bootstrap_deltas(
                binary_subset,
                ten_subset,
                metrics=METRICS,
                iterations=iterations,
                seed=bootstrap_seed + seed_index * 10 + group_index,
            )
        by_id_binary = {str(row["image_id"]): row for row in binary_rows}
        by_id_ten = {str(row["image_id"]): row for row in ten_rows}
        wins = sum(
            float(by_id_ten[key]["dice"]) > float(by_id_binary[key]["dice"])
            for key in by_id_binary
        )
        ties = sum(
            float(by_id_ten[key]["dice"]) == float(by_id_binary[key]["dice"])
            for key in by_id_binary
        )
        seed_reports.append({
            "seed": seed,
            "paired_group_bootstrap": paired,
            "ten_class_vs_binary_image_wins": wins,
            "ties": ties,
            "losses": 184 - wins - ties,
        })

        binary_summary = next(
            item["summary"] for item in binary_audit["seed_results"]
            if int(item["seed"]) == seed
        )
        ten_summary = next(
            item["summary"] for item in ten_audit["seed_results"]
            if int(item["seed"]) == seed
        )
        binary_dice.append(float(binary_summary["overall"]["dice"]))
        ten_dice.append(float(ten_summary["overall"]["dice"]))
        for group in SUBGROUPS:
            subgroup_binary[group].append(float(binary_summary[group]["dice"]))
            subgroup_ten[group].append(float(ten_summary[group]["dice"]))

    deltas = [ten - binary for binary, ten in zip(binary_dice, ten_dice)]
    subgroup = {}
    for group in SUBGROUPS:
        group_deltas = [
            ten - binary
            for binary, ten in zip(subgroup_binary[group], subgroup_ten[group])
        ]
        subgroup[group] = {
            "binary": mean_sd(subgroup_binary[group]),
            "ten_class": mean_sd(subgroup_ten[group]),
            "ten_class_minus_binary": mean_sd(group_deltas),
            "per_seed_delta": dict(zip(map(str, SEEDS), group_deltas)),
        }
    return {
        "schema_version": 1,
        "study": "G4 E1 matched binary-versus-ten-class downstream WSSS",
        "delta_direction": "ten_class_minus_binary",
        "binary_audit_sha256": sha256(binary_audit_path),
        "ten_class_audit_sha256": sha256(ten_audit_path),
        "overall": {
            "binary": mean_sd(binary_dice),
            "ten_class": mean_sd(ten_dice),
            "ten_class_minus_binary": mean_sd(deltas),
            "per_seed_delta": dict(zip(map(str, SEEDS), deltas)),
        },
        "subgroup": subgroup,
        "seed_reports": seed_reports,
        "statistical_note": (
            "Three training seeds are reported individually and as mean/sample SD. "
            "Within each matched seed, uncertainty is a paired nonparametric "
            "bootstrap over heuristic groups; group_id is not a verified patient ID."
        ),
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--ten-class-root", type=Path, required=True)
    parser.add_argument("--binary-audit", type=Path, required=True)
    parser.add_argument("--ten-class-audit", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("E1 paired comparison output already exists")
    report = summarize(
        args.binary_root,
        args.ten_class_root,
        args.binary_audit,
        args.ten_class_audit,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_sha256": sha256(args.output),
        "overall": report["overall"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
