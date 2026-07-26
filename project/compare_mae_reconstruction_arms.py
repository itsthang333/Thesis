from __future__ import annotations

"""Paired complete-group bootstrap for the two predeclared MAE probe arms."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "pixel_ap",
    "pixel_auroc",
    "argmax_hit",
    "saliency_mass_in_gt",
    "dice_p90",
    "dice_p95",
    "dice_p97",
    "dice_p99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-per-image", type=Path, required=True)
    parser.add_argument("--adapted-per-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image_id"]: row for row in rows}


def _paired_group_bootstrap(
    rows: list[tuple[str, float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    groups: dict[str, list[float]] = {}
    for group_id, delta in rows:
        groups.setdefault(group_id, []).append(delta)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [value for group in sampled for value in groups[str(group)]]
        boot[index] = np.mean(values)
    observed = float(np.mean([delta for _, delta in rows]))
    return {
        "delta_adapted_minus_base": observed,
        "ci95": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "n_images": len(rows),
        "n_groups": len(group_ids),
    }


def main() -> None:
    args = parse_args()
    base = _read(args.base_per_image)
    adapted = _read(args.adapted_per_image)
    if set(base) != set(adapted) or len(base) != 184:
        raise ValueError("Paired MAE cohorts differ or are incomplete")
    result: dict[str, object] = {
        "method": "paired complete-group bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "interpretation": (
            "mechanism feasibility only; results cannot select a segmentation "
            "threshold or authorize a downstream consumer"
        ),
        "metrics": {},
        "test_evaluated": False,
    }
    metric_results: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        strata: dict[str, object] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in base.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            pairs = [
                (
                    base[name]["group_id"],
                    float(adapted[name][metric]) - float(base[name][metric]),
                )
                for name in names
            ]
            strata[stratum] = _paired_group_bootstrap(
                pairs,
                replicates=args.bootstrap_replicates,
                seed=args.seed + metric_index * 10 + len(stratum),
            )
        metric_results[metric] = strata
    result["metrics"] = metric_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
