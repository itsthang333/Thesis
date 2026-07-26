from __future__ import annotations

"""Paired complete-group bootstrap for fixed single/multiscale memory arms."""

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
    parser.add_argument("--single-scale-per-image", type=Path, required=True)
    parser.add_argument("--multiscale-per-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 184 or len({row["image_id"] for row in rows}) != 184:
        raise ValueError("Evaluation must contain 184 unique tumor images")
    return {row["image_id"]: row for row in rows}


def paired_group_bootstrap(
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
    return {
        "delta_multiscale_minus_single_scale": float(
            np.mean([delta for _, delta in rows])
        ),
        "ci95": [
            float(value) for value in np.percentile(boot, [2.5, 97.5])
        ],
        "n_images": len(rows),
        "n_groups": len(group_ids),
    }


def main() -> None:
    args = parse_args()
    single = _read(args.single_scale_per_image)
    multiscale = _read(args.multiscale_per_image)
    if set(single) != set(multiscale):
        raise ValueError("Paired arm cohorts differ")
    metric_results: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        strata: dict[str, object] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in single.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            strata[stratum] = paired_group_bootstrap(
                [
                    (
                        single[name]["group_id"],
                        float(multiscale[name][metric])
                        - float(single[name][metric]),
                    )
                    for name in names
                ],
                replicates=args.bootstrap_replicates,
                seed=args.seed + metric_index * 10 + len(stratum),
            )
        metric_results[metric] = strata
    result = {
        "method": "paired complete-group bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "interpretation": (
            "mechanism feasibility only; no arm/threshold promotion and no "
            "downstream consumer without a separate predeclared protocol"
        ),
        "metrics": metric_results,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
