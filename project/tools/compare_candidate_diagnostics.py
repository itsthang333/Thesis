from __future__ import annotations

"""Paired complete-group comparison for frozen candidate diagnostics."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path: Path, key: str = "image_name") -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows) or "" in indexed:
        raise ValueError(f"Duplicate/empty {key} in {path}")
    return indexed


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def finite_or_complete_miss_zero(value: str) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def paired_group_bootstrap(
    rows: list[dict[str, object]],
    *,
    baseline_key: str,
    candidate_key: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for row in rows:
        baseline = float(row[baseline_key])
        candidate = float(row[candidate_key])
        grouped.setdefault(str(row["group_id"]), []).append(candidate - baseline)
        baseline_values.append(baseline)
        candidate_values.append(candidate)
    if not rows:
        raise ValueError("Cannot bootstrap an empty cohort")
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled: list[float] = []
        for _ in group_ids:
            sampled.extend(grouped[rng.choice(group_ids)])
        samples.append(statistics.fmean(sampled))
    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    return {
        "images": len(rows),
        "groups": len(group_ids),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_delta": candidate_mean - baseline_mean,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "complete frozen validation group",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--baseline-prompt-quality", type=Path, required=True)
    parser.add_argument("--candidate-per-image", type=Path, required=True)
    parser.add_argument("--candidate-prompt-quality", type=Path, required=True)
    parser.add_argument("--expected-baseline-per-image-sha256", required=True)
    parser.add_argument("--expected-baseline-prompt-quality-sha256", required=True)
    parser.add_argument("--expected-candidate-per-image-sha256", required=True)
    parser.add_argument("--expected-candidate-prompt-quality-sha256", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_hashes = {
        args.baseline_per_image: args.expected_baseline_per_image_sha256,
        args.baseline_prompt_quality: args.expected_baseline_prompt_quality_sha256,
        args.candidate_per_image: args.expected_candidate_per_image_sha256,
        args.candidate_prompt_quality: args.expected_candidate_prompt_quality_sha256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Caller-locked artifact hash mismatch: {path}")

    baseline_per = read_index(args.baseline_per_image)
    candidate_per = read_index(args.candidate_per_image)
    baseline_prompt = read_index(args.baseline_prompt_quality)
    candidate_prompt = read_index(args.candidate_prompt_quality)
    if baseline_per.keys() != candidate_per.keys() or len(candidate_per) != 371:
        raise ValueError("Baseline/candidate full validation cohorts differ")
    tumor_names = {
        name for name, row in candidate_per.items() if row.get("group", "").lower() == "tumor"
    }
    if len(tumor_names) != 184:
        raise ValueError("Candidate evaluation must contain exactly 184 tumor images")
    if set(baseline_prompt) != tumor_names or set(candidate_prompt) != tumor_names:
        raise ValueError("Prompt diagnostic cohort must contain all and only 184 tumors")

    paired: list[dict[str, object]] = []
    for name in sorted(tumor_names):
        baseline_row = baseline_per[name]
        candidate_row = candidate_per[name]
        if baseline_row["group_id"] != candidate_row["group_id"]:
            raise ValueError(f"Frozen group mismatch for {name}")
        baseline_area = float(baseline_row["gt_area_ratio"])
        candidate_area = float(candidate_row["gt_area_ratio"])
        if abs(baseline_area - candidate_area) > 1e-12:
            raise ValueError(f"Frozen GT area mismatch for {name}")
        candidate_prompt_area = float(candidate_prompt[name]["tumor_area_ratio"])
        if abs(candidate_area - candidate_prompt_area) > 1e-12:
            raise ValueError(f"Candidate evaluator GT area mismatch for {name}")
        paired.append(
            {
                "image_name": name,
                "group_id": candidate_row["group_id"],
                "size_group": size_group(candidate_area),
                "baseline_final_dice": float(baseline_row["dice"]),
                "candidate_final_dice": float(candidate_row["dice"]),
                "baseline_oracle_dice": finite_or_complete_miss_zero(
                    baseline_prompt[name]["oracle_best_single_dice"]
                ),
                "candidate_oracle_dice": finite_or_complete_miss_zero(
                    candidate_prompt[name]["oracle_best_single_dice"]
                ),
            }
        )
    expected_sizes = {"small": 94, "medium": 72, "large": 18}
    actual_sizes = {
        group: sum(row["size_group"] == group for row in paired)
        for group in expected_sizes
    }
    if actual_sizes != expected_sizes:
        raise ValueError(f"Frozen tumor-size subgroup counts changed: {actual_sizes}")

    reports: dict[str, object] = {}
    for metric, baseline_key, candidate_key in (
        ("final_dice", "baseline_final_dice", "candidate_final_dice"),
        ("oracle_best_single_dice", "baseline_oracle_dice", "candidate_oracle_dice"),
    ):
        reports[metric] = {
            "overall": paired_group_bootstrap(
                paired,
                baseline_key=baseline_key,
                candidate_key=candidate_key,
                iterations=args.iterations,
                seed=args.seed,
            ),
            "subgroups": {
                group: paired_group_bootstrap(
                    [row for row in paired if row["size_group"] == group],
                    baseline_key=baseline_key,
                    candidate_key=candidate_key,
                    iterations=args.iterations,
                    seed=args.seed,
                )
                for group in ("small", "medium", "large")
            },
        }
    oracle_gate = (
        reports["oracle_best_single_dice"]["subgroups"]["small"]["ci95_low"] > 0.0
        and reports["oracle_best_single_dice"]["overall"]["mean_delta"] >= 0.0
    )
    direct_gate = (
        reports["final_dice"]["overall"]["ci95_low"] > 0.0
        and reports["final_dice"]["subgroups"]["small"]["mean_delta"] >= 0.0
    )
    result = {
        "status": "PASS",
        "split": "val",
        "test_evaluated": False,
        "cohort": {
            "images": 371,
            "tumor_images": 184,
            "subgroups": expected_sizes,
        },
        "artifact_hashes": {
            str(path): expected for path, expected in expected_hashes.items()
        },
        "metrics": reports,
        "promotion_gates": {
            "localization_source_to_selector_research": (
                "PASS" if oracle_gate else "FAIL"
            ),
            "direct_train_pseudo_mask_generation": (
                "PASS" if direct_gate else "FAIL"
            ),
        },
        "decision": (
            "DIRECT_PROMOTE"
            if direct_gate
            else "SELECTOR_RESEARCH_ONLY"
            if oracle_gate
            else "REJECT"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
