from __future__ import annotations

"""Summarize the matched 4x3 G4 E2 attribution-by-prompt experiment.

The script never opens images or annotations.  It consumes only the already
audited per-image metric tables produced by the frozen-mask evaluator and uses
paired, complete heuristic groups for uncertainty estimates.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("cam", "gradcam", "gradcam_plus_plus", "layercam")
PROMPTS = ("point", "box", "box_point")
EXPECTED_ARMS = tuple(f"{method}__{prompt}" for method in METHODS for prompt in PROMPTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid bool literal: {value!r}")


def load_arms(roots: list[Path]) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, str]]:
    tables: dict[str, dict[str, dict[str, str]]] = {}
    summary_hashes: dict[str, str] = {}
    for root in roots:
        for arm_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            arm = arm_dir.name
            if arm not in EXPECTED_ARMS:
                continue
            if arm in tables:
                raise ValueError(f"duplicate arm {arm}")
            summary = arm_dir / "summary.json"
            audit = arm_dir / "audit.json"
            per_image = arm_dir / "per_image.csv"
            if not all(path.is_file() for path in (summary, audit, per_image)):
                raise FileNotFoundError(f"incomplete evaluation arm {arm_dir}")
            audit_obj = json.loads(audit.read_text(encoding="utf-8"))
            if audit_obj.get("pass") is not True:
                raise ValueError(f"arm {arm} did not pass its evaluator audit")
            rows: dict[str, dict[str, str]] = {}
            with per_image.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    image_id = row["image_id"]
                    if image_id in rows:
                        raise ValueError(f"duplicate image {image_id} in {arm}")
                    rows[image_id] = row
            tumor_rows = {
                image_id: row for image_id, row in rows.items()
                if _bool(row["gt_positive"])
            }
            if len(rows) != 371 or len(tumor_rows) != 184:
                raise ValueError(
                    f"arm {arm} is not the exact 371/184 validation population"
                )
            tables[arm] = tumor_rows
            summary_hashes[arm] = sha256(summary)

    missing = sorted(set(EXPECTED_ARMS) - set(tables))
    extra = sorted(set(tables) - set(EXPECTED_ARMS))
    if missing or extra:
        raise ValueError(f"factorial is incomplete: missing={missing}, extra={extra}")
    image_ids = sorted(next(iter(tables.values())))
    for arm, rows in tables.items():
        if sorted(rows) != image_ids:
            raise ValueError(f"image IDs differ in {arm}")
    groups = [tables[EXPECTED_ARMS[0]][image_id]["group_id"] for image_id in image_ids]
    for arm, rows in tables.items():
        if [rows[image_id]["group_id"] for image_id in image_ids] != groups:
            raise ValueError(f"group IDs differ in {arm}")
    values = {
        arm: np.asarray([float(tables[arm][image_id]["dice"]) for image_id in image_ids])
        for arm in EXPECTED_ARMS
    }
    return image_ids, groups, values, summary_hashes


def paired_group_ci(
    delta: np.ndarray,
    groups: list[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    unique_groups = sorted(set(groups))
    indices = {
        group: np.asarray([index for index, value in enumerate(groups) if value == group])
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        samples[iteration] = float(delta[selected].mean())
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "delta_mean_dice": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "iterations": int(iterations),
        "groups": len(unique_groups),
    }


def summarize(
    roots: list[Path],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    image_ids, groups, values, hashes = load_arms(roots)
    arm_means = {arm: float(values[arm].mean()) for arm in EXPECTED_ARMS}
    best = max(arm_means, key=arm_means.get)
    method_values = {
        method: np.mean([values[f"{method}__{prompt}"] for prompt in PROMPTS], axis=0)
        for method in METHODS
    }
    prompt_values = {
        prompt: np.mean([values[f"{method}__{prompt}"] for method in METHODS], axis=0)
        for prompt in PROMPTS
    }

    comparisons: dict[str, object] = {}
    for arm in EXPECTED_ARMS:
        if arm != best:
            comparisons[f"{best}_minus_{arm}"] = paired_group_ci(
                values[best] - values[arm], groups, iterations=iterations, seed=seed
            )
    for first, second in (("point", "box"), ("point", "box_point"), ("box_point", "box")):
        comparisons[f"prompt_{first}_minus_{second}"] = paired_group_ci(
            prompt_values[first] - prompt_values[second],
            groups,
            iterations=iterations,
            seed=seed,
        )
    for index, first in enumerate(METHODS):
        for second in METHODS[index + 1 :]:
            comparisons[f"method_{first}_minus_{second}"] = paired_group_ci(
                method_values[first] - method_values[second],
                groups,
                iterations=iterations,
                seed=seed,
            )

    return {
        "schema_version": 1,
        "study": "G4 E2 matched attribution-by-prompt factorial summary",
        "population": "184 canonical validation tumor images",
        "endpoint": "macro mean tumor Dice on the common 320x320 grid",
        "best_arm": best,
        "arm_mean_dice": arm_means,
        "method_marginal_mean_dice": {
            key: float(value.mean()) for key, value in method_values.items()
        },
        "prompt_marginal_mean_dice": {
            key: float(value.mean()) for key, value in prompt_values.items()
        },
        "paired_complete_group_bootstrap": comparisons,
        "uncertainty_note": (
            "Nonparametric paired bootstrap over complete heuristic groups; "
            "group_id is not a verified patient/case identifier."
        ),
        "images": len(image_ids),
        "groups": len(set(groups)),
        "iterations": iterations,
        "seed": seed,
        "source_summary_sha256": hashes,
        "spatial_ground_truth_opened_by_this_script": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    result = summarize(
        args.evaluation_root, iterations=args.iterations, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "best_arm": result["best_arm"],
        "best_dice": result["arm_mean_dice"][result["best_arm"]],
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
