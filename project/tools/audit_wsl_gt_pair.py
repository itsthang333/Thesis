from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


SIZE_ORDER = ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_text(path: Path) -> str:
    """Hash a text artifact after normalizing platform line endings to LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def lesion_size(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small_lt_1pct"
    if area_ratio < 0.05:
        return "medium_1_to_5pct"
    return "large_ge_5pct"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def audit_population(
    split_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
) -> dict[str, Any]:
    eligible_val = {
        row["image_id"]: row
        for row in split_rows
        if row.get("eligible") == "1" and row.get("split") == "val"
    }
    if len(eligible_val) != 371:
        raise ValueError(f"Frozen validation population must contain 371 images, got {len(eligible_val)}")
    indexed: dict[str, dict[str, str]] = {}
    for row in metric_rows:
        name = row.get("image_name", "")
        if not name or name in indexed:
            raise ValueError(f"Missing or duplicate image_name: {name!r}")
        indexed[name] = row
    if set(indexed) != set(eligible_val):
        missing = sorted(set(eligible_val) - set(indexed))
        extra = sorted(set(indexed) - set(eligible_val))
        raise ValueError(f"Metric cohort differs from frozen validation: missing={missing}, extra={extra}")

    tumors: list[dict[str, str]] = []
    normals = 0
    for name, row in indexed.items():
        expected_tumor = int(float(eligible_val[name].get("tumor", "0") or 0))
        actual_tumor = str(row.get("gt_positive", "")).strip().casefold() in {"1", "true"}
        if actual_tumor != bool(expected_tumor):
            raise ValueError(f"GT-positive mismatch for {name}")
        dice = float(row["dice"])
        if not math.isfinite(dice) or not 0.0 <= dice <= 1.0:
            raise ValueError(f"Invalid Dice for {name}: {dice}")
        if actual_tumor:
            tumors.append(row)
        else:
            normals += 1
    if len(tumors) != 184 or normals != 187:
        raise ValueError(f"Expected 184 tumors and 187 normals, got {len(tumors)}/{normals}")

    size_counts = {name: 0 for name in SIZE_ORDER}
    for row in tumors:
        size_counts[lesion_size(float(row["gt_area_ratio"]))] += 1
    expected_counts = {
        "small_lt_1pct": 94,
        "medium_1_to_5pct": 72,
        "large_ge_5pct": 18,
    }
    if size_counts != expected_counts:
        raise ValueError(f"Frozen lesion-size counts changed: {size_counts}")
    return {
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "tumor_groups": len({row["group_id"] for row in tumors}),
        "size_counts": size_counts,
    }


def means_by_size(rows: list[dict[str, str]]) -> dict[str, float]:
    tumors = [
        row
        for row in rows
        if str(row.get("gt_positive", "")).strip().casefold() in {"1", "true"}
    ]
    result = {"overall": statistics.fmean(float(row["dice"]) for row in tumors)}
    for subgroup in SIZE_ORDER:
        selected = [
            float(row["dice"])
            for row in tumors
            if lesion_size(float(row["gt_area_ratio"])) == subgroup
        ]
        result[subgroup] = statistics.fmean(selected)
    return result


def paired_bootstrap(
    reference_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    iterations: int,
    seed: int,
    goal_tolerance: float = 0.05,
) -> dict[str, dict[str, float | int | bool]]:
    if not 0.0 <= goal_tolerance <= 1.0:
        raise ValueError("goal_tolerance must lie in [0,1]")
    reference = {row["image_name"]: row for row in reference_rows}
    candidate = {row["image_name"]: row for row in candidate_rows}
    output: dict[str, dict[str, float | int | bool]] = {}
    rng = random.Random(seed)
    for subgroup in ("overall", *SIZE_ORDER):
        grouped: dict[str, list[float]] = {}
        for name, reference_row in reference.items():
            if str(reference_row.get("gt_positive", "")).strip().casefold() not in {"1", "true"}:
                continue
            if subgroup != "overall" and lesion_size(float(reference_row["gt_area_ratio"])) != subgroup:
                continue
            delta = float(candidate[name]["dice"]) - float(reference_row["dice"])
            grouped.setdefault(reference_row["group_id"], []).append(delta)
        group_ids = sorted(grouped)
        image_deltas = [value for group_id in group_ids for value in grouped[group_id]]
        samples: list[float] = []
        for _ in range(iterations):
            sampled = [
                value
                for _ in group_ids
                for value in grouped[rng.choice(group_ids)]
            ]
            samples.append(statistics.fmean(sampled))
        gap = statistics.fmean(image_deltas)
        output[subgroup] = {
            "images": len(image_deltas),
            "groups": len(group_ids),
            "signed_gap_candidate_minus_reference": gap,
            "absolute_gap": abs(gap),
            "criterion_abs_gap_le_0_05": abs(gap) <= 0.05,
            "goal_tolerance": goal_tolerance,
            "criterion_abs_gap_le_goal_tolerance": abs(gap) <= goal_tolerance,
            "paired_group_bootstrap_ci95_low": percentile(samples, 0.025),
            "paired_group_bootstrap_ci95_high": percentile(samples, 0.975),
        }
    return output


def build_audit(
    split_manifest: Path,
    reference_per_image: Path,
    candidate_per_image: Path | None = None,
    *,
    iterations: int = 10_000,
    seed: int = 42,
    goal_tolerance: float = 0.05,
) -> dict[str, Any]:
    split_rows = read_csv(split_manifest)
    reference_rows = read_csv(reference_per_image)
    population = audit_population(split_rows, reference_rows)
    result: dict[str, Any] = {
        "protocol": "paired GT-trained reference versus image-label-only WSL consumer",
        "split": "val",
        "test_evaluated": False,
        "split_manifest_sha256": sha256_file(split_manifest),
        "reference_per_image_canonical_lf_sha256": sha256_canonical_text(
            reference_per_image
        ),
        "population": population,
        "reference_mean_tumor_dice": means_by_size(reference_rows),
        "size_definitions": {
            "small_lt_1pct": "gt_area_ratio < 0.01",
            "medium_1_to_5pct": "0.01 <= gt_area_ratio < 0.05",
            "large_ge_5pct": "gt_area_ratio >= 0.05",
            "usage": "post-prediction validation diagnostic only; never an inference input",
        },
    }
    if candidate_per_image is not None:
        candidate_rows = read_csv(candidate_per_image)
        candidate_population = audit_population(split_rows, candidate_rows)
        if candidate_population != population:
            raise ValueError("Reference and candidate populations differ")
        reference_index = {row["image_name"]: row for row in reference_rows}
        for row in candidate_rows:
            reference_row = reference_index[row["image_name"]]
            if row["group_id"] != reference_row["group_id"]:
                raise ValueError(f"Group mismatch for {row['image_name']}")
            if not math.isclose(
                float(row["gt_area_ratio"]),
                float(reference_row["gt_area_ratio"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"GT area-ratio mismatch for {row['image_name']}")
        gaps = paired_bootstrap(
            reference_rows,
            candidate_rows,
            iterations=iterations,
            seed=seed,
            goal_tolerance=goal_tolerance,
        )
        result.update(
            {
                "candidate_per_image_canonical_lf_sha256": sha256_canonical_text(
                    candidate_per_image
                ),
                "candidate_mean_tumor_dice": means_by_size(candidate_rows),
                "paired_gap": gaps,
                "primary_success": all(
                    gaps[subgroup]["criterion_abs_gap_le_goal_tolerance"]
                    for subgroup in SIZE_ORDER
                ),
                "goal_tolerance": goal_tolerance,
                "bootstrap": {
                    "unit": "complete validation group",
                    "iterations": iterations,
                    "seed": seed,
                },
            }
        )
    return result


def validate_goal_protocol(
    protocol_path: Path,
    *,
    expected_canonical_lf_sha256: str,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    actual_hash = sha256_canonical_text(protocol_path)
    if actual_hash != expected_canonical_lf_sha256:
        raise ValueError("Paired goal protocol canonical-LF SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "effective_for_future_wsl_consumers":
        raise ValueError("Paired goal protocol is not effective")
    tolerance = float(protocol.get("goal_tolerance", -1))
    if not math.isclose(tolerance, 0.10, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Paired goal v2 tolerance must be 0.10")
    if protocol.get("reference_lock") != "reference_lock.json":
        raise ValueError("Paired goal protocol references an unexpected GT lock")
    if protocol["consumer_invariants"].get("test_evaluated") is not False:
        raise ValueError("Paired goal protocol does not keep test locked")
    expected_minima = {
        "small_lt_1pct": 0.22895493248574225,
        "medium_1_to_5pct": 0.5624417783635557,
        "large_ge_5pct": 0.5937033565801355,
    }
    for subgroup, expected in expected_minima.items():
        actual = float(protocol["subgroup_contract"][subgroup]["new_minimum_wsl_dice"])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Paired goal minimum changed: {subgroup}")
    return {
        "protocol_id": protocol["protocol_id"],
        "canonical_lf_sha256": actual_hash,
        "goal_tolerance": tolerance,
        "test_evaluated": False,
    }


def verify_reference_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "hash_locked":
        raise ValueError("Reference lock is not hash_locked")
    isolation = lock["weak_supervision_isolation"]
    if isolation.get("test_evaluated") is not False:
        raise ValueError("Reference lock does not keep test evaluation disabled")
    if isolation.get("train_gt_may_influence_wsl_generation_or_selection") is not False:
        raise ValueError("Reference lock permits GT leakage into WSL")

    root = lock_path.parent
    snapshot = (root / lock["reference_snapshot_root"]).resolve()
    checkpoint = (root / lock["artifact_hashes"]["checkpoint_path"]).resolve()
    expected_checkpoint_bytes = int(lock["artifact_hashes"]["checkpoint_bytes"])
    if checkpoint.stat().st_size != expected_checkpoint_bytes:
        raise ValueError("Reference checkpoint byte size mismatch")
    if sha256_file(checkpoint) != lock["artifact_hashes"]["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint SHA-256 mismatch")

    canonical_artifacts = {
        "reference_per_image_canonical_lf_sha256": "evaluation/selected_per_image.csv",
        "training_log_canonical_lf_sha256": "training/training_log.csv",
        "selected_summary_canonical_lf_sha256": "evaluation/selected_summary.json",
        "selected_subgroups_canonical_lf_sha256": "evaluation/selected_per_image_subgroups.csv",
    }
    for key, relative in canonical_artifacts.items():
        if sha256_canonical_text(snapshot / relative) != lock["artifact_hashes"][key]:
            raise ValueError(f"Reference artifact hash mismatch: {relative}")
    for relative, expected in lock["source_canonical_lf_sha256"].items():
        if sha256_canonical_text(snapshot / relative) != expected:
            raise ValueError(f"Reference source hash mismatch: {relative}")

    split = (root / lock["data"]["split_manifest"]).resolve()
    per_image = snapshot / "evaluation/selected_per_image.csv"
    metric_audit = build_audit(split, per_image)
    if metric_audit["population"] != {
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "tumor_groups": 167,
        "size_counts": {
            "small_lt_1pct": 94,
            "medium_1_to_5pct": 72,
            "large_ge_5pct": 18,
        },
    }:
        raise ValueError("Reference population audit mismatch")
    expected_means = {
        "overall": lock["reference_result"]["overall_mean_tumor_dice"],
        "small_lt_1pct": lock["reference_result"]["small_lt_1pct_mean_tumor_dice"],
        "medium_1_to_5pct": lock["reference_result"]["medium_1_to_5pct_mean_tumor_dice"],
        "large_ge_5pct": lock["reference_result"]["large_ge_5pct_mean_tumor_dice"],
    }
    for subgroup, expected in expected_means.items():
        if not math.isclose(
            metric_audit["reference_mean_tumor_dice"][subgroup],
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Reference metric mismatch: {subgroup}")
    return {
        "status": "PASS",
        "reference_id": lock["reference_id"],
        "checkpoint_sha256": lock["artifact_hashes"]["checkpoint_sha256"],
        "test_evaluated": False,
        "weak_supervision_isolation": "PASS",
        "population": metric_audit["population"],
        "reference_mean_tumor_dice": metric_audit["reference_mean_tumor_dice"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--reference-per-image", type=Path, required=True)
    parser.add_argument("--candidate-per-image", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=None,
        help="Allowed gap. Omit for historical v1=0.05; v2 is loaded from --goal-protocol.",
    )
    parser.add_argument("--goal-protocol", type=Path)
    parser.add_argument("--expected-goal-protocol-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    goal_protocol = None
    if args.goal_protocol is not None:
        if not args.expected_goal_protocol_sha256:
            raise ValueError(
                "--goal-protocol requires --expected-goal-protocol-sha256"
            )
        goal_protocol = validate_goal_protocol(
            args.goal_protocol,
            expected_canonical_lf_sha256=args.expected_goal_protocol_sha256,
        )
        if (
            args.goal_tolerance is not None
            and not math.isclose(
                args.goal_tolerance,
                float(goal_protocol["goal_tolerance"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("--goal-tolerance conflicts with locked goal protocol")
        goal_tolerance = float(goal_protocol["goal_tolerance"])
    else:
        goal_tolerance = 0.05 if args.goal_tolerance is None else args.goal_tolerance
    result = build_audit(
        args.split_manifest,
        args.reference_per_image,
        args.candidate_per_image,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        goal_tolerance=goal_tolerance,
    )
    result["goal_protocol"] = goal_protocol or {
        "protocol_id": "historical_inline_v1",
        "goal_tolerance": goal_tolerance,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
