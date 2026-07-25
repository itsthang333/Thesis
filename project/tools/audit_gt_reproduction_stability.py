from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from audit_gt_reproduction import (
    EXPECTED_WRAPPER_SHA256 as V2_WRAPPER_SHA256,
    audit_gt_reproduction,
)
from audit_wsl_gt_pair import (
    SIZE_ORDER,
    build_audit,
    lesion_size,
    read_csv,
    sha256_file,
)


def _tumor_index(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    indexed = {
        row["image_name"]: row
        for row in rows
        if str(row.get("gt_positive", "")).strip().casefold() in {"1", "true"}
    }
    if len(indexed) != 184:
        raise ValueError(f"Expected 184 tumor rows in {path}, got {len(indexed)}")
    return indexed


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must have equal length >= 2")
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return float("nan")
    return sum(
        l_value * r_value
        for l_value, r_value in zip(centered_left, centered_right, strict=True)
    ) / denominator


def per_image_stability(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _tumor_index(left_path)
    right = _tumor_index(right_path)
    if set(left) != set(right):
        raise ValueError("Tumor cohorts differ between repeated runs")

    output: dict[str, Any] = {}
    for subgroup in ("overall", *SIZE_ORDER):
        names = [
            name
            for name, row in left.items()
            if subgroup == "overall"
            or lesion_size(float(row["gt_area_ratio"])) == subgroup
        ]
        left_values = [float(left[name]["dice"]) for name in names]
        right_values = [float(right[name]["dice"]) for name in names]
        deltas = [
            right_value - left_value
            for left_value, right_value in zip(
                left_values, right_values, strict=True
            )
        ]
        abs_deltas = [abs(value) for value in deltas]
        output[subgroup] = {
            "images": len(names),
            "mean_signed_delta_right_minus_left": statistics.fmean(deltas),
            "mean_absolute_per_image_delta": statistics.fmean(abs_deltas),
            "rmse_per_image_delta": math.sqrt(
                statistics.fmean(value * value for value in deltas)
            ),
            "pearson_per_image_dice": _pearson(left_values, right_values),
            "images_abs_delta_gt_0_05": sum(value > 0.05 for value in abs_deltas),
            "images_abs_delta_gt_0_10": sum(value > 0.10 for value in abs_deltas),
            "maximum_absolute_per_image_delta": max(abs_deltas),
        }
    largest = sorted(
        (
            {
                "image_name": name,
                "group_id": left[name]["group_id"],
                "lesion_size": lesion_size(float(left[name]["gt_area_ratio"])),
                "left_dice": float(left[name]["dice"]),
                "right_dice": float(right[name]["dice"]),
                "delta_right_minus_left": (
                    float(right[name]["dice"]) - float(left[name]["dice"])
                ),
            }
            for name in left
        ),
        key=lambda row: abs(row["delta_right_minus_left"]),
        reverse=True,
    )
    output["largest_absolute_deltas"] = largest[:20]
    return output


def environment_difference(
    left_summary_path: Path,
    right_summary_path: Path,
) -> dict[str, Any]:
    left = json.loads(left_summary_path.read_text(encoding="utf-8"))["environment"]
    right = json.loads(right_summary_path.read_text(encoding="utf-8"))["environment"]
    keys = sorted(set(left) | set(right))
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def audit_stability(
    *,
    reference_lock: Path,
    v2_root: Path,
    v2_wrapper: Path,
    v3_root: Path,
    v3_wrapper: Path,
    v3_wrapper_sha256: str,
    v3_protocol: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    protocol = json.loads(v3_protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_launch":
        raise ValueError("V3 protocol was not locked before launch")
    if protocol.get("test_evaluated") is not False:
        raise ValueError("V3 protocol does not keep test locked")
    if protocol.get("wrapper_sha256") != v3_wrapper_sha256:
        raise ValueError("V3 protocol wrapper hash mismatch")
    if sha256_file(v3_wrapper) != v3_wrapper_sha256:
        raise ValueError("V3 physical wrapper hash mismatch")

    v2_audit = audit_gt_reproduction(
        reference_lock,
        v2_root,
        wrapper_path=v2_wrapper,
        expected_wrapper_sha256=V2_WRAPPER_SHA256,
        iterations=iterations,
        seed=seed,
    )
    v3_audit = audit_gt_reproduction(
        reference_lock,
        v3_root,
        wrapper_path=v3_wrapper,
        expected_wrapper_sha256=v3_wrapper_sha256,
        iterations=iterations,
        seed=seed,
    )

    lock = json.loads(reference_lock.read_text(encoding="utf-8"))
    lock_root = reference_lock.parent
    split_manifest = (lock_root / lock["data"]["split_manifest"]).resolve()
    v2_per_image = v2_root / "evaluation" / "selected_per_image.csv"
    v3_per_image = v3_root / "evaluation" / "selected_per_image.csv"
    v2_to_v3 = build_audit(
        split_manifest,
        v2_per_image,
        v3_per_image,
        iterations=iterations,
        seed=seed,
    )
    v2_to_v3["protocol"] = (
        "paired repeated fully supervised GT runs; v3 candidate minus v2"
    )

    v2_checkpoint = v2_audit["candidate_contract"]["checkpoint_sha256"]
    v3_checkpoint = v3_audit["candidate_contract"]["checkpoint_sha256"]
    subgroup_repeat_gaps = {
        subgroup: abs(
            float(
                v2_to_v3["paired_gap"][subgroup][
                    "signed_gap_candidate_minus_reference"
                ]
            )
        )
        for subgroup in SIZE_ORDER
    }
    material_repeat_instability = (
        v2_checkpoint != v3_checkpoint
        and any(value > 0.05 for value in subgroup_repeat_gaps.values())
    )
    return {
        "status": "PASS",
        "audit_role": (
            "fully supervised GT reference stability audit; not a WSL result"
        ),
        "test_evaluated": False,
        "v3_protocol": {
            "path": str(v3_protocol.resolve()),
            "sha256": sha256_file(v3_protocol),
            "wrapper_sha256": v3_wrapper_sha256,
            "status": "PASS",
        },
        "v2_against_frozen_reference": v2_audit,
        "v3_against_frozen_reference": v3_audit,
        "v3_against_v2": v2_to_v3,
        "v3_vs_v2_per_image_stability": per_image_stability(
            v2_per_image,
            v3_per_image,
        ),
        "environment_differences_v2_to_v3": environment_difference(
            v2_root / "convergence_summary.json",
            v3_root / "convergence_summary.json",
        ),
        "stability_conclusion": {
            "metric_split_and_population_contract_consistent": True,
            "checkpoint_sha256_equal_between_repeats": (
                v2_checkpoint == v3_checkpoint
            ),
            "absolute_mean_repeat_gap_by_size": subgroup_repeat_gaps,
            "material_repeat_instability_detected": material_repeat_instability,
            "interpretation": (
                "A different checkpoint or score after both fail-closed audits "
                "is training-run variability, not evidence of a changed split "
                "or metric. This audit alone does not identify the exact numeric "
                "kernel responsible."
                if v2_checkpoint != v3_checkpoint
                else
                "The repeated run is byte-identical at the selected checkpoint."
            ),
            "frozen_reference_replacement_authorized": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-lock", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v2-wrapper", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--v3-wrapper", type=Path, required=True)
    parser.add_argument("--v3-wrapper-sha256", required=True)
    parser.add_argument("--v3-protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    result = audit_stability(
        reference_lock=args.reference_lock.resolve(),
        v2_root=args.v2_root.resolve(),
        v2_wrapper=args.v2_wrapper.resolve(),
        v3_root=args.v3_root.resolve(),
        v3_wrapper=args.v3_wrapper.resolve(),
        v3_wrapper_sha256=args.v3_wrapper_sha256,
        v3_protocol=args.v3_protocol.resolve(),
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
