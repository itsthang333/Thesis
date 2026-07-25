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


def compare_training_logs(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = read_csv(left_path)
    right = read_csv(right_path)
    if not left or not right:
        raise ValueError("Training logs must be non-empty")
    if list(left[0]) != list(right[0]):
        raise ValueError("Training-log schemas differ")
    numeric_columns = [column for column in left[0] if column != "epoch"]
    common_epochs = min(len(left), len(right))
    exact_prefix = 0
    first_numeric_divergence: int | None = None
    for index in range(common_epochs):
        if left[index] == right[index] and exact_prefix == index:
            exact_prefix += 1
        if first_numeric_divergence is None and any(
            not math.isclose(
                float(left[index][column]),
                float(right[index][column]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for column in numeric_columns
        ):
            first_numeric_divergence = index + 1
    return {
        "left_epochs": len(left),
        "right_epochs": len(right),
        "exact_equal_prefix_epochs": exact_prefix,
        "first_numeric_divergence_epoch": first_numeric_divergence,
        "epoch_1": {
            "left_val_positive_dice": float(left[0]["val_positive_dice"]),
            "right_val_positive_dice": float(right[0]["val_positive_dice"]),
            "delta_right_minus_left": (
                float(right[0]["val_positive_dice"])
                - float(left[0]["val_positive_dice"])
            ),
        },
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

    determinism_audit_path = (
        reference_lock.parent / "reproducibility_static_audit.json"
    )
    pretrained_audit_path = (
        reference_lock.parent / "pretrained_weight_audit.json"
    )
    determinism_audit = json.loads(
        determinism_audit_path.read_text(encoding="utf-8")
    )
    pretrained_audit = json.loads(
        pretrained_audit_path.read_text(encoding="utf-8")
    )
    if determinism_audit.get("status") != "PASS_WITH_LIMITATIONS":
        raise ValueError("GT determinism limitation audit is absent or changed")
    if (
        determinism_audit.get("reference_status", {}).get(
            "hash_locked_reference_invalidated"
        )
        is not False
    ):
        raise ValueError("GT determinism audit invalidates the frozen reference")
    if pretrained_audit.get("status") != "PASS_WITH_PROVENANCE_LIMITATION":
        raise ValueError("Pretrained-weight provenance audit is absent or changed")
    if pretrained_audit.get("sha256") != (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    ):
        raise ValueError("Pretrained ResNet-18 weight hash changed")
    if int(pretrained_audit.get("bytes", -1)) != 46_830_571:
        raise ValueError("Pretrained ResNet-18 weight size changed")
    if (
        pretrained_audit.get("reference_status", {}).get(
            "test_evaluated"
        )
        is not False
    ):
        raise ValueError("Pretrained-weight audit does not keep test locked")

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
    reference_snapshot = (
        lock_root / lock["reference_snapshot_root"]
    ).resolve()
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
    reference_summary = json.loads(
        (reference_snapshot / "convergence_summary.json").read_text(
            encoding="utf-8"
        )
    )
    reference_log = reference_snapshot / "training" / "training_log.csv"
    v2_log = (
        v2_root / "fs_resnet18_pw10_full_448_seed42" / "training_log.csv"
    )
    v3_log = (
        v3_root / "fs_resnet18_pw10_full_448_seed42" / "training_log.csv"
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
        "reference_reproducibility_evidence": {
            "determinism_static_audit": {
                "path": str(determinism_audit_path.resolve()),
                "sha256": sha256_file(determinism_audit_path),
                "status": determinism_audit["status"],
                "bitwise_reproduction_guaranteed": (
                    determinism_audit["limitations"][
                        "bitwise_checkpoint_reproduction_guaranteed"
                    ]
                ),
            },
            "pretrained_encoder_weight_audit": {
                "path": str(pretrained_audit_path.resolve()),
                "sha256": sha256_file(pretrained_audit_path),
                "status": pretrained_audit["status"],
                "weight_sha256": pretrained_audit["sha256"],
                "weight_bytes": pretrained_audit["bytes"],
                "in_kernel_hash_assertion": False,
            },
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
        "training_lineage_and_trajectory": {
            "frozen_reference": {
                "resume_epoch": reference_summary["training"].get(
                    "resume_epoch"
                ),
                "resume_checkpoint_sha256": reference_summary[
                    "environment"
                ].get("resume_sha256"),
                "best_epoch": reference_summary["training"]["best_epoch"],
                "last_completed_epoch": reference_summary["training"][
                    "last_completed_epoch"
                ],
            },
            "v2": {
                "start_epoch": 1,
                "best_epoch": v2_audit["candidate_contract"][
                    "checkpoint_selection_recomputed"
                ]["best_epoch"],
            },
            "v3": {
                "start_epoch": 1,
                "best_epoch": v3_audit["candidate_contract"][
                    "checkpoint_selection_recomputed"
                ]["best_epoch"],
            },
            "reference_vs_v2": compare_training_logs(reference_log, v2_log),
            "reference_vs_v3": compare_training_logs(reference_log, v3_log),
            "v2_vs_v3": compare_training_logs(v2_log, v3_log),
            "interpretation_rule": (
                "If v2 and v3 are identical while each differs from the "
                "historical resumed reference, the current fresh-run recipe "
                "is reproducible and historical lineage explains the external "
                "score mismatch. If v2 and v3 diverge from epoch 1 under the "
                "same environment and hashes, training itself is not bitwise "
                "reproducible under the recorded determinism controls."
            ),
        },
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
