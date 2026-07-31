from __future__ import annotations

"""Audit the terminal R1 version-3 OOF count-guard rejection without GT."""

import argparse
import json
import math
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

import numpy as np

from audit_mask_bag_normal_prototype_r1_output import (
    BASELINE_ABSOLUTE_COUNT_SPEARMAN,
    BASELINE_CHECKPOINT_SHA256,
    BOUND_WRAPPER_SHA256,
    CACHE_FREEZE_SHA256,
    CHECKOUT_COMMIT,
    EXPECTED_FOLD_SUMMARY,
    FOLDS,
    KERNEL,
    KERNEL_VERSION,
    PROTOCOL_SHA256,
    PROTOTYPE_COUNTS,
    SOURCE_COMMIT,
    SPLIT_SHA256,
    _csv,
    _json,
    _spearman,
    _verify_oof,
    sha256_file,
)


EXPECTED_TRAIN = 2981
EXPECTED_LOG_FAILURE = "all prototype counts increase the frozen count shortcut"
EXPECTED_FOCUSED_TEST = "53 passed in 7.72s"
EXPECTED_FULL_TEST = "332 passed, 1 skipped, 4 warnings in 15.73s"
COUNT_TOLERANCE = 0.02


def expected_relative_paths() -> set[str]:
    paths = {"crossfit_assignment.json"}
    for prototype_count in PROTOTYPE_COUNTS:
        aggregate = f"oof/k_{prototype_count}"
        paths.update(
            {
                f"{aggregate}/oof_predictions.csv",
                f"{aggregate}/oof_summary.json",
            }
        )
        for fold in FOLDS:
            root = f"{aggregate}/fold_{fold}"
            paths.update(
                {
                    f"{root}/adapter.pt",
                    f"{root}/fold_audit.json",
                    f"{root}/heldout_predictions.csv",
                    f"{root}/normal_prototypes.npz",
                }
            )
    return paths


def build_oof_inventory(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    expected = expected_relative_paths()
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"R1 rejected-output inventory differs; missing={missing}, "
            f"unexpected={unexpected}"
        )
    hashes = {
        relative: sha256_file(root / relative)
        for relative in sorted(actual)
    }
    inventory: dict[str, object] = {}
    for prototype_count in PROTOTYPE_COUNTS:
        for fold in FOLDS:
            prefix = f"oof/k_{prototype_count}/fold_{fold}"
            inventory[f"k_{prototype_count}_fold_{fold}"] = {
                "prototype_sha256": hashes[f"{prefix}/normal_prototypes.npz"],
                "adapter_sha256": hashes[f"{prefix}/adapter.pt"],
                "audit_sha256": hashes[f"{prefix}/fold_audit.json"],
                "predictions_sha256": hashes[
                    f"{prefix}/heldout_predictions.csv"
                ],
            }
        prefix = f"oof/k_{prototype_count}"
        inventory[f"k_{prototype_count}_aggregate"] = {
            "oof_predictions_sha256": hashes[f"{prefix}/oof_predictions.csv"],
            "oof_summary_sha256": hashes[f"{prefix}/oof_summary.json"],
        }
    return inventory, hashes


def _command_arguments(tokens: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 2
    while index < len(tokens):
        key = tokens[index]
        if not key.startswith("--") or index + 1 >= len(tokens):
            raise ValueError("R1 runner command is not a flag/value sequence")
        result[key] = tokens[index + 1]
        index += 2
    return result


def verify_kernel_log(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Direct Kaggle log must be a nonempty JSON event list")
    events = [str(row.get("data", "")) for row in payload if isinstance(row, dict)]
    text = "".join(events)
    required = (
        f"$ git checkout --detach {CHECKOUT_COMMIT}",
        EXPECTED_FOCUSED_TEST,
        EXPECTED_FULL_TEST,
        f'RuntimeError: {EXPECTED_LOG_FAILURE}',
    )
    for value in required:
        if value not in text:
            raise ValueError(f"Direct Kaggle log lacks required evidence: {value}")
    runner_lines = [
        value.strip()[2:]
        for value in events
        if value.startswith("$ ")
        and "project/run_mask_bag_normal_prototype_arm.py" in value
    ]
    if len(runner_lines) != 1:
        raise ValueError("Direct Kaggle log must contain exactly one R1 invocation")
    tokens = shlex.split(runner_lines[0])
    if len(tokens) < 3 or not tokens[1].endswith(
        "project/run_mask_bag_normal_prototype_arm.py"
    ):
        raise ValueError("R1 runner command path differs")
    arguments = _command_arguments(tokens)
    expected_arguments = {
        "--expected-split-sha256": SPLIT_SHA256,
        "--expected-selector-cache-freeze-sha256": CACHE_FREEZE_SHA256,
        "--expected-baseline-checkpoint-sha256": BASELINE_CHECKPOINT_SHA256,
        "--expected-baseline-source-commit": "fda732941664e67d4b87a8c3cba071b6979b2214",
        "--expected-baseline-protocol-sha256": "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555",
        "--source-commit": SOURCE_COMMIT,
        "--protocol-sha256": PROTOCOL_SHA256,
        "--baseline-absolute-count-probability-spearman": str(
            BASELINE_ABSOLUTE_COUNT_SPEARMAN
        ),
        "--epochs": "16",
        "--batch-size": "16",
        "--learning-rate": "0.0003",
        "--weight-decay": "0.0001",
        "--prototype-temperature": "0.10",
        "--adapter-hidden-dim": "128",
        "--consistency-weight": "0.10",
        "--residual-drift-weight": "0.001",
        "--count-association-tolerance": "0.02",
        "--fold-count": "5",
        "--seed": "42",
    }
    for key, expected in expected_arguments.items():
        if arguments.get(key) != expected:
            raise ValueError(f"R1 runner argument differs: {key}")
    if "prediction_freeze.json" in text or "validation GT" in text:
        raise ValueError("Direct error log unexpectedly mentions post-OOF evidence")
    return {
        "sha256": sha256_file(path),
        "events": len(payload),
        "focused_tests": EXPECTED_FOCUSED_TEST,
        "full_tests": EXPECTED_FULL_TEST,
        "runner_invocations": 1,
        "terminal_exception": EXPECTED_LOG_FAILURE,
    }


def summarize_count_guard(root: Path) -> list[dict[str, object]]:
    maximum = BASELINE_ABSOLUTE_COUNT_SPEARMAN + COUNT_TOLERANCE
    rows: list[dict[str, object]] = []
    for prototype_count in PROTOTYPE_COUNTS:
        aggregate = root / "oof" / f"k_{prototype_count}"
        predictions = _csv(aggregate / "oof_predictions.csv")
        if len(predictions) != EXPECTED_TRAIN:
            raise ValueError("OOF aggregate row count differs")
        association = _spearman(
            [int(row["candidate_count"]) for row in predictions],
            [float(row["bag_probability"]) for row in predictions],
        )
        losses = np.asarray(
            [float(row["image_bce"]) for row in predictions], dtype=np.float64
        )
        summary = _json(aggregate / "oof_summary.json")
        fold_losses = np.asarray(summary["fold_image_bce"], dtype=np.float64)
        if fold_losses.shape != (5,) or not np.isfinite(fold_losses).all():
            raise ValueError("OOF fold-loss vector differs")
        if not math.isclose(
            float(summary["count_probability_spearman"]),
            association,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("OOF count association does not reproduce")
        if not math.isclose(
            float(summary["mean_oof_image_bce"]),
            float(losses.mean()),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("OOF BCE does not reproduce")
        absolute = abs(association)
        rows.append(
            {
                "prototype_count": prototype_count,
                "mean_oof_image_bce": float(losses.mean()),
                "standard_error_oof_image_bce": float(
                    fold_losses.std(ddof=1) / math.sqrt(5.0)
                ),
                "count_probability_spearman": association,
                "absolute_count_probability_spearman": absolute,
                "maximum_allowed_absolute_count_probability_spearman": maximum,
                "excess_over_guard": absolute - maximum,
                "count_guard_pass": absolute <= maximum,
            }
        )
    if any(row["count_guard_pass"] for row in rows):
        raise ValueError("At least one frozen K unexpectedly passes the count guard")
    return rows


def audit_error_output(root: Path, kernel_log: Path) -> dict[str, object]:
    inventory, hashes = build_oof_inventory(root)
    assignment = _json(root / "crossfit_assignment.json")
    if (
        assignment.get("schema_version") != 1
        or assignment.get("rows") != EXPECTED_TRAIN
        or assignment.get("folds") != 5
        or assignment.get("fold_summary") != EXPECTED_FOLD_SUMMARY
        or len(str(assignment.get("row_payload_sha256", ""))) != 64
    ):
        raise ValueError("R1 cross-fit assignment contract differs")
    try:
        _verify_oof(
            root,
            {"oof_artifact_hashes": inventory},
            {},
            expected_train=EXPECTED_TRAIN,
        )
    except ValueError as error:
        if str(error) != "Every frozen K fails the count-shortcut guard":
            raise
    else:
        raise ValueError("R1 OOF verifier did not reproduce the expected rejection")
    count_guard = summarize_count_guard(root)
    physical_files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "audit_id": "independent_mask_bag_normal_prototype_r1_v3_error_output_v1",
        "status": "OOF_COUNT_SHORTCUT_GUARD_REJECTION_PHYSICALLY_VERIFIED_GT_BLIND",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bound_wrapper_sha256": BOUND_WRAPPER_SHA256,
        "split_sha256": SPLIT_SHA256,
        "selector_cache_freeze_sha256": CACHE_FREEZE_SHA256,
        "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
        "baseline_absolute_count_probability_spearman": (
            BASELINE_ABSOLUTE_COUNT_SPEARMAN
        ),
        "count_association_tolerance": COUNT_TOLERANCE,
        "kernel_log": verify_kernel_log(kernel_log),
        "crossfit_assignment_sha256": hashes["crossfit_assignment.json"],
        "crossfit_assignment": assignment,
        "count_guard_candidates": count_guard,
        "physical_oof_files_verified": len(physical_files) - 1,
        "physical_output_files_verified": len(physical_files),
        "physical_output_bytes_verified": sum(
            path.stat().st_size for path in physical_files
        ),
        "physical_file_sha256": hashes,
        "error_boundary": (
            "after all 15 five-fold group-OOF image-label fits and aggregates; "
            "before prototype-count selection, final fit, validation inference, "
            "prediction freeze, validation GT evaluation or consumer training"
        ),
        "scientific_conclusion": (
            "Every frozen normal-prototype K increases absolute candidate-count "
            "association beyond the predeclared baseline+tolerance guard; R1 is "
            "rejected without K/tolerance rescue or validation Dice access."
        ),
        "t4x2_evidence": (
            "The exact bound wrapper executes and fails its two-T4 real-convolution "
            "guard before baseline/cache/tests/runner; the direct log reaches the "
            "runner. Device names/checksums were not serialized because wrapper "
            "output audit is post-prediction only."
        ),
        "prediction_freeze_created": False,
        "validation_prediction_created": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kernel-log", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_json.exists():
        raise FileExistsError(f"Audit output already exists: {args.audit_json}")
    audit = audit_error_output(args.output_root, args.kernel_log)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
