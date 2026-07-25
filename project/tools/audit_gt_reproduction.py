from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from audit_wsl_gt_pair import (
    SIZE_ORDER,
    build_audit,
    sha256_file,
    verify_reference_lock,
)


EXPECTED_WRAPPER_SHA256 = (
    "afa5d4be59a062c2822f467a0dfcb1019842e81dc369646653abd859df8bfcb1"
)
EXPECTED_THRESHOLD_GRID = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_best_checkpoint(
    rows: list[dict[str, str]],
    *,
    tolerance: float = 1e-4,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Training log is empty")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError(f"Training epochs are not contiguous from 1: {epochs}")
    best_dice = -math.inf
    best_specificity = -math.inf
    best_epoch = -1
    for row in rows:
        dice = float(row["val_positive_dice"])
        specificity = float(row["val_empty_specificity"])
        if not math.isfinite(dice) or not math.isfinite(specificity):
            raise ValueError(f"Non-finite validation metric at epoch {row['epoch']}")
        improved = dice > best_dice + tolerance
        tied = abs(dice - best_dice) <= tolerance
        if improved or (tied and specificity > best_specificity):
            best_dice = dice
            best_specificity = specificity
            best_epoch = int(row["epoch"])
    return {
        "best_epoch": best_epoch,
        "best_val_positive_dice_at_0_5": best_dice,
        "best_val_normal_specificity_at_0_5": best_specificity,
        "last_completed_epoch": epochs[-1],
    }


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def normalize_source_hashes(hashes: dict[str, str]) -> dict[str, str]:
    """Normalize the optional repository-root prefix without weakening hashes."""
    normalized: dict[str, str] = {}
    for raw_path, digest in hashes.items():
        path = str(raw_path).replace("\\", "/")
        if path.startswith("project/"):
            path = path[len("project/") :]
        if path in normalized:
            raise ValueError(f"Duplicate normalized source-hash path: {path}")
        normalized[path] = str(digest)
    return normalized


def audit_gt_reproduction(
    reference_lock_path: Path,
    candidate_root: Path,
    *,
    wrapper_path: Path | None = None,
    expected_wrapper_sha256: str = EXPECTED_WRAPPER_SHA256,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    reference_lock_path = reference_lock_path.resolve()
    candidate_root = candidate_root.resolve()
    reference_verification = verify_reference_lock(reference_lock_path)
    lock = json.loads(reference_lock_path.read_text(encoding="utf-8"))
    lock_root = reference_lock_path.parent
    snapshot = (lock_root / lock["reference_snapshot_root"]).resolve()
    split_manifest = (lock_root / lock["data"]["split_manifest"]).resolve()
    reference_per_image = snapshot / "evaluation" / "selected_per_image.csv"

    summary_path = candidate_root / "convergence_summary.json"
    training_log_path = (
        candidate_root
        / "fs_resnet18_pw10_full_448_seed42"
        / "training_log.csv"
    )
    evaluation_root = candidate_root / "evaluation"
    candidate_per_image = evaluation_root / "selected_per_image.csv"
    selected_summary_path = evaluation_root / "selected_summary.json"
    fixed_summary_path = evaluation_root / "fixed_summary.json"
    threshold_selection_path = (
        evaluation_root / "fixed_per_image_threshold_selection.json"
    )
    required = [
        summary_path,
        training_log_path,
        candidate_per_image,
        selected_summary_path,
        fixed_summary_path,
        threshold_selection_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Candidate evidence is incomplete: {missing}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_summary = json.loads(selected_summary_path.read_text(encoding="utf-8"))
    fixed_summary = json.loads(fixed_summary_path.read_text(encoding="utf-8"))
    threshold_selection = json.loads(
        threshold_selection_path.read_text(encoding="utf-8")
    )
    if summary.get("test_evaluated") is not False:
        raise ValueError("Candidate does not keep test locked")
    if "independent epoch-1 GT training" not in str(summary.get("protocol", "")):
        raise ValueError("Candidate is not labelled as an independent GT reproduction")
    environment = summary["environment"]
    if environment.get("reproduction_mode") != (
        "independent epoch-1 training with frozen seed-42 contract"
    ):
        raise ValueError("Candidate reproduction mode mismatch")
    if environment.get("split_sha256") != lock["data"]["split_manifest_sha256"]:
        raise ValueError("Candidate split SHA-256 mismatch")
    cloud_source_hashes = environment.get("source_canonical_lf_sha256")
    if not isinstance(cloud_source_hashes, dict):
        raise ValueError("Candidate source-hash map is missing or malformed")
    if normalize_source_hashes(cloud_source_hashes) != normalize_source_hashes(
        lock["source_canonical_lf_sha256"]
    ):
        raise ValueError("Candidate frozen source hashes mismatch")

    training_rows = read_csv(training_log_path)
    selected_checkpoint = select_best_checkpoint(training_rows)
    cloud_training = summary["training"]
    if int(cloud_training["start_epoch"]) != 1:
        raise ValueError("Independent reproduction did not start at epoch 1")
    if int(cloud_training["best_epoch"]) != selected_checkpoint["best_epoch"]:
        raise ValueError("Cloud best epoch disagrees with independent log selection")
    if int(cloud_training["last_completed_epoch"]) != selected_checkpoint[
        "last_completed_epoch"
    ]:
        raise ValueError("Cloud last epoch disagrees with the training log")
    _require_close(
        float(cloud_training["best_val_positive_dice_at_0_5"]),
        float(selected_checkpoint["best_val_positive_dice_at_0_5"]),
        "best validation positive Dice",
    )
    last_epoch = int(selected_checkpoint["last_completed_epoch"])
    best_epoch = int(selected_checkpoint["best_epoch"])
    if last_epoch > 35:
        raise ValueError("Candidate exceeded the frozen 35-epoch budget")
    if last_epoch < 35 and last_epoch - best_epoch != 10:
        raise ValueError("Early stopping does not match frozen patience 10")

    if threshold_selection.get("selection_split") != "val":
        raise ValueError("Threshold was not selected on validation")
    if int(threshold_selection.get("candidate_count", -1)) != len(
        EXPECTED_THRESHOLD_GRID
    ):
        raise ValueError("Threshold-grid candidate count mismatch")
    selected_threshold = float(threshold_selection["selected"]["threshold"])
    if selected_threshold not in EXPECTED_THRESHOLD_GRID:
        raise ValueError("Selected threshold is outside the frozen grid")
    _require_close(
        selected_threshold,
        float(summary["selected_threshold"]),
        "summary selected threshold",
    )
    _require_close(
        selected_threshold,
        float(selected_summary["threshold"]),
        "selected-evaluation threshold",
    )
    if float(fixed_summary["threshold"]) != 0.5:
        raise ValueError("Fixed evaluation did not use threshold 0.5")

    paired = build_audit(
        split_manifest,
        reference_per_image,
        candidate_per_image,
        iterations=iterations,
        seed=seed,
    )
    paired["protocol"] = (
        "paired independent GT reproduction versus hash-locked GT reference"
    )
    cloud_artifact_hashes = summary["artifact_sha256"]
    if sha256_file(candidate_per_image) != cloud_artifact_hashes[
        "selected_per_image"
    ]:
        raise ValueError("Candidate per-image hash differs from cloud summary")
    if sha256_file(training_log_path) != cloud_artifact_hashes["training_log"]:
        raise ValueError("Candidate training-log hash differs from cloud summary")

    checkpoint_path = (
        candidate_root
        / "fs_resnet18_pw10_full_448_seed42"
        / "best_unet.pt"
    )
    checkpoint_file_verified = checkpoint_path.is_file()
    if checkpoint_file_verified:
        if sha256_file(checkpoint_path) != cloud_training["checkpoint_sha256"]:
            raise ValueError("Downloaded candidate checkpoint hash mismatch")

    wrapper_verification: dict[str, Any] | None = None
    if wrapper_path is not None:
        actual_wrapper_hash = sha256_file(wrapper_path.resolve())
        if actual_wrapper_hash != expected_wrapper_sha256:
            raise ValueError("Independent-reproduction wrapper SHA-256 mismatch")
        wrapper_verification = {
            "path": str(wrapper_path.resolve()),
            "sha256": actual_wrapper_hash,
            "status": "PASS",
        }

    return {
        "status": "PASS",
        "audit_role": "GT reference reproducibility; not a WSL success claim",
        "test_evaluated": False,
        "reference_verification": reference_verification,
        "candidate_contract": {
            "mode": environment["reproduction_mode"],
            "source_hashes": "PASS",
            "split_hash": "PASS",
            "checkpoint_selection_recomputed": selected_checkpoint,
            "threshold_selection": {
                "grid": EXPECTED_THRESHOLD_GRID,
                "selected": selected_threshold,
                "status": "PASS",
            },
            "checkpoint_sha256": cloud_training["checkpoint_sha256"],
            "checkpoint_file_verified": checkpoint_file_verified,
            "wrapper": wrapper_verification,
        },
        "paired_reproduction": paired,
        "reproduction_within_abs_0_05_all_size_subgroups": all(
            paired["paired_gap"][subgroup]["criterion_abs_gap_le_0_05"]
            for subgroup in SIZE_ORDER
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-lock", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument(
        "--expected-wrapper-sha256",
        default=EXPECTED_WRAPPER_SHA256,
        help="Exact predeclared wrapper hash; defaults to the v2 reproduction wrapper.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    result = audit_gt_reproduction(
        args.reference_lock,
        args.candidate_root,
        wrapper_path=args.wrapper,
        expected_wrapper_sha256=args.expected_wrapper_sha256,
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
