from __future__ import annotations

"""Independent no-GT auditor for rich-gallery G2 Stage-A output."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_rich_gallery_g2_selector_pair import verify_stage_a
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_g2_selector_pair import ARM_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("G2 audit requires canonical 371-image validation")
    selections, freeze = verify_stage_a(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if len(candidate_rows) != 371 or candidate_audit.get("cohort") != "all":
        raise ValueError("G2 candidate cohort audit failed")
    if len(selections) != 371 * 8:
        raise ValueError("G2 audited selection count mismatch")

    for name in ARM_NAMES:
        checkpoint = args.prediction_root / "checkpoints" / f"{name}.pt"
        history = args.prediction_root / "training_history" / f"{name}.json"
        if sha256_file(checkpoint) != freeze["arm_checkpoint_sha256"][name]:
            raise ValueError(f"G2 checkpoint changed: {name}")
        if sha256_file(history) != freeze["training_history_sha256"][name]:
            raise ValueError(f"G2 history changed: {name}")
        rows = json.loads(history.read_text(encoding="utf-8"))
        if len(rows) != 16 or [int(row["epoch"]) for row in rows] != list(range(1, 17)):
            raise ValueError(f"G2 terminal epoch contract failed: {name}")
        if not all(
            np.isfinite(float(value))
            for row in rows
            for value in row.values()
        ):
            raise ValueError(f"G2 history contains non-finite values: {name}")
    hierarchical = json.loads(
        (args.prediction_root / "training_history" / "hierarchical_shared_negative_only.json").read_text(
            encoding="utf-8"
        )
    )
    if hierarchical[0]["temperature"] != 1.0 or not np.isclose(
        hierarchical[-1]["temperature"], 0.2
    ):
        raise ValueError("G2 continuation-temperature endpoints changed")

    for split_name, expected_normal, expected_tumor in (
        ("train_source_report", 1493, 1488),
        ("validation_source_report", 187, 184),
    ):
        report = freeze[split_name]
        if (
            report["normal"]["images"] != expected_normal
            or report["tumor"]["images"] != expected_tumor
            or report["external_source_label_shortcut"]["normal_presence"] != 0
            or report["external_source_label_shortcut"]["tumor_presence"] != expected_tumor
            or report["external_source_label_shortcut"]["confirmed"] is not True
        ):
            raise ValueError(f"G2 source-shortcut evidence changed: {split_name}")

    run_manifest_path = args.prediction_root / "run_manifest.json"
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        run.get("source_commit") != args.expected_source_commit
        or run.get("protocol_sha256") != args.expected_protocol_sha256
        or run.get("split_sha256") != args.expected_split_sha256
        or run.get("prediction_freeze_sha256")
        != args.expected_prediction_freeze_sha256
        or run.get("validation_gt_read") is not False
        or run.get("spatial_ground_truth_used") is not False
        or run.get("consumer_trained") is not False
        or run.get("test_images_read") != 0
        or run.get("test_evaluated") is not False
    ):
        raise ValueError("G2 run manifest safety/provenance mismatch")
    audit = {
        "audit_pass": True,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "validation_images": 371,
        "selection_rows": len(selections),
        "variants": len(freeze["variants"]),
        "g1_reproduction_exact": True,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
