from __future__ import annotations

"""Independent no-GT auditor for cross-view co-witness Stage-A output."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_rich_gallery_cross_view_cowitness_pair import verify_stage_a
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_cross_view_cowitness import CrossViewCoWitnessConfig
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_cross_view_cowitness_pair import (
    ARM_NAMES,
    _read_pair_manifest,
    frozen_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--expected-model-preprocessor-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if len(val_rows) != 371:
        raise RuntimeError("cross-view audit requires canonical 371-image validation")
    pair_rows = _read_pair_manifest(
        args.pair_manifest, args.expected_pair_manifest_sha256
    )
    selections, freeze = verify_stage_a(
        args, val_rows, require_independent_audit=False
    )
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if len(candidate_rows) != 371 or candidate_audit.get("cohort") != "all":
        raise ValueError("cross-view candidate cohort audit failed")
    if len(selections) != 371 * len(frozen_variants()) or len(pair_rows) != 384:
        raise ValueError("cross-view audited population mismatch")

    selection_path = args.prediction_root / "stage_a_selection_manifest.csv"
    with selection_path.open("r", newline="", encoding="utf-8-sig") as handle:
        selection_rows = list(csv.DictReader(handle))
    for row in selection_rows:
        candidate = candidate_rows[Path(row["image_id"]).stem]
        if row["candidate_payload_sha256"] != candidate["diagnostic_sha256"]:
            raise ValueError(f"cross-view candidate binding changed: {row['image_id']}")

    expected_model = {
        "config.json": args.expected_model_config_sha256,
        "preprocessor_config.json": args.expected_model_preprocessor_sha256,
        "model.safetensors": args.expected_model_weight_sha256,
    }
    if any(
        freeze["model_snapshot"].get(name, {}).get("sha256") != expected
        for name, expected in expected_model.items()
    ):
        raise ValueError("cross-view frozen model snapshot mismatch")
    expected_projection = projection_sha256(
        make_seeded_random_projection(input_dim=768, output_dim=128, seed=42)
    )
    if freeze.get("projection_sha256") != expected_projection:
        raise ValueError("cross-view descriptor projection changed")

    checkpoint_hashes: dict[str, str] = {}
    history_hashes: dict[str, str] = {}
    for arm in ARM_NAMES:
        checkpoint_path = args.prediction_root / "checkpoints" / f"{arm}.pt"
        history_path = args.prediction_root / "training_history" / f"{arm}.json"
        checkpoint_hashes[arm] = sha256_file(checkpoint_path)
        history_hashes[arm] = sha256_file(history_path)
        if checkpoint_hashes[arm] != freeze["checkpoint_sha256"][arm]:
            raise ValueError(f"cross-view checkpoint changed: {arm}")
        if history_hashes[arm] != freeze["training_history_sha256"][arm]:
            raise ValueError(f"cross-view history changed: {arm}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if (
            len(history) != 2
            or [int(row["epoch"]) for row in history] != [1, 2]
            or any(int(row["steps"]) != 384 for row in history)
            or not all(
                np.isfinite(float(value))
                for row in history
                for value in row.values()
            )
        ):
            raise ValueError(f"cross-view two-pass history contract failed: {arm}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("arm") != arm
            or checkpoint.get("final_epoch") != 2
            or checkpoint.get("source_commit") != args.expected_source_commit
            or checkpoint.get("protocol_sha256") != args.expected_protocol_sha256
            or checkpoint.get("split_sha256") != args.expected_split_sha256
            or checkpoint.get("pair_manifest_sha256")
            != args.expected_pair_manifest_sha256
            or checkpoint.get("training_labels")
            != "binary_image_labels_plus_heuristic_train_group_relation"
            or checkpoint.get("spatial_ground_truth_used") is not False
            or checkpoint.get("test_evaluated") is not False
            or CrossViewCoWitnessConfig(**checkpoint["config"])
            != CrossViewCoWitnessConfig()
        ):
            raise ValueError(f"cross-view checkpoint contract failed: {arm}")
        state = checkpoint.get("model_state_dict", {})
        if not state or not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f"cross-view checkpoint is empty/non-finite: {arm}")

    run_path = args.prediction_root / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run.get("run_id") != "btxrd_rich_gallery_cross_view_cowitness_pair_v1"
        or run.get("source_commit") != args.expected_source_commit
        or run.get("protocol_sha256") != args.expected_protocol_sha256
        or run.get("split_sha256") != args.expected_split_sha256
        or run.get("pair_manifest_sha256") != args.expected_pair_manifest_sha256
        or run.get("prediction_freeze_sha256")
        != args.expected_prediction_freeze_sha256
        or run.get("validation_gt_read") is not False
        or run.get("spatial_ground_truth_used") is not False
        or run.get("test_images_read") != 0
        or run.get("test_evaluated") is not False
    ):
        raise ValueError("cross-view run manifest contract mismatch")

    result = {
        "audit_pass": True,
        "stage": "rich_gallery_cross_view_cowitness_pair_stage_a_audit_v1",
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "pair_manifest_sha256": args.expected_pair_manifest_sha256,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "run_manifest_sha256": sha256_file(run_path),
        "checkpoint_sha256": checkpoint_hashes,
        "training_history_sha256": history_hashes,
        "validation_images": 371,
        "pair_rows": 384,
        "selection_rows": len(selections),
        "variants": frozen_variants(),
        "g1_reproduction_exact": True,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
