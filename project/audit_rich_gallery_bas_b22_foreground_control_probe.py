from __future__ import annotations

"""Independent output audit for the bounded BAS-B2.2 mechanics probe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mae_reconstruction_io import sha256_file
from run_rich_gallery_bas_candidate_descriptor_b1 import _binary_metrics
from run_rich_gallery_bas_b22_foreground_control_probe import (
    _mechanics_gate,
    _spatial_mechanics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    summary_path = args.output_root / "probe_summary.json"
    freeze_path = args.output_root / "mechanics_probe_freeze.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        summary.get("stage")
        != "rich_gallery_bas_b22_foreground_control_mechanics_probe_v1"
        or summary.get("source_commit") != args.expected_source_commit
        or summary.get("protocol_sha256") != args.expected_protocol_sha256
        or summary.get("split_sha256") != args.expected_split_sha256
        or summary.get("candidate_manifest_sha256")
        != args.expected_candidate_manifest_sha256
        or summary.get("classifier_activation") != "softplus"
        or summary.get("scientific_delta_from_b21")
        != "hard_gated_background_ratio_to_continuous_foreground_control"
        or summary.get("objective_weights")
        != {
            "full_ce": 1.0,
            "foreground_ce": 0.5,
            "foreground_control": 1.5,
            "area": 1.2,
            "reference_ratio": 0.5,
        }
        or summary.get("epochs") != 5
        or summary.get("validation_images") != 371
        or summary.get("validation_tumors") != 184
        or summary.get("validation_gt_read") is not False
        or summary.get("test_images_read") != 0
        or summary.get("test_evaluated") is not False
        or summary.get("spatial_efficacy_evaluated") is not False
    ):
        raise ValueError("B2.2 summary contract mismatch")
    if (
        freeze.get("stage")
        != "rich_gallery_bas_b22_foreground_control_mechanics_probe_freeze_v1"
        or freeze.get("candidate_scores_or_choices_frozen") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
        or freeze.get("probe_summary_sha256") != sha256_file(summary_path)
    ):
        raise ValueError("B2.2 freeze contract mismatch")

    history_path = args.output_root / "training_history.csv"
    predictions_path = args.output_root / "validation_predictions.csv"
    manifest_path = args.output_root / "activation_manifest.csv"
    checkpoint_path = args.output_root / "bas_b22_foreground_control_probe_fp16.pt"
    for path, expected in {
        history_path: summary["training_history_sha256"],
        predictions_path: summary["validation_predictions_sha256"],
        manifest_path: summary["activation_manifest_sha256"],
        checkpoint_path: summary["checkpoint_sha256"],
    }.items():
        if sha256_file(path) != expected:
            raise ValueError(f"B2.2 artifact changed: {path}")

    history = _rows(history_path)
    if len(history) != 5 or [int(row["epoch"]) for row in history] != list(range(1, 6)):
        raise ValueError("B2.2 training history mismatch")
    predictions = _rows(predictions_path)
    if len(predictions) != 371 or len({row["image_id"] for row in predictions}) != 371:
        raise ValueError("B2.2 prediction cohort mismatch")
    labels = np.asarray([int(row["tumor"]) for row in predictions], dtype=np.int64)
    probabilities = np.asarray([float(row["tumor_probability"]) for row in predictions])
    ranges = np.asarray([float(row["activation_range"]) for row in predictions])
    if int(labels.sum()) != 184 or not np.isfinite(probabilities).all():
        raise ValueError("B2.2 image-label diagnostics mismatch")
    validation = {
        **_binary_metrics(labels, probabilities),
        "activation_range_mean": float(ranges.mean()),
        "tumor_nondegenerate_activation_fraction": float(
            np.mean(ranges[labels == 1] > 1.0e-4)
        ),
    }
    for key, value in validation.items():
        if not np.isclose(value, float(summary["validation_diagnostics"][key]), atol=1e-12, rtol=0):
            raise ValueError(f"B2.2 validation diagnostic changed: {key}")

    activation_rows = _rows(manifest_path)
    if len(activation_rows) != 371 or len({row["image_id"] for row in activation_rows}) != 371:
        raise ValueError("B2.2 activation cohort mismatch")
    activations: dict[str, np.ndarray] = {}
    for row in activation_rows:
        path = args.output_root / row["activation_path"]
        if sha256_file(path) != row["activation_sha256"]:
            raise ValueError(f"B2.2 activation changed: {row['image_id']}")
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float32 or value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"invalid B2.2 activation: {row['image_id']}")
        activations[row["image_id"]] = value
    spatial = _spatial_mechanics(activations, predictions)
    for key, value in spatial.items():
        if not np.isclose(value, float(summary["spatial_mechanics_diagnostics"][key]), atol=1e-12, rtol=0):
            raise ValueError(f"B2.2 spatial mechanics changed: {key}")

    tumor_area = [
        float(row["bas_area_spearman"])
        for row in activation_rows
        if int(row["tumor"]) == 1
    ]
    if len(tumor_area) != 184:
        raise ValueError("B2.2 area cohort mismatch")
    area = {
        "tumor_bas_area_spearman_mean": float(np.mean(tumor_area)),
        "tumor_bas_area_spearman_median": float(np.median(tumor_area)),
        "tumor_bas_area_spearman_fraction_above_0_9": float(
            np.mean(np.asarray(tumor_area) > 0.9)
        ),
    }
    for key, value in area.items():
        if not np.isclose(value, float(summary["candidate_area_diagnostics"][key]), atol=1e-12, rtol=0):
            raise ValueError(f"B2.2 area diagnostic changed: {key}")

    gate = _mechanics_gate(history, validation, spatial, area)
    if gate != summary["mechanics_gate"] or bool(gate["pass"]) != bool(summary["full_training_authorized"]):
        raise ValueError("B2.2 mechanics gate does not reproduce")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment_id") != summary["experiment_id"]
        or checkpoint.get("classifier_activation") != "softplus"
        or checkpoint.get("objective") != "continuous_foreground_control_ratio"
        or checkpoint.get("source_commit") != args.expected_source_commit
        or checkpoint.get("protocol_sha256") != args.expected_protocol_sha256
        or checkpoint.get("split_sha256") != args.expected_split_sha256
        or checkpoint.get("epochs") != 5
        or checkpoint.get("mechanics_gate_pass") != gate["pass"]
        or checkpoint.get("validation_gt_read") is not False
        or checkpoint.get("test_images_read") != 0
        or checkpoint.get("test_evaluated") is not False
        or not isinstance(checkpoint.get("model_state_dict"), dict)
    ):
        raise ValueError("B2.2 checkpoint contract mismatch")

    audit = {
        "pass": True,
        "probe_summary_sha256": sha256_file(summary_path),
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "activation_maps_verified": 371,
        "validation_predictions_verified": 371,
        "mechanics_gate_pass": bool(gate["pass"]),
        "full_training_authorized": bool(gate["pass"]),
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if args.audit_output.exists():
        raise FileExistsError("B2.2 audit output already exists")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
