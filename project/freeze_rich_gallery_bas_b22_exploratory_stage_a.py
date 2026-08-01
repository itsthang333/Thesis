from __future__ import annotations

"""Freeze one non-promotable B2.2 candidate-fusion diagnostic.

This consumer performs no training and never opens spatial annotations.  It
uses the already audited B2.2 activation maps and the five variants frozen by
the original BAS-B2 design.  The mechanics gate is intentionally recorded as
failed; Stage B is authorized only as failure analysis, never as promotion.
"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_bas_candidate_descriptor_b1 import (
    VARIANTS,
    _freeze_selections,
    _verify_g1_stage_a,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--g1-stage-a-root", type=Path, required=True)
    parser.add_argument("--expected-g1-stage-a-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--b22-output-root", type=Path, required=True)
    parser.add_argument("--b22-independent-audit", type=Path, required=True)
    parser.add_argument("--expected-b22-audit-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_audited_activations(
    output_root: Path,
    audit_path: Path,
    *,
    expected_audit_sha256: str,
    expected_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    if sha256_file(audit_path) != expected_audit_sha256:
        raise ValueError("B2.2 independent-audit SHA-256 mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("pass") is not True
        or audit.get("activation_maps_verified") != 371
        or audit.get("mechanics_gate_pass") is not False
        or audit.get("validation_gt_read") is not False
        or audit.get("test_images_read") != 0
        or audit.get("test_evaluated") is not False
    ):
        raise ValueError("B2.2 audit is not the expected locked mechanics failure")
    summary_path = output_root / "probe_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("stage")
        != "rich_gallery_bas_b22_foreground_control_mechanics_probe_v1"
        or summary.get("mechanics_gate", {}).get("pass") is not False
        or summary.get("full_training_authorized") is not False
        or summary.get("validation_gt_read") is not False
        or summary.get("test_images_read") != 0
    ):
        raise ValueError("B2.2 probe-summary contract mismatch")
    manifest_path = output_root / "activation_manifest.csv"
    if sha256_file(manifest_path) != summary["activation_manifest_sha256"]:
        raise ValueError("B2.2 activation manifest changed")
    rows = _read_csv(manifest_path)
    if len(rows) != 371 or {row["image_id"] for row in rows} != expected_ids:
        raise ValueError("B2.2 activation cohort mismatch")
    activations: dict[str, np.ndarray] = {}
    for row in rows:
        path = output_root / row["activation_path"]
        if sha256_file(path) != row["activation_sha256"]:
            raise ValueError(f"B2.2 activation changed: {row['image_id']}")
        values = np.load(path, allow_pickle=False).astype(np.float32)
        if values.shape != (56, 56) or not np.isfinite(values).all():
            raise ValueError(f"invalid B2.2 activation: {row['image_id']}")
        activations[row["image_id"]] = values
    return activations, audit, summary


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("B2.2 exploratory Stage-A output must not exist")
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise ValueError("canonical validation cohort mismatch")
    expected_ids = {row["image_id"] for row in val_rows}
    activations, audit, summary = _load_audited_activations(
        args.b22_output_root,
        args.b22_independent_audit,
        expected_audit_sha256=args.expected_b22_audit_sha256,
        expected_ids=expected_ids,
    )
    g1_freeze, g1_rows = _verify_g1_stage_a(
        args.g1_stage_a_root,
        expected_freeze_sha256=args.expected_g1_stage_a_freeze_sha256,
        expected_split_sha256=args.expected_split_sha256,
        expected_val_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_val_pseudo_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all" or len(candidate_rows) != 371:
        raise ValueError("candidate-diagnostic cohort mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    consumer = SimpleNamespace(
        output_dir=args.output_dir,
        g1_stage_a_root=args.g1_stage_a_root,
        val_candidate_root=args.val_candidate_root,
    )
    selection_sha, activation_sha, diagnostics = _freeze_selections(
        consumer,
        val_rows,
        g1_rows,
        candidate_rows,
        activations,
    )
    diagnostic_path = args.output_dir / "label_safe_diagnostics.json"
    diagnostic_payload = {
        **diagnostics,
        "b22_mechanics_gate_pass": False,
        "b22_final_full_ce": float(summary["mechanics_gate"]["values"]["final_full_ce"]),
        "exploratory_non_promotable": True,
        "diagnostics_do_not_block_spatial_failure_analysis": True,
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    diagnostic_path.write_text(
        json.dumps(diagnostic_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        # Keep the established transport schema so the independent B1 Stage-B
        # evaluator can verify/reproduce all selections without a fork.
        "stage": "rich_gallery_bas_b2_stage_a_v1",
        "scientific_role": "B2.2 post-freeze failure diagnostic only",
        "experiment_id": "EXP-20260802-codex-rich-gallery-bas-b22-exploratory-stage-a-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "g1_stage_a_freeze_sha256": args.expected_g1_stage_a_freeze_sha256,
        "g1_checkpoint_sha256": g1_freeze["g1_checkpoint_sha256"],
        "val_candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.expected_val_pseudo_manifest_sha256,
        "bas_checkpoint_sha256": audit["checkpoint_sha256"],
        "b22_independent_audit_sha256": args.expected_b22_audit_sha256,
        "b22_probe_summary_sha256": audit["probe_summary_sha256"],
        "selection_manifest_sha256": selection_sha,
        "activation_manifest_sha256": activation_sha,
        "label_safe_diagnostics_sha256": sha256_file(diagnostic_path),
        "validation_images": 371,
        "selection_rows": 371 * len(VARIANTS),
        "variants": list(VARIANTS),
        "baseline_reproduction_images": int(
            diagnostics["g1_upstream_baseline_reproduced"]
        ),
        "candidate_choices_frozen_before_validation_gt": True,
        "training_labels": "image_level_normal_tumor_only",
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "b22_mechanics_gate_pass": False,
        "exploratory_non_promotable": True,
        "promotion_authorized": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
