from __future__ import annotations

import ast
from pathlib import Path
import re

import numpy as np

from run_mask_bag_same_family_graph_s3_arm import _float32_scalar_identity


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_same_family_graph_s3_arm.py"


def test_s3_runner_is_gt_free_fit_free_and_test_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "annotation_name",
        'split="test"',
        "candidate_quality",
        "optimizer",
        "backward(",
    ):
        assert forbidden not in lowered
    assert re.search(r"\bdice\b", lowered) is None
    assert '"arm_fit": "none_fixed_operator"' in source
    assert '"training_labels": "image_level_only"' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s3_runner_freezes_one_exact_graph_contract() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "args.minimum_iou != 0.25",
        "args.minimum_containment != 0.50",
        "args.graph_alpha != 0.50",
        "args.graph_iterations != 10",
        "args.batch_size != 16",
        "SameFamilyGraphConfig(",
    ):
        assert required in source
    for forbidden in ("sweep", "early_stop", "best_epoch", "validation_loss"):
        assert forbidden not in source.lower()


def test_s3_runner_uses_t4x2_and_freezes_physical_gt_blind_gates() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "shards = [val_records[::2], val_records[1::2]]" in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    identity = source.index("identity_path, identity_summary = _write_pregraph_identity_audit(")
    outputs = source.index("prediction_manifest_sha256, score_manifest_sha256 = _write_validation_outputs(")
    diagnostics = source.index("diagnostics_path = _write_gt_blind_diagnostics")
    freeze = source.index('freeze_path = args.output_dir / "prediction_freeze.json"')
    assert identity < outputs < diagnostics < freeze
    for required in (
        '"view_swap_exact_records"',
        '"alpha_zero_identity_exact_records"',
        '"graph_symmetric_records"',
        '"cross_family_edge_count"',
        '"non_self_edge_count"',
        '"isolated_logits_exact_records"',
        '"accepted_baseline_identity_verified_records"',
        '"gt_blind_gate_pass"',
    ):
        assert required in source


def test_s3_scalar_identity_is_ulp_bounded_and_fail_closed() -> None:
    accepted = float(np.float32(16.0))
    spacing = abs(float(np.spacing(np.float32(accepted))))
    within = _float32_scalar_identity(accepted + 4 * spacing, accepted)
    outside = _float32_scalar_identity(accepted + 8 * spacing, accepted)
    assert within["tolerance"] == 4 * spacing
    assert within["within_tolerance"] == 1
    assert outside["within_tolerance"] == 0


def test_s3_identity_audit_serializes_numeric_evidence() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "NUMERIC_IDENTITY_ADDENDUM_SHA256",
        "SCALAR_IDENTITY_MIN_ABS_TOLERANCE = 2.0e-6",
        "SCALAR_IDENTITY_MAX_FLOAT32_ULPS = 4",
        '"accepted_selected_logit_abs_delta"',
        '"accepted_bag_logit_tolerance"',
        '"accepted_bag_probability_within_tolerance"',
        '"accepted_row_identity_pass"',
    ):
        assert required in source
