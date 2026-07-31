from __future__ import annotations

import ast
from pathlib import Path
import re


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
        '"accepted_baseline_identity_exact_records"',
        '"gt_blind_gate_pass"',
    ):
        assert required in source
