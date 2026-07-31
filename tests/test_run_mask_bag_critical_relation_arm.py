from __future__ import annotations

import ast
from pathlib import Path
import re

import numpy as np

from run_mask_bag_critical_relation_arm import _absolute_spearman


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_critical_relation_arm.py"


def test_r3_runner_is_image_label_only_and_test_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "annotation_name",
        'split="test"',
        "candidate_quality",
    ):
        assert forbidden not in lowered
    assert re.search(r"\bdice\b", lowered) is None
    assert '"training_labels": "image_level_only"' in source
    assert '"epoch_selection": "fixed_final_epoch_only"' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_r3_has_one_fixed_final_fit_and_pretraining_identity_audit() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    training = source.index("adapter, history = train_critical_relation_adapter(")
    assert source.count("train_critical_relation_adapter(") == 1
    assert source.index("initial_audit = {") < training
    assert source.index('"pretraining_identity_audit_sha256"') > training
    for required in (
        "args.epochs != 16",
        "args.batch_size != 16",
        "args.learning_rate != 3.0e-4",
        "args.hidden_dim != 128",
        "args.instance_loss_weight != 0.25",
        "args.consistency_weight != 0.10",
        "args.instance_warmup_epochs != 2",
        "args.seed != 42",
    ):
        assert required in source
    lowered = source.lower()
    assert "early_stop" not in lowered
    assert "validation_loss" not in lowered
    assert "best_epoch" not in lowered


def test_r3_uses_t4x2_and_freezes_before_gt_blind_decision() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "shards = [val_records[::2], val_records[1::2]]" in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert "len(unordered_scored) != 371" in source
    outputs = source.index("_write_validation_outputs(\n        args, val_records, scored_val")
    diagnostics = source.index("_write_gt_blind_diagnostics(args.output_dir, scored_val)")
    freeze = source.index('freeze_path = args.output_dir / "prediction_freeze.json"')
    assert outputs < diagnostics < freeze
    assert '"validation_predictions": 371' in source
    assert '"gt_blind_gate": gate' in source
    assert '"gt_blind_diagnostics_sha256"' in source


def test_r3_spearman_matches_average_tie_ranks() -> None:
    observed = _absolute_spearman(
        np.asarray([1, 1, 2, 3, 3], dtype=np.float64),
        np.asarray([5, 4, 3, 2, 1], dtype=np.float64),
    )
    expected = abs(float(np.corrcoef([1.5, 1.5, 3, 4.5, 4.5], [5, 4, 3, 2, 1])[0, 1]))
    assert observed == expected
