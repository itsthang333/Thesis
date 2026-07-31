from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_orbit_relation_arm.py"


def test_r4_runner_is_image_label_only_and_test_locked() -> None:
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
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_r4_has_one_fixed_fit_and_no_consistency_alternative() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("train_orbit_relation_adapter(") == 1
    for required in (
        "args.epochs != 16",
        "args.batch_size != 16",
        "args.learning_rate != 3.0e-4",
        "args.hidden_dim != 128",
        "args.instance_loss_weight != 0.25",
        "args.instance_warmup_epochs != 2",
        "args.seed != 42",
    ):
        assert required in source
    lowered = source.lower()
    assert "consistency_weight" not in lowered
    assert "early_stop" not in lowered
    assert "validation_loss" not in lowered
    assert "best_epoch" not in lowered


def test_r4_uses_t4x2_and_freezes_reproducible_gt_blind_gates() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "shards = [val_records[::2], val_records[1::2]]" in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    outputs = source.index("_write_validation_outputs(\n        args, val_records, scored_val")
    diagnostics = source.index("_write_gt_blind_diagnostics(args.output_dir, scored_val)")
    freeze = source.index('freeze_path = args.output_dir / "prediction_freeze.json"')
    assert outputs < diagnostics < freeze
    assert '"view_swap_expected_records": 371' in source
    assert '"gt_blind_diagnostics_sha256"' in source
    assert '"validation_predictions": 371' in source
