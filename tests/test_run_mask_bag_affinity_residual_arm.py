from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_affinity_residual_arm.py"


def test_r2_runner_is_image_label_only_and_test_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "mask_tensor",
        "annotation_name",
        'split=\"test\"',
        "test_loader",
        "candidate_quality",
        "self_guided_instance_loss",
    ):
        assert forbidden not in lowered
    assert re.search(r"\bdice\b", lowered) is None
    assert '"training_labels": "image_level_only"' in source
    assert '"epoch_selection": "fixed_final_epoch_only"' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_r2_verifies_cache_before_training_and_requires_affinity_alignment() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    training = source.index("adapter, history = train_affinity_residual_adapter(")
    for required in (
        "cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)",
        "cache, validated_cache_rows = _load_cache_records(",
        "descriptor_dim = _validate_affinity_cache(train_records + val_records)",
    ):
        assert source.index(required) < training
    assert "AFFINITY_DIM" in source
    assert "R2 cache descriptor/affinity alignment mismatch" in source


def test_r2_has_one_fixed_final_fit_without_validation_selection() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("train_affinity_residual_adapter(") == 1
    for required in (
        "args.epochs != 16",
        "args.batch_size != 16",
        "args.learning_rate != 3.0e-4",
        "args.adapter_hidden_dim != 128",
        "args.consistency_weight != 0.10",
        "args.residual_drift_weight != 1.0e-3",
        "args.seed != 42",
    ):
        assert required in source
    lowered = source.lower()
    assert "early_stop" not in lowered
    assert "validation_loss" not in lowered
    assert "best_epoch" not in lowered


def test_r2_uses_t4x2_for_complete_frozen_validation_scoring() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "shards = [val_records[::2], val_records[1::2]]" in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert "len(unordered_scored) != 371" in source
    assert '"validation_shards": [len(shards[0]), len(shards[1])]' in source


def test_r2_freezes_all_candidate_scores_and_maps_before_evaluation() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    prediction = source.index(
        "_write_validation_outputs(args, val_records, scored_val)"
    )
    freeze = source.index('freeze_path = args.output_dir / "prediction_freeze.json"')
    assert prediction < freeze
    assert '"candidate_score_manifest_sha256"' in source
    assert '"prediction_manifest_sha256"' in source
    assert '"validation_predictions": 371' in source
