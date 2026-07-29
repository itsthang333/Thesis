from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_family_balanced_pair.py"


def test_s1_runner_is_image_label_only_and_test_locked() -> None:
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
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s1_is_a_matched_pair_with_only_pooling_changed() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "for device_index, pool_mode in enumerate(POOL_MODES)" in source
    assert "initial_state," in source
    assert '"sole_changed_variable": "standard_vs_family_balanced_bag_pool"' in source
    for matched in (
        '"descriptor_cache"',
        '"frozen_baseline"',
        '"adapter_architecture"',
        '"adapter_initial_state"',
        '"batch_order"',
        '"optimizer"',
        '"epochs"',
        '"loss_weights"',
        '"validation_cohort"',
    ):
        assert matched in source
    assert "maximum_initial_probe_delta > 5.0e-6" in source
    assert '"cross_device_initial_candidate_logit_max_delta"' in source


def test_s1_verifies_all_inputs_before_parallel_training() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    training = source.index("with ThreadPoolExecutor(max_workers=2) as executor:")
    for required in (
        "cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)",
        "cache, validated_cache_rows = _load_cache_records(",
        "initial_state = _initial_residual_state(",
        "initial_state_sha256 = sha256_file(initial_state_path)",
    ):
        assert source.index(required) < training


def test_s1_uses_t4x2_and_fixed_final_only_contract() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert '"parallel_training_workers": 2' in source
    for required in (
        "args.epochs != 16",
        "args.batch_size != 16",
        "args.learning_rate != 3.0e-4",
        "args.hidden_dim != 128",
        "args.consistency_weight != 0.10",
        "args.residual_drift_weight != 1.0e-3",
        "args.seed != 42",
    ):
        assert required in source
    assert '"epoch_selection": "fixed_final_epoch_only"' in source


def test_both_arms_freeze_scores_and_maps_before_pair_freeze() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    arm_output = source.index("_write_validation_outputs(")
    arm_freeze = source.index('freeze_path = arm_root / "prediction_freeze.json"')
    pair_freeze = source.index(
        'pair_freeze_path = args.output_dir / "pair_prediction_freeze.json"'
    )
    assert arm_output < arm_freeze < pair_freeze
    assert '"candidate_score_manifest_sha256"' in source
    assert '"prediction_manifest_sha256"' in source
    assert '"validation_predictions": 371' in source
