from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "run_mask_bag_normal_prototype_arm.py"


def test_runner_is_image_label_only_and_test_locked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "mask_tensor",
        "annotation_name",
        'split="test"',
        "test_loader",
        "self_guided_instance_loss",
        "candidate_quality",
    ):
        assert forbidden not in lowered
    assert re.search(r"\bdice\b", lowered) is None
    assert '"training_labels": "image_level_only"' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_runner_verifies_and_opens_every_cache_record_before_fitting() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    fit = source.index("fold_ids = assign_group_stratified_folds(")
    assert source.index("cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)") < fit
    assert source.index("validate_selector_cache_manifest(") < fit
    assert source.index("load_selector_cache_record(") < fit
    assert source.index("cache, validated_cache_rows = _load_cache_records(") < fit


def test_k_and_fold_selection_are_finite_group_excluded_and_train_only() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "PROTOTYPE_COUNTS = (8, 16, 32)" in source
    assert "FOLD_COUNT = 5" in source
    assert "fit_normal_oof_fold(" in source
    assert "assemble_normal_oof_candidate(" in source
    assert "select_prototype_count_one_standard_error(" in source
    assert "baseline_absolute_count_probability_spearman" in source
    assert "count_association_tolerance != 0.02" in source
    assert "assignment[\"fold_summary\"] != expected_fold_summary" in source


def test_oof_jobs_use_t4x2_with_preinitialized_deterministic_states() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert "jobs_by_device = [jobs[::2], jobs[1::2]]" in source
    assert "_initial_adapter_state(" in source
    assert "initial_adapter_state=initial_state" in source
    assert '"oof_parallel_workers": 2' in source
    assert 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"' in source
    assert "torch.use_deterministic_algorithms(True)" in source


def test_final_fit_uses_selected_k_and_freezes_all_candidate_scores_first() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    selected = source.index('selected_k = int(selection["selected_prototype_count"])')
    final_fit = source.index("final_prototypes, final_prototype_audit =")
    prediction = source.index("_write_validation_outputs(args, val_records, scored_val)")
    freeze = source.index('freeze_path = args.output_dir / "prediction_freeze.json"')
    assert selected < final_fit < prediction < freeze
    assert "save_candidate_score_evidence(" in source
    assert "write_candidate_score_manifest(" in source
    assert '"candidate_logit_tta": "mean_original_aligned_horizontal_flip"' in source
    assert '"validation_predictions": 371' in source


def test_fixed_scientific_controls_cannot_drift_at_runtime() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "args.epochs != 16",
        "args.batch_size != 16",
        "args.learning_rate != 3.0e-4",
        "args.prototype_temperature != 0.10",
        "args.adapter_hidden_dim != 128",
        "args.consistency_weight != 0.10",
        "args.residual_drift_weight != 1.0e-3",
    ):
        assert required in source
