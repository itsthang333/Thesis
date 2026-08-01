from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from run_mask_bag_proposal_cluster_s4_arm import (
        _audit_full_teacher_group_exclusion,
        _frozen_training_config,
        _write_gt_blind_diagnostics,
    )


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "fold_count": 5,
        "epochs": 16,
        "batch_size": 16,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "teacher_instance_loss_weight": 0.25,
        "consistency_weight": 0.10,
        "instance_warmup_epochs": 2,
        "maximum_clusters": 4,
        "minimum_iou": 0.50,
        "minimum_containment": 0.75,
        "start_temperature": 1.0,
        "end_temperature": 0.20,
        "residual_hidden_dim": 128,
        "count_probability_spearman_ceiling": 0.5013777759365411,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_s4_runner_is_prediction_first_and_serializes_clusters_before_student() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "run_mask_bag_proposal_cluster_s4_arm.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "build_segmentation_dataset",
        "annotations/",
        "lesion_size",
        "size_group",
    ):
        assert forbidden not in lowered
    main_source = source[source.index("def main()") :]
    assert main_source.index("oof_coverage_audit.json") < main_source.index(
        "train_cluster_residual("
    )
    assert main_source.index('"train", enriched_train') < main_source.index(
        "train_cluster_residual("
    )
    assert main_source.index('"val", enriched_val') < main_source.index(
        "train_cluster_residual("
    )
    assert main_source.index("oof_initial_states = {") < main_source.index(
        "ThreadPoolExecutor(max_workers=2)"
    )
    assert "torch.cuda.manual_seed_all" not in source
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert '"training_labels": "image_level_only"' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_s4_runner_freezes_every_scientific_control() -> None:
    config = _frozen_training_config(_args())
    assert config.fold_count == 5
    assert config.maximum_clusters == 4
    assert config.minimum_iou == 0.50
    assert config.minimum_containment == 0.75
    assert config.start_temperature == 1.0
    assert config.end_temperature == 0.20
    with pytest.raises(ValueError, match="frozen finite contract"):
        _frozen_training_config(_args(maximum_clusters=5))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_s4_gt_blind_count_gate_fails_before_prediction_serialization(tmp_path: Path) -> None:
    records = [
        {
            "image_id": f"i{index}",
            "teacher_selected_view_agreement": True,
            "candidate_indices": list(range(index + 1)),
        }
        for index in range(3)
    ]
    scored = [
        {
            "image_id": f"i{index}",
            "candidate_count": index + 1,
            "bag_probability": 0.1 * (index + 1),
            "final_selected_view_agreement": True,
            "cluster_count": 1,
            "cluster_member_count": 1,
            "outside_cluster_count": index,
            "outside_cluster_original_residual_exact_zero": True,
            "outside_cluster_flipped_residual_exact_zero": True,
            "candidate_logits": np.zeros(index + 1, dtype=np.float32),
            "original_base_logits": np.zeros(index + 1, dtype=np.float32),
            "flipped_base_logits": np.zeros(index + 1, dtype=np.float32),
            "original_residual_logits": np.zeros(index + 1, dtype=np.float32),
            "flipped_residual_logits": np.zeros(index + 1, dtype=np.float32),
            "original_candidate_logits": np.zeros(index + 1, dtype=np.float32),
            "flipped_candidate_logits": np.zeros(index + 1, dtype=np.float32),
            "cluster_member_flags": np.ones(index + 1, dtype=np.uint8),
        }
        for index in range(3)
    ]
    with pytest.raises(RuntimeError, match="count/probability"):
        _write_gt_blind_diagnostics(tmp_path, records, scored, ceiling=0.5)
    assert not (tmp_path / "gt_blind_diagnostics.csv").exists()


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_s4_full_teacher_rejects_any_validation_group_overlap() -> None:
    train = [{"group_id": "train-a"}, {"group_id": "train-b"}]
    val = [{"group_id": "val-c"}]
    audit = _audit_full_teacher_group_exclusion(train, val)
    assert audit["group_overlap"] == 0
    assert audit["training_group_count"] == 2
    assert audit["validation_group_count"] == 1
    with pytest.raises(RuntimeError, match="validation group"):
        _audit_full_teacher_group_exclusion(train, [{"group_id": "train-b"}])
