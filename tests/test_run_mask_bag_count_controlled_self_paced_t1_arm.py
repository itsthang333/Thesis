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
    from run_mask_bag_count_controlled_self_paced_t1_arm import (
        _frozen_training_config,
        _write_gt_blind_diagnostics,
        _write_target_bundle,
    )


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "fold_count": 5,
        "producer_epochs": 16,
        "producer_batch_size": 16,
        "producer_learning_rate": 3.0e-4,
        "producer_weight_decay": 1.0e-4,
        "view_consistency_weight": 0.10,
        "count_independence_weight": 1.0,
        "maximum_count_spearman": 0.5013777759365411,
        "minimum_oof_auroc": 0.75,
        "minimum_view_agreement": 0.60,
        "pace_fractions": "0.20,0.40,0.60",
        "consumer_epochs": 12,
        "consumer_learning_rate": 1.0e-4,
        "supervised_contrastive_weight": 0.25,
        "contrastive_temperature": 0.10,
        "residual_hidden_dim": 128,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_t1_runner_is_gt_blind_and_gates_targets_before_consumer() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "run_mask_bag_count_controlled_self_paced_t1_arm.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "build_segmentation_dataset",
        "annotations/",
        "lesion_size",
        "size_group",
        "proposal_cluster",
    ):
        assert forbidden not in lowered
    main = source[source.index("def main()") :]
    assert main.index("producer_gate_audit.json") < main.index(
        "if producer_gate[\"producer_gate_pass\"] is not True"
    )
    assert main.index("if producer_gate[\"producer_gate_pass\"] is not True") < main.index(
        "build_self_paced_targets("
    )
    assert main.index("_write_target_bundle(") < main.index(
        "train_self_paced_consumer("
    )
    assert main.index("audit_consumer_residual_identity(") < main.index(
        "train_self_paced_consumer("
    )
    assert main.index("_write_gt_blind_diagnostics(") < main.index(
        "_write_validation_outputs("
    )
    assert "torch.cuda.device_count() != 2" in source
    assert 'all("T4" in name for name in device_names)' in source
    assert '"training_labels": "image_level_only"' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_runner_freezes_every_control() -> None:
    config = _frozen_training_config(_args())
    assert config.fold_count == 5
    assert config.count_independence_weight == 1.0
    assert config.maximum_count_spearman == 0.5013777759365411
    assert config.pace_fractions == (0.20, 0.40, 0.60)
    assert config.contrastive_temperature == 0.10
    with pytest.raises(ValueError, match="frozen finite contract"):
        _frozen_training_config(_args(count_independence_weight=0.5))


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_final_count_gate_fails_before_prediction_manifest(tmp_path: Path) -> None:
    records = [
        {"image_id": f"i{index}", "candidate_indices": list(range(index + 1))}
        for index in range(3)
    ]
    scored = []
    for index in range(3):
        count = index + 1
        zero = np.zeros(count, dtype=np.float32)
        scored.append(
            {
                "image_id": f"i{index}",
                "candidate_count": count,
                "bag_probability": 0.1 * (index + 1),
                "selected_view_agreement": True,
                "candidate_logits": zero,
                "original_base_logits": zero,
                "flipped_base_logits": zero,
                "original_residual_logits": zero,
                "flipped_residual_logits": zero,
                "original_candidate_logits": zero,
                "flipped_candidate_logits": zero,
            }
        )
    with pytest.raises(RuntimeError, match="count/probability"):
        _write_gt_blind_diagnostics(tmp_path, records, scored, ceiling=0.5)
    assert not (tmp_path / "gt_blind_diagnostics.csv").exists()
    assert not (tmp_path / "predictions").exists()


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_t1_target_bundle_physically_freezes_each_stage(tmp_path: Path) -> None:
    producer_gate = tmp_path / "producer_gate.json"
    producer_gate.write_text("{}\n", encoding="utf-8")
    bundle = {
        "eligible_positive_bags": 3,
        "negative_targets": [
            {
                "image_id": "n",
                "candidate_index": 0,
                "target": 0,
                "weight": 1.0,
                "family_id": "0",
                "producer_fold": 0,
            }
        ],
        "stages": [
            {
                "stage": stage,
                "fraction": fraction,
                "positive_targets": [
                    {
                        "image_id": f"p{stage}",
                        "candidate_index": 0,
                        "margin": 1.0,
                        "producer_fold": 1,
                        "target": 1,
                        "weight": 1.0,
                    }
                ],
            }
            for stage, fraction in enumerate((0.2, 0.4, 0.6), start=1)
        ],
    }
    freeze = _write_target_bundle(tmp_path, bundle, producer_gate)
    assert freeze["pace_fractions"] == [0.2, 0.4, 0.6]
    assert len(freeze["target_hashes"]) == 4
    assert (tmp_path / "self_paced_targets" / "target_freeze.json").is_file()
