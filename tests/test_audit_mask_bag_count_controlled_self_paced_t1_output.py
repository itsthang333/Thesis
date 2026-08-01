from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import audit_mask_bag_count_controlled_self_paced_t1_output as audit


PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "research_protocols"
    / "rad_dino_mask_bag_count_controlled_self_paced_t1_v1.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_t1_auditor_is_gt_blind_and_pins_frozen_protocol() -> None:
    source = (PROJECT / "audit_mask_bag_count_controlled_self_paced_t1_output.py").read_text(
        encoding="utf-8"
    )
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert audit.PROTOCOL_SHA256 == _sha(PROTOCOL)
    assert audit.SOURCE_COMMIT == "c7f0937d515ded9bbd8928a2236cbe44b7a25f79"
    assert audit.COUNT_SPEARMAN_CEILING == 0.5013777759365411
    assert audit.EXPECTED_TRAINING_CONFIG == {
        "fold_count": 5,
        "producer_epochs": 16,
        "producer_batch_size": 16,
        "producer_learning_rate": 0.0003,
        "producer_weight_decay": 0.0001,
        "view_consistency_weight": 0.1,
        "count_independence_weight": 1.0,
        "maximum_count_spearman": 0.5013777759365411,
        "minimum_oof_auroc": 0.75,
        "minimum_view_agreement": 0.6,
        "pace_fractions": [0.2, 0.4, 0.6],
        "consumer_epochs": 12,
        "consumer_learning_rate": 0.0001,
        "supervised_contrastive_weight": 0.25,
        "contrastive_temperature": 0.1,
        "residual_hidden_dim": 128,
        "seed": 42,
    }
    checkpoint_config = dict(audit.EXPECTED_TRAINING_CONFIG)
    checkpoint_config["pace_fractions"] = (0.2, 0.4, 0.6)
    assert audit._training_config_matches(checkpoint_config)
    assert audit.EXPECTED_MODEL_CONFIG["bag_temperature"] == 0.2
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("from ", "import "))
    )
    assert "evaluate_mask_bag_selector_arm" not in import_lines
    assert "mask_bag_count_controlled_self_paced" not in import_lines
    assert protocol["producer_gate_before_targets"] == {
        "oof_records": 2981,
        "group_overlap": 0,
        "absolute_candidate_count_probability_spearman_maximum": 0.5013777759365411,
        "minimum_image_auroc": 0.75,
        "minimum_original_flip_top1_agreement": 0.6,
        "failure_action": "reject T1 before target construction and confirmation-residual optimizer; no tolerance, lambda, fold, epoch or threshold rescue",
    }
    assert protocol["non_duplicate_scope"]["collaborator_exploratory_fusion_adopted"] is False
    assert protocol["post_freeze_evaluation"]["adoption_gate"]["operational_goals"] == {
        "overall": 0.34024039,
        "small": 0.17895493,
        "medium": 0.51244178,
        "large": 0.49370336,
    }


def test_t1_independent_rank_metrics_handle_ties() -> None:
    assert audit._absolute_spearman([1, 1, 2, 3], [0.2, 0.2, 0.7, 0.9]) == pytest.approx(1.0)
    assert audit._binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_t1_candidate_score_loader_accepts_shared_versioned_contract(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        candidate_indices=np.asarray([1, 4], dtype=np.int64),
        candidate_logits=np.asarray([-0.5, 0.25], dtype=np.float32),
    )
    indices, logits = audit._load_candidate_score_payload(path, image_id="fixture")
    assert np.array_equal(indices, np.asarray([1, 4], dtype=np.int64))
    assert np.array_equal(logits, np.asarray([-0.5, 0.25], dtype=np.float32))


def test_t1_candidate_score_loader_rejects_missing_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    np.savez_compressed(
        path,
        candidate_indices=np.asarray([1], dtype=np.int64),
        candidate_logits=np.asarray([0.25], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="candidate-score schema mismatch"):
        audit._load_candidate_score_payload(path, image_id="fixture")


def _target_fixture(root: Path, *, corrupt_weight: bool) -> tuple[dict, dict, dict]:
    target_root = root / "self_paced_targets"
    cache_rows = {
        "n0": {"image_id": "n0", "group_id": "ng0", "split": "train", "tumor": "0"},
        "n1": {"image_id": "n1", "group_id": "ng1", "split": "train", "tumor": "0"},
        "p0": {"image_id": "p0", "group_id": "pg0", "split": "train", "tumor": "1"},
        "p1": {"image_id": "p1", "group_id": "pg1", "split": "train", "tumor": "1"},
    }
    oof = {
        "n0": {
            "heldout_fold": 0,
            "family_ids": np.asarray([3, 3, 7]),
            "original_logits": np.asarray([-1.0, -0.5, -0.2]),
            "flipped_logits": np.asarray([-1.1, -0.4, -0.1]),
        },
        "n1": {
            "heldout_fold": 1,
            "family_ids": np.asarray([4, 4]),
            "original_logits": np.asarray([-0.2, -0.3]),
            "flipped_logits": np.asarray([-0.1, -0.4]),
        },
        "p0": {
            "heldout_fold": 2,
            "family_ids": np.asarray([1, 1]),
            "original_logits": np.asarray([3.0, 1.0]),
            "flipped_logits": np.asarray([2.5, 1.2]),
        },
        "p1": {
            "heldout_fold": 3,
            "family_ids": np.asarray([2, 2]),
            "original_logits": np.asarray([0.1, 2.0]),
            "flipped_logits": np.asarray([0.2, 1.8]),
        },
    }
    negative = [
        {"image_id": "n0", "candidate_index": 0, "target": 0, "weight": 0.125, "family_id": "3", "producer_fold": 0},
        {"image_id": "n0", "candidate_index": 1, "target": 0, "weight": 0.125, "family_id": "3", "producer_fold": 0},
        {"image_id": "n0", "candidate_index": 2, "target": 0, "weight": 0.25, "family_id": "7", "producer_fold": 0},
        {"image_id": "n1", "candidate_index": 0, "target": 0, "weight": 0.25, "family_id": "4", "producer_fold": 1},
        {"image_id": "n1", "candidate_index": 1, "target": 0, "weight": 0.25, "family_id": "4", "producer_fold": 1},
    ]
    if corrupt_weight:
        negative[0]["weight"] = 0.2
    positive_1 = [
        {"image_id": "p0", "candidate_index": 0, "margin": 1.3, "producer_fold": 2, "target": 1, "weight": 1.0}
    ]
    positive_2 = [dict(positive_1[0])]
    positive_3 = [
        {**positive_1[0], "weight": 0.5},
        {"image_id": "p1", "candidate_index": 1, "margin": 1.6, "producer_fold": 3, "target": 1, "weight": 0.5},
    ]
    # The independent rule orders p1 before p0 because 1.6 > 1.3.
    positive_1 = [dict(positive_3[1], weight=1.0)]
    positive_2 = [dict(positive_3[1], weight=1.0)]
    positive_3 = [dict(positive_3[1]), dict(positive_3[0])]
    files = {
        "negative_targets.csv": negative,
        "positive_targets_stage_1.csv": positive_1,
        "positive_targets_stage_2.csv": positive_2,
        "positive_targets_stage_3.csv": positive_3,
    }
    hashes = {}
    for name, rows in files.items():
        path = target_root / name
        _write_csv(path, rows)
        hashes[name] = _sha(path)
    target_freeze = {
        "producer_gate_sha256": "producer-gate",
        "eligible_positive_bags": 2,
        "pace_fractions": [0.2, 0.4, 0.6],
        "target_hashes": hashes,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    target_freeze_path = target_root / "target_freeze.json"
    target_freeze_path.write_text(json.dumps(target_freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "producer_gate_audit_sha256": "producer-gate",
        "target_freeze_sha256": _sha(target_freeze_path),
    }
    return freeze, cache_rows, oof


def test_t1_target_auditor_reconstructs_nested_targets_and_weights(tmp_path: Path) -> None:
    freeze, cache_rows, oof = _target_fixture(tmp_path, corrupt_weight=False)
    result = audit._verify_targets(tmp_path, freeze, cache_rows, oof)
    assert result["eligible_positive_bags"] == 2
    assert result["negative_candidates"] == 5
    assert result["positive_stage_counts"] == [1, 1, 2]


def test_t1_target_auditor_rejects_mutated_hierarchical_weight(tmp_path: Path) -> None:
    freeze, cache_rows, oof = _target_fixture(tmp_path, corrupt_weight=True)
    with pytest.raises(ValueError, match="negative target weight"):
        audit._verify_targets(tmp_path, freeze, cache_rows, oof)


def test_t1_launch_binding_must_pin_every_runtime_source(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runtime_hashes = {
        path: digest
        for path, digest in protocol["canonical_lf_source_hashes"].items()
        if path not in audit.POST_FREEZE_ONLY_SOURCE_PATHS
    }
    binding = {
        "schema_version": 1,
        "experiment_id": audit.EXPERIMENT_ID,
        "kernel": audit.KERNEL,
        "kernel_version": 1,
        "scientific_source_commit": audit.SOURCE_COMMIT,
        "protocol_sha256": audit.PROTOCOL_SHA256,
        "source_hashes": runtime_hashes,
        "checkout_commit": "a" * 40,
        "bound_wrapper_sha256": "b" * 64,
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    assert audit._verify_binding(path, protocol)["source_hashes"] == runtime_hashes
    binding["source_hashes"].pop(next(iter(runtime_hashes)))
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="launch binding"):
        audit._verify_binding(path, protocol)
