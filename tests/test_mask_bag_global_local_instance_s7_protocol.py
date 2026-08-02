from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1.json"
)


def _canonical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s7_protocol_is_frozen_safe_and_hash_bound() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["status"] == "FROZEN_PRELAUNCH"
    assert payload["experiment_id"] == (
        "EXP-20260802-codex-s7-global-local-instance-v1"
    )
    assert payload["registration_commit"] == (
        "d316269b141664b9881eea601491d0fe45227d80"
    )
    assert payload["scientific_source"]["real_data_execution_before_protocol_freeze"] is False
    assert payload["matched_pair"]["shared_accepted_bag_probability"] is True
    assert payload["training"] == {
        "epochs": 40,
        "batch_size": 16,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "seed": 42,
        "hidden_dim": 128,
        "dropout": 0.1,
        "bag_temperature": 0.2,
        "start_positive_mass": 0.5,
        "target_positive_mass": 0.15,
        "mass_transition_epochs": 20,
        "projection_iterations": 96,
        "consistency_weight": 0.1,
        "residual_drift_weight": 0.001,
        "candidate_weight": "equal image then equal family then equal candidate",
        "epoch_selection": "fixed final epoch only",
        "early_stopping": False,
        "hyperparameter_sweep": False,
        "training_labels": "binary image-level only",
        "bag_classification_loss": False,
    }
    assert payload["pre_gt_contract"]["target_snapshots"] == 40
    assert payload["pre_gt_contract"]["accepted_bag_probability_preserved"] is True
    assert payload["post_freeze_evaluation"]["post_hoc_rescue_or_sweep"] is False
    assert payload["post_freeze_evaluation"]["failure_analysis_required_before_successor"] is True
    assert payload["safety"] == {
        "validation_gt_read_before_prediction_freeze": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "validation_subtype_used": False,
        "collaborator_output_accessed": False,
    }
    for section in ("canonical_lf_source_hashes", "post_freeze_only_source_hashes"):
        for relative, expected in payload[section].items():
            assert _canonical_sha256(ROOT / relative) == expected, relative
    source_commit = payload["scientific_source"]["commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=ROOT,
        check=True,
    )
