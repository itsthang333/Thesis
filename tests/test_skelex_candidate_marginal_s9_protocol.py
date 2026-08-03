from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "artifacts"
    / "research_protocols"
    / "skelex_candidate_marginal_s9_v1.json"
)


def _canonical_lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_s9_protocol_freezes_high_capacity_one_shot_and_safety() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["status"] == "FROZEN_PRELAUNCH"
    assert payload["experiment_id"] == (
        "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
    )
    assert payload["scientific_source"]["commit"] == (
        "7dcd6c6f055c69f3f048a005ed2fea6177dc7ed8"
    )
    representation = payload["representation"]
    assert representation["input_size"] == 512
    assert representation["token_grid"] == [32, 32]
    assert representation["selected_hidden_layers"] == [8, 16]
    assert representation["trainable_parameters"] == 524_801
    assert representation["encoder_frozen"] is True
    assert payload["training"] == {
        "optimizer": "AdamW",
        "epochs": 32,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "checkpoint_selection": "fixed final epoch only; no validation selection",
        "early_stopping": False,
        "augmentation": False,
        "hyperparameter_sweep": False,
    }
    assert payload["safety"] == {
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "collaborator_output_accessed": False,
    }


def test_s9_protocol_closes_every_declared_source() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for section in ("canonical_lf_source_hashes", "post_freeze_source_hashes"):
        for relative, expected in payload[section].items():
            assert _canonical_lf_sha256(ROOT / relative) == expected, relative


def test_s9_protocol_has_no_validation_gt_selector_or_sweep() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    boundary = payload["image_label_only_boundary"]
    assert boundary["annotation_paths_resolved"] is False
    assert boundary["validation_size_group_used_for_training_or_selection"] is False
    assert payload["objective"]["current_selector_winner_or_pseudo_instance_target"] is False
    assert payload["finite_arms"]["weight_or_threshold_alternatives"] is False
    assert payload["post_freeze_evaluation"]["post_hoc_rescue_or_sweep"] is False
