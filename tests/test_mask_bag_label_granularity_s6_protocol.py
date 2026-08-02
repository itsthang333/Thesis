from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "artifacts"
    / "research_protocols"
    / "rad_dino_mask_bag_label_granularity_s6_v1.json"
)
CORRECTION = (
    ROOT
    / "artifacts"
    / "research_protocols"
    / "rad_dino_mask_bag_label_granularity_s6_v1_map_audit_numeric_correction.json"
)


def _canonical_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s6_protocol_closes_scientific_and_postfreeze_sources() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    overrides = correction["allowed_canonical_lf_source_overrides"]
    assert protocol["status"] == "FROZEN_PRELAUNCH"
    assert protocol["experiment_id"] == (
        "EXP-20260802-codex-s6-label-granularity-mil-v1"
    )
    for section in ("canonical_lf_source_hashes", "post_freeze_only_source_hashes"):
        for relative, expected in protocol[section].items():
            override = overrides.get(relative)
            if override is None:
                assert _canonical_lf_sha256(ROOT / relative) == expected, relative
            else:
                assert section == "canonical_lf_source_hashes"
                assert override["protocol_v1_sha256"] == expected
                assert _canonical_lf_sha256(ROOT / relative) == override[
                    "corrected_sha256"
                ], relative


def test_s6_protocol_is_finite_prediction_first_and_image_label_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["matched_pair"]["arms"] == [
        "coarse_control",
        "hierarchical_entropy_routed",
    ]
    assert protocol["training"]["hyperparameter_sweep"] is False
    assert protocol["training"]["early_stopping"] is False
    assert protocol["training"]["training_labels"] == "image-level only"
    assert protocol["matched_pair"]["candidate_target"] is None
    assert protocol["matched_pair"]["instance_winner_target"] is None
    assert protocol["pre_gt_contract"][
        "prediction_pair_physically_frozen_before_validation_segmentation_gt"
    ] is True
    assert protocol["safety"] == {
        "validation_gt_read_before_prediction_freeze": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "validation_subtype_label_used_for_routing": False,
        "collaborator_output_accessed": False,
    }
