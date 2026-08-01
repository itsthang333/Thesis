from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from audit_mask_bag_proposal_cluster_s4_output import (
    KERNEL,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    _recompute_clusters,
    _verify_binding,
    _verify_score_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_proposal_cluster_s4_output.py"
PROTOCOL = (
    ROOT
    / "artifacts"
    / "research_protocols"
    / "rad_dino_mask_bag_proposal_cluster_s4_v1.json"
)


def test_s4_auditor_is_gt_blind_and_pins_the_frozen_protocol() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "build_segmentation_dataset",
        "mask_tensor",
        "size_group",
        'split="test"',
    ):
        assert forbidden not in lowered
    assert sha256_file(PROTOCOL) == PROTOCOL_SHA256
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["scientific_source"]["commit"] == SOURCE_COMMIT
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_s4_auditor_recomputes_the_exact_disjoint_cluster_rule() -> None:
    logits = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    iou = np.eye(5, dtype=np.float32)
    containment = np.eye(5, dtype=np.float32)
    iou[0, 1] = iou[1, 0] = 0.5
    containment[2, 3] = 0.8
    clusters, valid, seeds = _recompute_clusters(logits, iou, containment)
    assert valid.tolist() == [1, 1, 1, 0]
    assert seeds.tolist() == [0, 2, 4, -1]
    assert clusters[0].tolist() == [1, 1, 0, 0, 0]
    assert clusters[1].tolist() == [0, 0, 1, 1, 0]
    assert np.all(clusters.sum(axis=0) == 1)


def _write_score_fixture(root: Path, *, corrupt_seed: bool) -> Path:
    score_root = root / "scores"
    score_root.mkdir()
    original = np.asarray([0.2, 0.8], dtype=np.float32)
    flipped = np.asarray([0.1, 0.7], dtype=np.float32)
    conservative = np.minimum(original, flipped)
    if corrupt_seed:
        conservative[0] += 0.01
    payload = score_root / "row.npz"
    np.savez_compressed(
        payload,
        candidate_indices=np.asarray([1, 3], dtype=np.int32),
        original_logits=original,
        flipped_logits=flipped,
        conservative_seed_logits=conservative,
    )
    manifest = root / "score_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "group_id",
                "image_label",
                "heldout_fold",
                "selected_view_agreement",
                "candidate_count",
                "bag_probability",
                "payload_path",
                "payload_sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "image.jpeg",
                "group_id": "group",
                "image_label": 1,
                "heldout_fold": 0,
                "selected_view_agreement": 1,
                "candidate_count": 2,
                "bag_probability": 0.7,
                "payload_path": payload.name,
                "payload_sha256": sha256_file(payload),
            }
        )
    return manifest


def test_s4_auditor_rejects_mutated_conservative_teacher_seed(tmp_path: Path) -> None:
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    valid_manifest = _write_score_fixture(valid_root, corrupt_seed=False)
    rows, _bytes = _verify_score_manifest(
        valid_manifest,
        valid_root / "scores",
        expected_rows=1,
        expected_fold=0,
    )
    assert set(rows) == {"image.jpeg"}
    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    corrupt_manifest = _write_score_fixture(corrupt_root, corrupt_seed=True)
    with pytest.raises(ValueError, match="payload content"):
        _verify_score_manifest(
            corrupt_manifest,
            corrupt_root / "scores",
            expected_rows=1,
            expected_fold=0,
        )


def test_s4_launch_binding_must_pin_every_runtime_source(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    postfreeze = {
        "project/evaluate_mask_bag_selector_arm.py",
        "project/models/mask_bag_ranking_diagnostics.py",
    }
    binding = {
        "schema_version": 1,
        "experiment_id": "EXP-20260801-codex-s4-oof-proposal-cluster-v1",
        "kernel": KERNEL,
        "kernel_version": 1,
        "checkout_commit": "a" * 40,
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bound_wrapper_sha256": "b" * 64,
        "source_hashes": {
            path: digest
            for path, digest in protocol["canonical_lf_source_hashes"].items()
            if path not in postfreeze
        },
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    assert _verify_binding(path, protocol)["kernel_version"] == 1
    binding["protocol_sha256"] = "0" * 64
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="binding contract"):
        _verify_binding(path, protocol)
