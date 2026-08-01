from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from project.audit_bas_candidate_descriptor_b1_output import _activation_scores, _rank
from project.models.mask_bag_selector_cache import pack_candidate_masks
from project.run_bas_candidate_descriptor_b1 import _score_arms
from project.models.mask_bag_selector_cache_io import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_score_arms_freezes_transferred_and_three_way_borda(tmp_path) -> None:
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    candidate_path = candidate_root / "x.npz"
    np.savez_compressed(
        candidate_path,
        selection_scores=np.asarray([1.0, 2.0, 0.0], dtype=np.float32),
    )
    candidate_sha = sha256_file(candidate_path)
    masks = np.asarray(
        [
            [[1, 1], [0, 0]],
            [[1, 1], [1, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )
    records = [
        {
            "image_id": "x.png",
            "group_id": "g",
            "label": 1,
            "candidate_payload_sha256": candidate_sha,
            "candidate_indices": np.asarray([0, 1, 2], dtype=np.int32),
            "packed_masks": pack_candidate_masks(masks),
        }
    ]
    base_scored = [
        {
            "base_candidate_logits": np.asarray([2.0, 1.0, 0.0], dtype=np.float32)
        }
    ]
    baseline_rows = [
        {"image_id": "x.png", "bag_logit": "0.25", "bag_probability": "0.6"}
    ]
    candidate_rows = {
        "x": {"diagnostic_path": "x.npz", "diagnostic_sha256": candidate_sha}
    }
    arms, manifest_sha, diagnostics = _score_arms(
        tmp_path / "output",
        records,
        base_scored,
        baseline_rows,
        {"x.png": np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)},
        candidate_root,
        candidate_rows,
    )
    transferred = arms["transferred_geometry_upstream"][0]["candidate_logits"]
    three_way = arms["three_way_geometry_upstream_bas"][0]["candidate_logits"]
    assert np.allclose(transferred, np.asarray([0.75, 0.75, 0.0]))
    assert np.allclose(three_way, np.asarray([5.0 / 6.0, 2.0 / 3.0, 0.0]))
    assert arms["three_way_geometry_upstream_bas"][0]["bag_probability"] == 0.6
    assert len(manifest_sha) == 64
    assert diagnostics["correlation_images"] == 1
    assert (tmp_path / "output" / "activation_evidence" / "activation_manifest.csv").is_file()


def test_independent_auditor_reproduces_ties_and_activation_evidence() -> None:
    assert np.allclose(_rank(np.asarray([3.0, 1.0, 3.0])), [0.75, 0.0, 0.75])
    activation = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    masks = np.asarray(
        [
            [[1, 1], [0, 0]],
            [[1, 1], [1, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.float32,
    )
    coverage, purity, harmonic, ranks = _activation_scores(activation, masks)
    assert np.allclose(coverage, [1.0, 1.0, 0.0])
    assert np.allclose(purity, [1.0, 0.5, 0.0])
    assert np.allclose(harmonic, [1.0, 2.0 / 3.0, 0.0])
    assert np.allclose(ranks, [1.0, 0.5, 0.0])


def test_b1_protocol_closes_every_declared_source() -> None:
    protocol_path = (
        REPO_ROOT
        / "artifacts"
        / "research_protocols"
        / "bas_candidate_descriptor_b1_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["status"] == "STATIC_PREDECLARED_NO_CLAIM_NO_BINDING_NO_LAUNCH"
    assert protocol["training"]["supervision"] == "binary image-level normal/tumor labels only"
    assert protocol["finite_arms"]["weight_or_threshold_alternatives"] is False
    for relative, expected in protocol["canonical_lf_source_hashes"].items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
