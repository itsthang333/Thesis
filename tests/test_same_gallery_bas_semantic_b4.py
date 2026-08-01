from __future__ import annotations

from pathlib import Path

import numpy as np

from project.audit_same_gallery_bas_semantic_b4_output import (
    ARMS,
    _audit_extra_evidence,
    _correlation_reference,
    _expected_arm_scores,
)
from project.models.mask_bag_selector_cache import pack_candidate_masks
from project.models.mask_bag_selector_cache_io import sha256_file
from project.run_same_gallery_bas_semantic_b4 import (
    CONTROL_ARM,
    EXPERIMENT_ID,
    SEMANTIC_ARM,
    _score_arms,
)


def test_b4_scores_only_geometry_and_bas_equal_rank(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    candidate_path = candidate_root / "x.npz"
    np.savez_compressed(
        candidate_path,
        selection_scores=np.asarray([0.0, -999.0, 999.0], dtype=np.float32),
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
        {"base_candidate_logits": np.asarray([2.0, 0.0, 1.0], dtype=np.float32)}
    ]
    arms, manifest_sha, diagnostics = _score_arms(
        tmp_path / "output",
        records,
        base_scored,
        [{"image_id": "x.png", "bag_logit": "0.25", "bag_probability": "0.6"}],
        {
            "x.png": np.asarray(
                [
                    [[1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[0.0, 0.0], [1.0, 1.0]],
                ],
                dtype=np.float32,
            )
        },
        candidate_root,
        {"x": {"diagnostic_path": "x.npz", "diagnostic_sha256": candidate_sha}},
    )
    assert np.allclose(arms[CONTROL_ARM][0]["candidate_logits"], [0.75, 0.0, 0.75])
    assert np.allclose(
        arms[SEMANTIC_ARM][0]["candidate_logits"],
        [0.5, 1.0 / 6.0, 5.0 / 6.0],
    )
    assert arms[SEMANTIC_ARM][0]["bag_probability"] == 0.6
    assert diagnostics["semantic_changed_selections"] == 1
    assert len(manifest_sha) == 64


def test_independent_b4_formula_matches_three_score_architecture() -> None:
    base_rank = np.asarray([1.0, 0.0, 0.5])
    bas_rank = np.asarray([0.0, 0.5, 1.0])
    upstream_rank = np.asarray([0.5, 0.0, 1.0])
    scores = _expected_arm_scores(base_rank, upstream_rank, bas_rank)
    assert ARMS == (CONTROL_ARM, SEMANTIC_ARM)
    assert np.allclose(scores[CONTROL_ARM], [0.75, 0.0, 0.75])
    assert np.allclose(scores[SEMANTIC_ARM], [0.5, 1.0 / 6.0, 5.0 / 6.0])
    assert np.allclose(
        _correlation_reference(base_rank, upstream_rank),
        [0.75, 0.0, 0.75],
    )


def test_b4_identity_is_unique_and_same_gallery() -> None:
    assert EXPERIMENT_ID == "EXP-20260801-codex-b4-same-gallery-bas-semantic-v1"
    assert "rich" not in EXPERIMENT_ID


def test_b4_stage_a_sources_keep_gt_and_collaborator_outputs_out() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "project/run_same_gallery_bas_semantic_b4.py").read_text(
        encoding="utf-8"
    )
    auditor = (
        root / "project/audit_same_gallery_bas_semantic_b4_output.py"
    ).read_text(encoding="utf-8")
    assert "BTXRDSegmentationDataset" not in runner
    assert "BTXRDSegmentationDataset" not in auditor
    assert "--dataset-root" not in auditor
    assert "wanwin" not in runner
    assert "wanwin" not in auditor
    assert "REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE = False" in runner


def test_independent_auditor_recomputes_class_contrast(tmp_path: Path) -> None:
    path = tmp_path / "evidence.npz"
    normal = np.asarray([[0.75, 0.25]], dtype=np.float32)
    tumor = np.asarray([[0.25, 0.75]], dtype=np.float32)
    contrast = tumor / (tumor + normal)
    np.savez_compressed(
        path,
        activation=contrast,
        normal_activation=normal,
        tumor_activation=tumor,
        class_contrast_activation=contrast,
    )
    with np.load(path, allow_pickle=False) as evidence:
        _audit_extra_evidence(evidence, "x.png")
