from __future__ import annotations

import numpy as np

from project.models.mask_bag_selector_cache import pack_candidate_masks
from project.run_bas_candidate_descriptor_b1 import _score_arms
from project.models.mask_bag_selector_cache_io import sha256_file


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
