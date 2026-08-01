from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audit_rich_gallery_stage_a_transport import (
    audit_g1_baseline_row,
    find_forbidden_transport_paths,
    safe_transport_path,
)


def _candidate_payload() -> dict[str, np.ndarray]:
    return {
        "sam_masks": np.ones((3, 4, 4), dtype=np.uint8),
        "selection_scores": np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        "proposal_source_ids": np.asarray(
            ["classifier448", "layercam320", "external_saliency"]
        ),
    }


def _stage_a_payload() -> dict[str, np.ndarray]:
    return {
        "candidate_indices": np.asarray([0, 1, 2], dtype=np.int32),
        "source_ids": np.asarray([0, 1, 2], dtype=np.int16),
        "upstream_scores": np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        "g1_frozen_candidate_logits": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
    }


def _row() -> dict[str, str]:
    return {
        "variant": "g1_frozen__rank_fusion",
        "image_id": "example.png",
        "selected_local_index": "1",
        "selected_candidate_index": "1",
    }


def test_independent_row_audit_reproduces_fixed_g1_upstream_choice() -> None:
    aligned = audit_g1_baseline_row(_row(), _candidate_payload(), _stage_a_payload())
    assert aligned.candidate_indices.tolist() == [0, 1, 2]


def test_independent_row_audit_rejects_changed_choice() -> None:
    row = _row()
    row["selected_local_index"] = "0"
    with pytest.raises(ValueError):
        audit_g1_baseline_row(row, _candidate_payload(), _stage_a_payload())


def test_transport_inventory_rejects_stage_b_and_gt_derived_paths(tmp_path: Path) -> None:
    (tmp_path / "stage_a_scores").mkdir()
    (tmp_path / "stage_a_scores" / "safe.npz").write_bytes(b"safe")
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "per_image.csv").write_text("forbidden")
    assert find_forbidden_transport_paths(tmp_path) == ["evaluation/per_image.csv"]


def test_safe_transport_path_blocks_escape(tmp_path: Path) -> None:
    safe = safe_transport_path(tmp_path, "stage_a_scores/image.npz")
    assert safe == (tmp_path / "stage_a_scores" / "image.npz").resolve()
    with pytest.raises(ValueError):
        safe_transport_path(tmp_path, "../ground_truth.csv")
