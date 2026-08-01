from __future__ import annotations

import numpy as np
import pytest

from models.rich_gallery_bas_residual import (
    align_transport_payloads,
    average_percentile_rank,
    bas_candidate_scores,
    score_rich_gallery_bas_pair,
)


def _candidate_payload() -> dict[str, np.ndarray]:
    masks = np.zeros((4, 8, 8), dtype=np.uint8)
    masks[0, :2, :2] = 1
    masks[1, 2:6, 2:6] = 1
    masks[2, 5:, 5:] = 1
    masks[3, 1:7, 1:7] = 1
    return {
        "sam_masks": masks,
        "selection_scores": np.asarray([0.1, 0.7, 0.4, 0.2], dtype=np.float32),
        "proposal_source_ids": np.asarray(
            ["classifier448", "LayerCAM-anchor", "biomed-external", "classifier448"]
        ),
    }


def _stage_a_payload() -> dict[str, np.ndarray]:
    return {
        "candidate_indices": np.asarray([2, 0, 3], dtype=np.int32),
        "source_ids": np.asarray([2, 0, 0], dtype=np.int16),
        "upstream_scores": np.asarray([0.4, 0.1, 0.2], dtype=np.float32),
        "g1_frozen_candidate_logits": np.asarray([0.5, 0.2, 0.1], dtype=np.float32),
    }


def test_transport_alignment_reconstructs_exact_kept_rich_gallery() -> None:
    aligned = align_transport_payloads(_candidate_payload(), _stage_a_payload())
    assert aligned.candidate_indices.tolist() == [2, 0, 3]
    assert aligned.source_ids.tolist() == [2, 0, 0]
    assert aligned.candidate_masks.shape == (3, 8, 8)
    np.testing.assert_array_equal(
        aligned.upstream_scores,
        np.asarray([0.4, 0.1, 0.2], dtype=np.float32),
    )


@pytest.mark.parametrize("field", ["upstream_scores", "source_ids"])
def test_transport_alignment_fails_on_stage_a_candidate_drift(field: str) -> None:
    stage_a = _stage_a_payload()
    stage_a[field] = stage_a[field].copy()
    stage_a[field][0] += 1
    with pytest.raises(ValueError):
        align_transport_payloads(_candidate_payload(), stage_a)


def test_exact_tie_rank_matches_collaborator_contract() -> None:
    actual = average_percentile_rank(np.asarray([2.0, 1.0, 2.0, 4.0]))
    np.testing.assert_allclose(actual, [0.5, 0.0, 0.5, 1.0], atol=0.0)


def test_bas_residual_can_change_choice_without_source_routing() -> None:
    pair = score_rich_gallery_bas_pair(
        np.asarray([3.0, 2.0, 1.0]),
        np.asarray([1.0, 3.0, 2.0]),
        np.asarray([3.0, 1.0, 2.0]),
    )
    assert pair.baseline_local_index == 1
    assert pair.bas_residual_local_index == 0
    np.testing.assert_allclose(pair.baseline_rank, [0.5, 0.75, 0.25])
    np.testing.assert_allclose(pair.bas_residual_rank, [2 / 3, 0.5, 1 / 3])


def test_bas_candidate_score_rewards_supported_compact_candidate() -> None:
    activation = np.zeros((8, 8), dtype=np.float32)
    activation[2:6, 2:6] = 1.0
    masks = _candidate_payload()["sam_masks"]
    coverage, purity, harmonic = bas_candidate_scores(activation, masks)
    assert coverage.shape == purity.shape == harmonic.shape == (4,)
    assert int(np.argmax(harmonic)) == 1
    assert harmonic[1] == pytest.approx(1.0)


def test_residual_pair_rejects_nonfinite_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        score_rich_gallery_bas_pair(
            np.asarray([1.0, np.nan]),
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 2.0]),
        )
    with pytest.raises(ValueError):
        score_rich_gallery_bas_pair(
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
            np.asarray([1.0]),
        )
