from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import project.audit_highres_candidate_pmil_s10_output as auditor
from project.models.bas_candidate_localizer import equal_rank_aggregate
from project.models.highres_candidate_pmil import pareto_guarded_selection
import project.run_highres_candidate_pmil_s10 as runner


def test_independent_rank_matches_tie_aware_reference() -> None:
    values = np.asarray([2.0, 1.0, 2.0, -1.0], dtype=np.float32)
    observed = auditor._rank(values)
    expected = np.asarray([5 / 6, 1 / 3, 5 / 6, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(observed, expected)


def test_independent_rank_aggregation_is_byte_exact() -> None:
    rng = np.random.default_rng(17)
    for count in (1, 2, 7, 81):
        values = [rng.integers(-3, 4, size=count).astype(np.float32) for _ in range(3)]
        valid = torch.ones(1, count, dtype=torch.bool)
        produced = equal_rank_aggregate(
            tuple(torch.from_numpy(value)[None] for value in values), valid
        )[0].numpy()
        observed = np.stack(tuple(auditor._rank(value) for value in values)).mean(
            axis=0, dtype=np.float32
        )
        np.testing.assert_array_equal(observed, produced)


def test_independent_pareto_matches_producer_with_ties() -> None:
    identity = np.asarray([0.0, 2.0, 2.0, 1.0], dtype=np.float32)
    capture = np.asarray([0.0, 1.0, 1.0, 3.0], dtype=np.float32)
    purity = np.asarray([0.0, 1.0, 1.0, 3.0], dtype=np.float32)
    indices = np.asarray([2, 4, 8, 9], dtype=np.int64)
    expected = pareto_guarded_selection(identity, capture, purity, indices, 0)
    local, count = auditor._pareto_local(
        identity, capture, purity, indices, control_local=0
    )
    assert indices[local] == expected.selected_index
    assert count == expected.dominator_count


def test_independent_projection_matches_runner() -> None:
    masks = np.zeros((2, 64, 32), dtype=np.uint8)
    masks[0, 20:28, 10:18] = 1
    masks[1, 32:50, 12:25] = 1
    projection = SimpleNamespace(padded_side=64, content_box=(16, 0, 48, 64))
    produced, _content = runner._project_square_supports(masks, projection=projection)
    observed = auditor._reference_direct_square(
        torch.from_numpy(masks),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
    ).to(torch.float16)
    assert torch.equal(observed, produced)


def test_independent_capture_purity_separates_extent_and_dilution() -> None:
    dense = np.full((4, 4), -8.0, dtype=np.float32)
    dense[:2, :2] = 8.0
    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, :2, :2] = 1
    masks[1] = 1
    rings = np.zeros_like(masks)
    rings[0, :3, :3] = 1
    rings[0] -= masks[0]
    rings[1, 0] = 1
    capture, purity = auditor._capture_purity_numpy(
        dense, masks, rings, np.ones((4, 4), dtype=np.float32)
    )
    assert capture[1] > capture[0]
    assert purity[0] > purity[1]
