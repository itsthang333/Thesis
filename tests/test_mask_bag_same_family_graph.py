from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

torch = pytest.importorskip("torch")

from models.mask_bag_same_family_graph import (  # noqa: E402
    SameFamilyGraphConfig,
    build_cached_same_family_graph,
    score_same_family_graph_records,
)


class _Base(torch.nn.Module):
    def score_descriptors(self, descriptors, valid):
        logits = descriptors[..., 0] * valid.to(descriptors.dtype)
        return logits, descriptors


def _record() -> dict[str, object]:
    return {
        "image_id": "sample.jpeg",
        "label": 1,
        "candidate_indices": np.asarray([2, 5, 9], dtype=np.int32),
        "descriptors": np.asarray([[2.0], [0.0], [1.0]], dtype=np.float16),
        "flipped_descriptors": np.asarray(
            [[0.0], [2.0], [1.0]], dtype=np.float16
        ),
        "family_ids": np.asarray([0, 0, 1], dtype=np.int32),
        "pairwise_iou": np.asarray(
            [[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "pairwise_containment": np.asarray(
            [[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    }


def test_cached_pairwise_graph_is_same_family_symmetric_and_isolation_safe() -> None:
    record = _record()
    graph = build_cached_same_family_graph(
        torch.from_numpy(record["pairwise_iou"])[None],
        torch.from_numpy(record["pairwise_containment"])[None],
        torch.ones((1, 3), dtype=torch.bool),
        torch.from_numpy(record["family_ids"])[None],
    )
    assert torch.equal(graph, graph.transpose(1, 2))
    assert graph[0, 0, 1] == 1
    assert graph[0, 0, 2] == 0
    assert graph[0, 1, 2] == 0
    assert graph[0, 2, 2] == 1


def test_fixed_graph_scoring_is_swap_invariant_and_preserves_isolated_logits() -> None:
    scored = score_same_family_graph_records(
        [_record()],
        _Base(),
        bag_temperature=0.2,
        graph_config=SameFamilyGraphConfig(),
        batch_size=1,
        device=torch.device("cpu"),
    )[0]
    assert scored["candidate_count"] == 3
    assert scored["view_swap_exact"] is True
    assert scored["alpha_zero_identity_exact"] is True
    assert scored["graph_symmetric"] is True
    assert scored["cross_family_edge_count"] == 0
    assert scored["non_self_edge_count"] == 1
    assert scored["isolated_candidate_count"] == 1
    assert scored["isolated_logits_exact"] is True
    assert scored["candidate_logits"][2] == scored["base_candidate_logits"][2]


def test_s3_contract_rejects_parameter_search_values() -> None:
    with pytest.raises(ValueError):
        SameFamilyGraphConfig(alpha=1.0)
    with pytest.raises(ValueError):
        SameFamilyGraphConfig(minimum_iou=-0.1)
    with pytest.raises(ValueError):
        SameFamilyGraphConfig(iterations=0)
