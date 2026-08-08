from __future__ import annotations

import numpy as np

from project.run_g4_e4_source_subset import (
    canonical_source,
    rank_fusion_subset,
    stable_select,
)


def test_source_mapping_is_fail_closed_and_specific() -> None:
    assert canonical_source("classifier448:layercam") == "classifier448"
    assert canonical_source("external_saliency") == "external_saliency"
    assert canonical_source("layercam") == "layercam320"


def test_subset_rank_fusion_only_ranks_eligible_candidates() -> None:
    g1 = np.asarray([100.0, 0.0, 1.0])
    upstream = np.asarray([100.0, 1.0, 0.0])
    eligible = np.asarray([False, True, True])
    fused = rank_fusion_subset(g1, upstream, eligible)
    assert np.isneginf(fused[0])
    assert np.allclose(fused[1:], [0.5, 0.5])
    assert stable_select(fused, g1, eligible) == 2
