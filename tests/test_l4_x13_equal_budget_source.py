from __future__ import annotations

import numpy as np

from project.run_l4_x13_equal_budget_source import budget_indices, r7_select


def test_every_subset_receives_exact_budget() -> None:
    sources = np.asarray(["layercam320"] * 3 + ["classifier448"] * 4 + ["external_saliency"] * 5)
    upstream = np.linspace(0.0, 1.0, len(sources))
    g1 = np.linspace(1.0, 0.0, len(sources))
    for subset in (
        ("layercam320",),
        ("classifier448",),
        ("external_saliency",),
        ("layercam320", "classifier448"),
        ("layercam320", "external_saliency"),
        ("classifier448", "external_saliency"),
        ("layercam320", "classifier448", "external_saliency"),
    ):
        kept = budget_indices(subset, sources, upstream, g1, 3)
        assert len(kept) == 3
        assert all(sources[index] in subset for index in kept)


def test_upstream_budget_rule_and_r7_are_deterministic() -> None:
    sources = np.asarray(["layercam320"] * 4)
    upstream = np.asarray([0.2, 0.9, 0.7, 0.1])
    g1 = np.asarray([0.8, 0.1, 0.6, 0.4])
    kept = budget_indices(("layercam320",), sources, upstream, g1, 3)
    assert kept.tolist() == [1, 2, 0]
    assert r7_select(g1, upstream, kept) in kept.tolist()
