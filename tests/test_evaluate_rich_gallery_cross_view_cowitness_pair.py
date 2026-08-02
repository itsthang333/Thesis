from __future__ import annotations

import inspect

import numpy as np

from project.evaluate_rich_gallery_cross_view_cowitness_pair import (
    _paired_group_bootstrap,
    dice,
    iou,
    size_group,
)


def test_binary_metrics_and_size_boundaries() -> None:
    target = np.asarray([[1, 1], [0, 0]], dtype=bool)
    prediction = np.asarray([[1, 0], [1, 0]], dtype=bool)
    assert dice(prediction, target) == 0.5
    assert np.isclose(iou(prediction, target), 1.0 / 3.0)
    assert size_group(0.009) == "small"
    assert size_group(0.01) == "medium"
    assert size_group(0.05) == "large"


def test_paired_bootstrap_preserves_group_pairing() -> None:
    rows = []
    for group, baseline, full in (("a", 0.1, 0.3), ("b", 0.2, 0.4)):
        rows.extend(
            [
                {"group_id": group, "variant": "baseline", "dice": baseline, "size_group": "small"},
                {"group_id": group, "variant": "full", "dice": full, "size_group": "small"},
            ]
        )
    result = _paired_group_bootstrap(
        rows, "full", "baseline", subgroup="overall", replicates=100, seed=5
    )
    assert np.isclose(result["mean_delta"], 0.2)
    assert result["groups"] == 2


def test_annotation_boundary_exists_only_in_stage_b_evaluator() -> None:
    import project.audit_rich_gallery_cross_view_cowitness_output as audit
    import project.evaluate_rich_gallery_cross_view_cowitness_pair as evaluate
    import project.run_rich_gallery_cross_view_cowitness_pair as run

    assert "build_segmentation_dataset" not in inspect.getsource(run)
    assert "build_segmentation_dataset" not in inspect.getsource(audit)
    assert "Annotation boundary" in inspect.getsource(evaluate)
