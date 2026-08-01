from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from package_rich_gallery_stage_a_transport import (
    inspect_npz_keys,
    inventory,
    path_tokens,
    reject_forbidden_path,
    safe_child,
)


def test_transport_paths_fail_closed_on_gt_stage_b_and_test_tokens() -> None:
    assert {"stage", "a", "scores", "image"}.issubset(
        path_tokens("stage_a_scores/image.npz")
    )
    reject_forbidden_path("stage_a_scores/image.npz")
    for path in (
        "stage_b/per_image.csv",
        "validation/ground_truth.json",
        "annotations/polygons.csv",
        "test/image.npz",
    ):
        with pytest.raises(ValueError):
            reject_forbidden_path(path)


def test_npz_inspection_rejects_gt_named_and_object_arrays(tmp_path: Path) -> None:
    safe = tmp_path / "safe.npz"
    np.savez_compressed(safe, candidate_indices=np.arange(3, dtype=np.int32))
    assert inspect_npz_keys(safe) == ("candidate_indices",)
    gt = tmp_path / "gt.npz"
    np.savez_compressed(gt, ground_truth=np.zeros(3, dtype=np.uint8))
    with pytest.raises(ValueError):
        inspect_npz_keys(gt)
    objects = tmp_path / "objects.npz"
    np.savez_compressed(objects, values=np.asarray([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        inspect_npz_keys(objects)


def test_safe_child_and_inventory_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "stage_a_scores").mkdir()
    (tmp_path / "prediction_freeze.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "stage_a_scores" / "b.npz").write_bytes(b"b")
    assert safe_child(tmp_path, "stage_a_scores/b.npz") == (
        tmp_path / "stage_a_scores" / "b.npz"
    ).resolve()
    with pytest.raises(ValueError):
        safe_child(tmp_path, "../escape")
    rows = inventory(
        tmp_path,
        [Path("stage_a_scores/b.npz"), Path("prediction_freeze.json")],
    )
    assert [row["path"] for row in rows] == [
        "prediction_freeze.json",
        "stage_a_scores/b.npz",
    ]
    assert [row["bytes"] for row in rows] == [4, 1]
