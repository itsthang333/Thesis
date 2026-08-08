from __future__ import annotations

import numpy as np

from project.freeze_x4_cam_masks import percentile_cam_mask, resize_binary_native


def test_percentile_cam_mask_matches_frozen_rule() -> None:
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = percentile_cam_mask(values, percentile=90.0)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 10
    assert mask[9].all()


def test_percentile_cam_mask_makes_constant_map_empty() -> None:
    assert not percentile_cam_mask(np.ones((7, 9), dtype=np.float32)).any()


def test_resize_binary_native_is_nearest_and_binary() -> None:
    source = np.asarray([[1, 0], [0, 0]], dtype=bool)
    resized = resize_binary_native(source, width=4, height=4)
    assert resized.dtype == np.bool_
    assert resized.shape == (4, 4)
    assert int(resized.sum()) == 4
    assert resized[:2, :2].all()
