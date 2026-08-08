from __future__ import annotations

import numpy as np
from PIL import Image

from render_x4_qualitative_panels import (
    candidate_montage,
    normalize_score_map,
    overlay_heatmap,
    overlay_mask,
)


def test_normalize_score_map_is_finite_and_bounded():
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    result = normalize_score_map(values)
    assert result.shape == values.shape
    assert np.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    assert np.array_equal(normalize_score_map(np.ones((4, 4), np.float32)), np.zeros((4, 4)))


def test_overlay_and_candidate_montage_geometry():
    image = Image.fromarray(np.full((24, 16), 100, dtype=np.uint8), mode="L").convert("RGB")
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    assert overlay_mask(image, mask, (255, 0, 0)).size == image.size
    assert overlay_heatmap(image, mask.astype(np.float32)).size == image.size
    masks = np.stack([np.roll(mask, shift, axis=0) for shift in range(5)])
    montage = candidate_montage(image, masks, np.arange(5, dtype=np.float32), tile_size= ninety_six())
    assert montage.size == (96, 96)


def ninety_six() -> int:
    # Named helper keeps the minimum supported renderer geometry explicit.
    return 96
