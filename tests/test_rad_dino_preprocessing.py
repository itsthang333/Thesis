from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from models.rad_dino_preprocessing import raw_and_normalized_square


def test_square_geometry_and_normalization_are_finite() -> None:
    image = Image.fromarray(np.full((3, 5, 3), 128, dtype=np.uint8), mode="RGB")
    raw, normalized, projection = raw_and_normalized_square(image, input_size=10)
    assert tuple(raw.shape) == (3, 10, 10)
    assert tuple(normalized.shape) == (3, 10, 10)
    assert projection.padded_side == 5
    assert projection.content_box == (0, 1, 5, 4)
    assert np.isfinite(raw.numpy()).all()
    assert np.isfinite(normalized.numpy()).all()
