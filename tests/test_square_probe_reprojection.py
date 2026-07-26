from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from project.models.mae_reconstruction import pad_to_square, project_square_map
from project.tools.reproject_frozen_square_probe_maps import (
    summarize_aspect_ratios,
)


def test_inverse_square_projection_removes_horizontal_padding() -> None:
    image = Image.new("RGB", (4, 2), 0)
    _square, projection = pad_to_square(image, fill=0)
    square_map = np.zeros((4, 4), dtype=np.float32)
    square_map[1:3] = 1.0
    restored = project_square_map(
        square_map,
        projection,
        output_height=2,
        output_width=4,
    )
    np.testing.assert_allclose(restored, np.ones((2, 4)), atol=1e-6)


def test_inverse_square_projection_removes_vertical_padding() -> None:
    image = Image.new("RGB", (2, 4), 0)
    _square, projection = pad_to_square(image, fill=0)
    square_map = np.zeros((4, 4), dtype=np.float32)
    square_map[:, 1:3] = 1.0
    restored = project_square_map(
        square_map,
        projection,
        output_height=4,
        output_width=2,
    )
    np.testing.assert_allclose(restored, np.ones((4, 2)), atol=1e-6)


def test_aspect_ratio_summary_is_json_serializable() -> None:
    summary = summarize_aspect_ratios([1.0, 0.8, 0.7])
    assert summary == {
        "square": 1,
        "below_0_90": 2,
        "below_0_75": 1,
        "minimum": 0.7,
        "mean": pytest.approx(5.0 / 6.0),
    }
    assert type(summary["square"]) is int
    assert type(summary["below_0_90"]) is int
    assert type(summary["below_0_75"]) is int
    json.dumps(summary)
