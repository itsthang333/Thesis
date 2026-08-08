from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from freeze_x4_training_targets import (  # noqa: E402
    binary_at_native,
    safe_relative,
    source_mask_path,
)
from frozen_io import sha256_file  # noqa: E402


def test_binary_at_native_uses_nearest_and_remains_binary() -> None:
    source = np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
    resized = binary_at_native(source, width=4, height=4)
    assert resized.dtype == bool
    assert resized.shape == (4, 4)
    assert np.array_equal(
        resized,
        np.asarray(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=bool,
        ),
    )


def test_source_mask_path_requires_one_safe_hash_locked_path(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = root / "masks" / "img.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((3, 5), dtype=np.uint8), mode="L").save(path)
    row = {"mask_path": "masks/img.png", "mask_sha256": sha256_file(path)}
    assert source_mask_path(row, root) == path
    with pytest.raises(ValueError, match="exactly one"):
        source_mask_path({**row, "mask_file": "masks/img.png"}, root)
    with pytest.raises(ValueError, match="unsafe"):
        safe_relative("../img.png")


def test_source_mask_path_rejects_hash_tamper(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(path)
    with pytest.raises(ValueError, match="SHA-256"):
        source_mask_path(
            {"mask_path": "img.png", "mask_sha256": "0" * 64}, tmp_path
        )
