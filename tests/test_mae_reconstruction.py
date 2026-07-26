from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from project.mae_reconstruction_io import (
    load_split_rows_without_annotations,
    save_float_map,
    sha256_file,
)
from project.models.mae_reconstruction import (
    SquareProjection,
    accumulate_masked_squared_error,
    make_noise_bank,
    noise_bank_sha256,
    pad_to_square,
    patchify,
    project_square_map,
    robust_foreground_normalize,
    unpatchify,
    validate_complete_mask_coverage,
)


def test_patchify_unpatchify_round_trip() -> None:
    image = torch.arange(2 * 3 * 8 * 12, dtype=torch.float32).reshape(2, 3, 8, 12)
    patches = patchify(image, 4)
    restored = unpatchify(
        patches, patch_size=4, channels=3, grid_height=2, grid_width=3
    )
    assert torch.equal(restored, image)


def test_noise_bank_is_reproducible_and_hashed() -> None:
    first = make_noise_bank(num_masks=10, num_patches=196, seed=42)
    second = make_noise_bank(num_masks=10, num_patches=196, seed=42)
    assert torch.equal(first, second)
    assert noise_bank_sha256(first) == noise_bank_sha256(second)
    assert not torch.equal(first, make_noise_bank(num_masks=10, num_patches=196, seed=43))


def test_masked_error_ignores_visible_patches() -> None:
    target = torch.zeros((1, 3, 4, 4))
    prediction = torch.ones((1, 4, 12))
    mask = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    errors, coverage = accumulate_masked_squared_error(
        prediction_patches=prediction,
        pixel_values=target,
        patch_mask=mask,
        patch_size=2,
    )
    assert torch.equal(errors, coverage)
    assert float(errors.sum()) == 8.0


def test_complete_coverage_rejects_unseen_patch() -> None:
    with pytest.raises(ValueError, match="never reconstructed"):
        validate_complete_mask_coverage(
            [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
            num_patches=2,
        )


def test_square_projection_and_normalization() -> None:
    image = Image.new("RGB", (4, 2), color=(10, 10, 10))
    padded, projection = pad_to_square(image)
    assert padded.size == (4, 4)
    projected = project_square_map(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        projection,
        output_height=2,
        output_width=4,
    )
    normalized = robust_foreground_normalize(
        projected, np.ones_like(projected, dtype=bool), low_percentile=0, high_percentile=100
    )
    assert projected.shape == (2, 4)
    assert normalized.min() == 0
    assert normalized.max() == 1


def test_projection_rejects_invalid_box() -> None:
    with pytest.raises(ValueError, match="outside"):
        project_square_map(
            np.zeros((4, 4), dtype=np.float32),
            SquareProjection(4, (0, 0, 5, 4)),
            output_height=4,
            output_width=4,
        )


def test_split_loader_does_not_require_annotation_column(tmp_path: Path) -> None:
    manifest = tmp_path / "split.csv"
    manifest.write_text(
        "image_id,group_id,split,eligible,tumor,image_sha256\n"
        f"normal.jpeg,g1,train,1,0,{'0' * 64}\n"
        f"tumor.jpeg,g2,val,1,1,{'1' * 64}\n",
        encoding="utf-8",
    )
    rows = load_split_rows_without_annotations(
        manifest,
        expected_sha256=sha256_file(manifest),
        split="val",
    )
    assert rows == [{
        "image_id": "tumor.jpeg",
        "group_id": "g2",
        "split": "val",
        "eligible": "1",
        "tumor": "1",
        "image_sha256": "1" * 64,
    }]


def test_save_float_map_is_bounded_and_non_pickle(tmp_path: Path) -> None:
    path = tmp_path / "map.npy"
    save_float_map(path, np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    loaded = np.load(path, allow_pickle=False)
    assert loaded.dtype == np.float16
    assert np.allclose(loaded, [[0.0, 0.5, 1.0]])
