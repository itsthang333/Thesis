import numpy as np

from btxrd_wsss.data.tiling import blend_tile_maps, extract_tiles, tile_layout
from btxrd_wsss.data.views import pad_to_multiple


def test_overlapping_tiles_cover_native_grid_and_blend_identity() -> None:
    image = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)
    tiles = extract_tiles(image, image_id="x", tile_size=6, overlap=0.5)
    maps = [tile.pixels[: tile.box[3] - tile.box[1], : tile.box[2] - tile.box[0]] for tile in tiles]
    blended = blend_tile_maps(maps, [tile.box for tile in tiles], image.shape)
    np.testing.assert_allclose(blended, image, rtol=1e-5, atol=1e-5)


def test_small_image_is_reflect_padded_without_changing_native_box() -> None:
    image = np.ones((3, 4), dtype=np.float32)
    tile = extract_tiles(image, image_id="x", tile_size=8, overlap=0.5)[0]
    assert tile.pixels.shape == (8, 8)
    assert tile.box == (0, 0, 4, 3)
    assert tile_layout(3, 4, 8, 0.5) == [(0, 0, 4, 3)]


def test_aspect_preserving_full_view_is_padded_to_hrnet_multiple() -> None:
    image = np.arange(107 * 128, dtype=np.float32).reshape(107, 128)
    padded, valid_shape = pad_to_multiple(image, 32)
    assert padded.shape == (128, 128)
    assert valid_shape == image.shape
    np.testing.assert_array_equal(padded[:107], image)
