from __future__ import annotations

import numpy as np

from export_x4_yolo_dataset import materialize_image, polygons_to_yolo_rows
from freeze_x4_yolo_predictions import union_instance_masks


def test_polygon_export_is_normalized_and_preserves_instances():
    payload = {
        "shapes": [
            {"shape_type": "polygon", "points": [[0, 0], [100, 0], [100, 50]]},
            {"shape_type": "polygon", "points": [[10, 5], [20, 5], [20, 10]]},
            {"shape_type": "rectangle", "points": [[0, 0], [1, 1]]},
        ]
    }
    rows = polygons_to_yolo_rows(payload, width=100, height=50)
    assert len(rows) == 2
    for row in rows:
        tokens = row.split()
        assert tokens[0] == "0"
        values = [float(value) for value in tokens[1:]]
        assert len(values) >= 6 and len(values) % 2 == 0
        assert all(0.0 <= value <= 1.0 for value in values)


def test_union_instance_masks_handles_empty_resize_and_union():
    assert not union_instance_masks(None, height=8, width=6).any()
    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, 0, 0] = 1
    masks[1, 3, 3] = 1
    union = union_instance_masks(masks, height=8, width=6)
    assert union.shape == (8, 6)
    assert union.any()


def test_copy_materialization_is_writable_and_does_not_modify_source(tmp_path):
    source = tmp_path / "readonly" / "image.jpeg"
    source.parent.mkdir()
    source.write_bytes(b"original-jpeg-bytes")
    source.chmod(0o444)
    destination = tmp_path / "export" / "image.jpeg"
    destination.parent.mkdir()
    materialize_image(source, destination, mode="copy")
    assert not destination.is_symlink()
    destination.write_bytes(b"ultralytics-repaired-copy")
    assert source.read_bytes() == b"original-jpeg-bytes"
