import numpy as np

from btxrd_wsss.evaluation.segmentation import segmentation_metrics


def test_native_segmentation_metrics() -> None:
    target = np.zeros((8, 8), dtype=bool)
    target[2:5, 2:5] = True
    prediction = target.copy()
    result = segmentation_metrics(prediction, target)
    assert result["dice"] == 1.0
    assert result["iou"] == 1.0
    assert result["hd95"] == 0.0
    assert not result["complete_miss"]


def test_complete_miss_is_explicit() -> None:
    target = np.zeros((4, 4), dtype=bool)
    target[0, 0] = True
    result = segmentation_metrics(np.zeros_like(target), target)
    assert result["dice"] == 0.0
    assert result["complete_miss"]
    assert not result["surface_defined"]
