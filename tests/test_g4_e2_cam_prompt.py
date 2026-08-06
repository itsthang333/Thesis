from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from evaluate_g4_pseudo_mask_variant import _resize, _size_group  # noqa: E402
from run_g4_e2_cam_prompt import _validated_subset, generation_command  # noqa: E402


def test_e2_generation_command_is_single_prompt_common_320() -> None:
    values = generation_command(
        project=PROJECT,
        data_root=Path("/data"),
        split=Path("/split.csv"),
        classifier_split=Path("/classifier_split.csv"),
        classifier=Path("/classifier.pt"),
        sam=Path("/sam.pth"),
        output_dir=Path("/out"),
        method="gradcam_plus_plus",
        prompt_mode="box",
    )
    joined = " ".join(values)
    assert "--attribution-method gradcam_plus_plus" in joined
    assert "--sam-prompt-mode box" in joined
    assert "--disable-sam-prompt-ensemble" in values
    assert "--allow-validation-prompt-ablation" in values
    assert "--image-size 320" in joined


def test_e2_subset_and_native_size_groups_are_fail_closed() -> None:
    assert _validated_subset("cam,layercam", ("cam", "layercam"), "x") == (
        "cam", "layercam"
    )
    with pytest.raises(ValueError):
        _validated_subset("cam,cam", ("cam", "layercam"), "x")
    assert _size_group(0.00999) == "small_lt_1pct"
    assert _size_group(0.01) == "medium_1_to_5pct"
    assert _size_group(0.05) == "large_ge_5pct"


def test_mask_resize_uses_nearest_binary_grid() -> None:
    import numpy as np

    mask = np.asarray([[0, 1], [0, 0]], dtype=bool)
    resized = _resize(mask, (4, 4))
    assert resized.dtype == bool
    assert resized.shape == (4, 4)
    assert int(resized.sum()) == 4
