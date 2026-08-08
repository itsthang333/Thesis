from __future__ import annotations

import pytest

from freeze_x4_gate_predictions import GATE_THRESHOLD, gate_uses_direct_mask


@pytest.mark.parametrize(
    ("arm", "known", "binary", "ten", "expected"),
    [
        ("known_binary_label", 0, 1.0, 1.0, False),
        ("known_binary_label", 1, 0.0, 0.0, True),
        ("binary_predicted_gate", 0, GATE_THRESHOLD - 1e-6, 1.0, False),
        ("binary_predicted_gate", 0, GATE_THRESHOLD, 0.0, True),
        ("ten_class_predicted_gate", 1, 1.0, GATE_THRESHOLD - 1e-6, False),
        ("ten_class_predicted_gate", 0, 0.0, GATE_THRESHOLD, True),
        ("label_free_rich_gallery_student", 1, 1.0, 1.0, None),
    ],
)
def test_gate_decisions(arm, known, binary, ten, expected):
    assert gate_uses_direct_mask(
        arm,
        known_tumor=known,
        binary_probability=binary,
        ten_class_probability=ten,
    ) is expected


def test_unknown_gate_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown X4 gate arm"):
        gate_uses_direct_mask(
            "invented",
            known_tumor=1,
            binary_probability=1.0,
            ten_class_probability=1.0,
        )
