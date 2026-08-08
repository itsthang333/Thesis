from __future__ import annotations

from select_x4_qualitative_cases import choose_closest


def test_choose_closest_prefers_unused_then_image_id():
    rows = [
        {"image_id": "b", "dice": 0.5},
        {"image_id": "a", "dice": 0.5},
        {"image_id": "c", "dice": 0.4},
    ]
    assert choose_closest(rows, field="dice", target=0.5, used=set())["image_id"] == "a"
    assert choose_closest(rows, field="dice", target=0.5, used={"a"})["image_id"] == "b"


def test_choose_closest_returns_none_for_empty_input():
    assert choose_closest([], field="dice", target=0.5, used=set()) is None
