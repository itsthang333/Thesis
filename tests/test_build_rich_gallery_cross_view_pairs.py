from __future__ import annotations

from project.build_rich_gallery_cross_view_pairs import build_pair_rows


def _row(image_id: str, group: str, view: str) -> dict[str, str]:
    return {
        "image_id": image_id,
        "group_id": group,
        "split": "train",
        "eligible": "1",
        "tumor": "1",
        "view": view,
        "anatomy": "femur",
        "tumor_type": "8",
        "image_sha256": (image_id[0] * 64),
    }


def test_pair_builder_matches_controls_and_is_deterministic() -> None:
    rows = [
        _row("a1.jpeg", "g1", "frontal"),
        _row("a2.jpeg", "g1", "lateral"),
        _row("b1.jpeg", "g2", "frontal"),
        _row("b2.jpeg", "g2", "lateral"),
        _row("c1.jpeg", "g3", "frontal"),
        _row("c2.jpeg", "g3", "lateral"),
    ]
    first, first_counts = build_pair_rows(rows, seed=7)
    second, second_counts = build_pair_rows(rows, seed=7)
    assert first == second
    assert first_counts == second_counts
    assert len(first) == 3
    for row in first:
        assert row["anchor_group_id"] == row["same_group_id"]
        assert len(
            {
                row["same_group_id"],
                row["control1_group_id"],
                row["control2_group_id"],
            }
        ) == 3
        assert row["same_view"] == row["control1_view"] == row["control2_view"]


def test_pair_builder_excludes_validation_and_single_view_groups() -> None:
    rows = [
        _row("a1.jpeg", "g1", "frontal"),
        _row("a2.jpeg", "g1", "lateral"),
        _row("b1.jpeg", "g2", "frontal"),
        _row("b2.jpeg", "g2", "lateral"),
        _row("c1.jpeg", "g3", "frontal"),
        _row("c2.jpeg", "g3", "lateral"),
        _row("single.jpeg", "single", "frontal"),
        {**_row("val.jpeg", "val", "lateral"), "split": "val"},
    ]
    pairs, counts = build_pair_rows(rows, seed=11)
    assert len(pairs) == 3
    assert counts["train_tumor_images"] == 7
    assert counts["distinct_view_groups"] == 3
    assert all("val.jpeg" not in row.values() for row in pairs)
