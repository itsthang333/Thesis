from __future__ import annotations

import pytest

from audit_split_manifest_semantic_bridge import compare_semantics


def _row(image_id: str, split: str = "train") -> dict[str, str]:
    return {
        "image_id": image_id,
        "group_id": f"g-{image_id}",
        "split": split,
        "tumor": "1",
        "image_sha256": "a" * 64,
        "dataset_table_sha256": "old",
    }


def test_bridge_ignores_only_declared_table_hash_fields() -> None:
    gallery = []
    for split, count in (("train", 2981), ("val", 371), ("test", 373), ("excluded", 21)):
        gallery.extend(_row(f"{split}-{index:04d}", split) for index in range(count))
    canonical = [dict(row) for row in gallery]
    for row in canonical:
        row["dataset_table_sha256"] = "new"
        row["dataset_table_semantic_sha256"] = "semantic"
    result = compare_semantics(gallery, canonical)
    assert result["semantic_match"] is True
    assert result["rows"] == 3746
    assert result["canonical_only_fields"] == ["dataset_table_semantic_sha256"]


def test_bridge_rejects_group_or_split_change() -> None:
    gallery = [_row("a")]
    canonical = [dict(gallery[0])]
    canonical[0]["group_id"] = "different"
    with pytest.raises(ValueError, match="scientific semantics differ"):
        compare_semantics(gallery, canonical)
