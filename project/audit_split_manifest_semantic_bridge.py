from __future__ import annotations

"""Fail-closed semantic bridge for byte-distinct BTXRD split manifests.

The rich-gallery artifacts were frozen against the historical CRLF manifest,
while later local experiments added a semantic table hash column.  This tool
proves that cohort, grouping and scientific metadata are identical without
pretending that the two files have the same byte hash.
"""

import argparse
import csv
import json
from pathlib import Path

from mae_reconstruction_io import sha256_file


VOLATILE_FIELDS = {"dataset_table_sha256", "dataset_table_semantic_sha256"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-split", type=Path, required=True)
    parser.add_argument("--expected-gallery-sha256", required=True)
    parser.add_argument("--canonical-split", type=Path, required=True)
    parser.add_argument("--expected-canonical-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row.get("image_id", "") for row in rows}) != len(rows):
        raise ValueError(f"split manifest is empty or has duplicate IDs: {path}")
    return rows


def compare_semantics(
    gallery_rows: list[dict[str, str]],
    canonical_rows: list[dict[str, str]],
) -> dict[str, object]:
    gallery = {row["image_id"]: row for row in gallery_rows}
    canonical = {row["image_id"]: row for row in canonical_rows}
    if set(gallery) != set(canonical):
        raise ValueError("split manifests do not contain the same image IDs")
    gallery_fields = set(gallery_rows[0])
    canonical_fields = set(canonical_rows[0])
    common_scientific = sorted((gallery_fields & canonical_fields) - VOLATILE_FIELDS)
    if not common_scientific:
        raise ValueError("split manifests have no scientific fields in common")
    differences: dict[str, int] = {}
    for field in common_scientific:
        count = sum(
            gallery[image_id].get(field, "") != canonical[image_id].get(field, "")
            for image_id in gallery
        )
        if count:
            differences[field] = count
    if differences:
        raise ValueError(f"split scientific semantics differ: {differences}")
    split_counts = {
        name: sum(row.get("split") == name for row in gallery_rows)
        for name in ("train", "val", "test", "excluded")
    }
    if split_counts != {"train": 2981, "val": 371, "test": 373, "excluded": 21}:
        raise ValueError(f"unexpected canonical split counts: {split_counts}")
    return {
        "semantic_match": True,
        "rows": len(gallery_rows),
        "image_ids_exact_match": True,
        "scientific_fields_compared": common_scientific,
        "scientific_field_count": len(common_scientific),
        "ignored_volatile_fields": sorted(VOLATILE_FIELDS),
        "gallery_only_fields": sorted(gallery_fields - canonical_fields),
        "canonical_only_fields": sorted(canonical_fields - gallery_fields),
        "split_counts": split_counts,
    }


def main() -> None:
    args = parse_args()
    if sha256_file(args.gallery_split) != args.expected_gallery_sha256:
        raise ValueError("gallery split byte SHA-256 mismatch")
    if sha256_file(args.canonical_split) != args.expected_canonical_sha256:
        raise ValueError("canonical split byte SHA-256 mismatch")
    comparison = compare_semantics(_read(args.gallery_split), _read(args.canonical_split))
    result = {
        "audit_pass": True,
        "stage": "btxrd_split_manifest_semantic_bridge_v1",
        "gallery_split_sha256": args.expected_gallery_sha256,
        "canonical_split_sha256": args.expected_canonical_sha256,
        **comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
