from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from build_x4_inner_split import assign_inner_roles  # noqa: E402


def make_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(2981):
        tumor = int(index % 2 == 0)
        rows.append(
            {
                "image_id": f"IMG{index:06d}.jpeg",
                "group_id": f"group-{index // 3:04d}",
                "split": "train",
                "eligible": "1",
                "tumor": str(tumor),
                "tumor_type": str(1 if tumor else 0),
            }
        )
    # Keep every synthetic group label-homogeneous, as the real manifest is.
    for group_id, group_rows in _groups(rows).items():
        value = group_rows[0]["tumor"]
        tumor_type = "1" if value == "1" else "0"
        for row in group_rows:
            row["tumor"] = value
            row["tumor_type"] = tumor_type
    return rows


def _groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["group_id"]].append(row)
    return result


def test_inner_split_is_complete_deterministic_and_group_disjoint() -> None:
    rows = make_rows()
    first = assign_inner_roles(rows)
    second = assign_inner_roles(list(reversed(rows)))
    assert first == second
    assert len(first) == 2981
    assert {row["inner_role"] for row in first} == {"inner_train", "inner_holdout"}
    roles = defaultdict(set)
    for row in first:
        roles[row["group_id"]].add(row["inner_role"])
    assert all(len(value) == 1 for value in roles.values())
    counts = Counter(row["inner_role"] for row in first)
    assert 0.10 < counts["inner_holdout"] / 2981 < 0.20

