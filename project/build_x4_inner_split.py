from __future__ import annotations

"""Build the deterministic group-aware inner split used by every X4 student."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from x4_contract import CANONICAL_SPLIT_SHA256


INNER_SPLIT_SEED = 20260808
HOLDOUT_FRACTION = 0.15


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"x4-inner-v1|{seed}|{group_id}".encode()).hexdigest()


def assign_inner_roles(
    rows: list[dict[str, str]],
    *,
    seed: int = INNER_SPLIT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> list[dict[str, object]]:
    train_rows = [
        row for row in rows if row.get("split") == "train" and row.get("eligible") == "1"
    ]
    if len(train_rows) != 2981:
        raise ValueError(f"X4 inner split requires 2,981 train images, got {len(train_rows)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        grouped[row["group_id"]].append(row)
    if any(len({row["tumor_type"] for row in group}) != 1 for group in grouped.values()):
        raise ValueError("X4 cannot stratify a heuristic group with mixed tumor types")

    strata: dict[int, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_id, group_rows in grouped.items():
        strata[int(group_rows[0]["tumor_type"])].append((group_id, group_rows))

    holdout_groups: set[str] = set()
    for tumor_type, groups in sorted(strata.items()):
        groups = sorted(groups, key=lambda item: (stable_rank(item[0], seed), item[0]))
        target_images = max(1, round(sum(len(item[1]) for item in groups) * holdout_fraction))
        selected_images = 0
        for group_id, group_rows in groups:
            if selected_images >= target_images:
                break
            holdout_groups.add(group_id)
            selected_images += len(group_rows)

    output = []
    for row in sorted(train_rows, key=lambda item: item["image_id"]):
        role = "inner_holdout" if row["group_id"] in holdout_groups else "inner_train"
        output.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": int(row["tumor"]),
                "tumor_type": int(row["tumor_type"]),
                "inner_role": role,
                "source_split": "train",
            }
        )
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    for row in output:
        roles_by_group[str(row["group_id"])].add(str(row["inner_role"]))
    if any(len(roles) != 1 for roles in roles_by_group.values()):
        raise RuntimeError("X4 inner split leaked a group across roles")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", default=CANONICAL_SPLIT_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")
    with args.split_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assigned = assign_inner_roles(rows)
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "x4_inner_split.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assigned[0]))
        writer.writeheader()
        writer.writerows(assigned)
    summary = {
        "schema_version": 1,
        "stage": "x4_group_aware_inner_split_v1",
        "canonical_split_sha256": args.expected_split_sha256,
        "inner_split_seed": INNER_SPLIT_SEED,
        "holdout_fraction_target": HOLDOUT_FRACTION,
        "images": len(assigned),
        "groups": len({row["group_id"] for row in assigned}),
        "inner_train_images": sum(row["inner_role"] == "inner_train" for row in assigned),
        "inner_holdout_images": sum(row["inner_role"] == "inner_holdout" for row in assigned),
        "inner_train_tumor_images": sum(
            row["inner_role"] == "inner_train" and row["tumor"] == 1 for row in assigned
        ),
        "inner_holdout_tumor_images": sum(
            row["inner_role"] == "inner_holdout" and row["tumor"] == 1 for row in assigned
        ),
        "cross_role_groups": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "outer_validation_images_used": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "x4_inner_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

