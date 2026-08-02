from __future__ import annotations

"""Freeze matched full/control training pairs without spatial annotations."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from mae_reconstruction_io import sha256_file


PAIR_FIELDS = (
    "pair_id",
    "anchor_image_id",
    "anchor_group_id",
    "anchor_view",
    "anchor_image_sha256",
    "same_image_id",
    "same_group_id",
    "same_view",
    "same_image_sha256",
    "control1_image_id",
    "control1_group_id",
    "control1_view",
    "control1_image_sha256",
    "control2_image_id",
    "control2_group_id",
    "control2_view",
    "control2_image_sha256",
    "anatomy",
    "tumor_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def _stable_order_key(seed: int, pair_id: str, image_id: str) -> str:
    return hashlib.sha256(f"{seed}|{pair_id}|{image_id}".encode("utf-8")).hexdigest()


def build_pair_rows(
    split_rows: list[dict[str, str]],
    *,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    train_tumor = [
        row
        for row in split_rows
        if row.get("split") == "train"
        and row.get("eligible") == "1"
        and row.get("tumor") == "1"
    ]
    if not train_tumor:
        raise ValueError("canonical train-tumor cohort is empty")
    by_group: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    by_stratum: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_tumor:
        required = (
            "image_id",
            "group_id",
            "view",
            "anatomy",
            "tumor_type",
            "image_sha256",
        )
        if any(not str(row.get(field, "")).strip() for field in required):
            raise ValueError("train-tumor pairing metadata are incomplete")
        by_group[row["group_id"]].append(row)
        by_stratum[(row["anatomy"], row["tumor_type"], row["view"])].append(row)

    candidate_pairs: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for group_id, group_rows in sorted(by_group.items()):
        by_view: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group_rows:
            by_view[row["view"]].append(row)
        views = sorted(by_view)
        if len(views) < 2:
            continue
        anchor = sorted(by_view[views[0]], key=lambda row: row["image_id"])[0]
        same = sorted(by_view[views[1]], key=lambda row: row["image_id"])[0]
        candidate_pairs.append((group_id, anchor, same))

    output: list[dict[str, str]] = []
    insufficient_controls = 0
    for group_id, anchor, same in candidate_pairs:
        pair_id = f"{group_id}|{anchor['image_id']}|{same['image_id']}"
        alternatives = [
            row
            for row in by_stratum[(same["anatomy"], same["tumor_type"], same["view"])]
            if row["group_id"] != group_id
        ]
        representatives: dict[str, dict[str, str]] = {}
        for row in alternatives:
            current = representatives.get(row["group_id"])
            if current is None or row["image_id"] < current["image_id"]:
                representatives[row["group_id"]] = row
        controls = sorted(
            representatives.values(),
            key=lambda row: (_stable_order_key(seed, pair_id, row["image_id"]), row["image_id"]),
        )
        if len(controls) < 2:
            insufficient_controls += 1
            continue
        control1, control2 = controls[:2]
        if len({group_id, control1["group_id"], control2["group_id"]}) != 3:
            raise RuntimeError("full/control groups are not distinct")
        if not (
            same["view"] == control1["view"] == control2["view"]
            and same["anatomy"] == control1["anatomy"] == control2["anatomy"]
            and same["tumor_type"] == control1["tumor_type"] == control2["tumor_type"]
        ):
            raise RuntimeError("matched control stratum changed")
        output.append(
            {
                "pair_id": pair_id,
                "anchor_image_id": anchor["image_id"],
                "anchor_group_id": group_id,
                "anchor_view": anchor["view"],
                "anchor_image_sha256": anchor["image_sha256"],
                "same_image_id": same["image_id"],
                "same_group_id": group_id,
                "same_view": same["view"],
                "same_image_sha256": same["image_sha256"],
                "control1_image_id": control1["image_id"],
                "control1_group_id": control1["group_id"],
                "control1_view": control1["view"],
                "control1_image_sha256": control1["image_sha256"],
                "control2_image_id": control2["image_id"],
                "control2_group_id": control2["group_id"],
                "control2_view": control2["view"],
                "control2_image_sha256": control2["image_sha256"],
                "anatomy": same["anatomy"],
                "tumor_type": same["tumor_type"],
            }
        )
    return output, {
        "train_tumor_images": len(train_tumor),
        "distinct_view_groups": len(candidate_pairs),
        "matched_pair_rows": len(output),
        "groups_without_two_controls": insufficient_controls,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("cross-view pair output must not exist")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")
    with args.split_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        split_rows = list(csv.DictReader(handle))
    pair_rows, counts = build_pair_rows(split_rows, seed=args.seed)
    if counts != {
        "train_tumor_images": 1488,
        "distinct_view_groups": 443,
        "matched_pair_rows": 384,
        "groups_without_two_controls": 59,
    }:
        raise ValueError(f"canonical cross-view population changed: {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "cross_view_pair_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pair_rows)
    summary = {
        "stage": "rich_gallery_cross_view_pair_manifest_v1",
        "split_sha256": args.expected_split_sha256,
        "seed": args.seed,
        **counts,
        "full_arm": "same_group_positive_vs_control1_negative",
        "control_arm": "control1_positive_vs_control2_negative",
        "matched_fields": ["anatomy", "tumor_type", "target_view"],
        "group_id_limitation": "heuristic_not_published_patient_identifier",
        "spatial_annotations_read": False,
        "validation_images_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "manifest_sha256": sha256_file(manifest_path),
    }
    summary_path = args.output_dir / "cross_view_pair_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pass": True, **summary}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
