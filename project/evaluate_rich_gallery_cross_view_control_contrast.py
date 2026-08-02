from __future__ import annotations

"""Validation-only actual-Dice evaluation of frozen control contrast choices."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


GROUPS = ("small", "medium", "large")
BASELINE_DICE = 0.28872948670665205


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--expected-stage-a-freeze-sha256", required=True)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--expected-per-candidate-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    required = (
        "image_id", "group_id", "size_group", "candidate_local_index", "source",
        "candidate_area_ratio", "candidate_dice", "is_eligible_oracle", "is_baseline_selected",
    )
    bags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: header.index(name) for name in required}
        for raw in reader:
            image_id = Path(raw[positions["image_id"]]).stem
            bags[image_id].append(
                {
                    "group_id": raw[positions["group_id"]],
                    "size_group": raw[positions["size_group"]],
                    "local": int(raw[positions["candidate_local_index"]]),
                    "source": raw[positions["source"]],
                    "area": float(raw[positions["candidate_area_ratio"]]),
                    "dice": float(raw[positions["candidate_dice"]]),
                    "oracle": int(raw[positions["is_eligible_oracle"]]),
                    "baseline": int(raw[positions["is_baseline_selected"]]),
                }
            )
    for image_id, rows in bags.items():
        rows.sort(key=lambda row: row["local"])
        if [row["local"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"candidate local indices changed: {image_id}")
    return dict(bags)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("control-contrast Stage-B output must not exist")
    freeze_path = args.stage_a_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_stage_a_freeze_sha256:
        raise ValueError("Stage-A freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest_path = args.stage_a_root / "selection_manifest.csv"
    if (
        freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
        or sha256_file(manifest_path) != freeze["selection_manifest_sha256"]
    ):
        raise ValueError("Stage-A freeze contract mismatch")
    if sha256_file(args.per_candidate) != args.expected_per_candidate_sha256:
        raise ValueError("per-candidate SHA-256 mismatch")

    bags = read_candidates(args.per_candidate)
    if len(bags) != 184 or sum(map(len, bags.values())) != 32519:
        raise ValueError("expected exact 184-image/32,519-candidate tumor table")
    counts = Counter(rows[0]["size_group"] for rows in bags.values())
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise ValueError(f"subgroup counts changed: {counts}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        selections = list(csv.DictReader(handle))
    variants = freeze["variants"]
    indexed = {(row["variant"], Path(row["image_id"]).stem): row for row in selections}
    if len(indexed) != 371 * len(variants):
        raise ValueError("frozen selection cohort incomplete")

    per_image: list[dict[str, Any]] = []
    for variant in variants:
        for image_id, candidates in sorted(bags.items()):
            selected_row = indexed[(variant, image_id)]
            selected = int(selected_row["selected_local_index"])
            if selected < 0 or selected >= len(candidates):
                raise ValueError(f"frozen candidate index out of range: {(variant, image_id)}")
            candidate = candidates[selected]
            if selected_row["selected_source"] != candidate["source"]:
                raise ValueError(f"frozen candidate source changed: {(variant, image_id)}")
            oracle_dice = max(row["dice"] for row in candidates)
            ratio = candidate["area"] / max(1.0e-12, next(row["area"] for row in candidates if row["baseline"] == 1))
            dice = float(candidate["dice"])
            per_image.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": candidate["group_id"],
                    "size_group": candidate["size_group"],
                    "dice": dice,
                    "iou": float(dice / (2.0 - dice)) if dice < 2.0 else 1.0,
                    "complete_miss": int(dice == 0.0),
                    "selected_source": candidate["source"],
                    "selected_area_ratio": candidate["area"],
                    "selected_to_baseline_area_ratio": ratio,
                    "eligible_oracle_dice": oracle_dice,
                    "selector_regret": oracle_dice - dice,
                }
            )

    metrics: dict[str, dict[str, Any]] = {}
    for variant in variants:
        current = [row for row in per_image if row["variant"] == variant]
        metrics[variant] = {}
        for group in ("overall", *GROUPS):
            rows = [row for row in current if group == "overall" or row["size_group"] == group]
            metrics[variant][group] = {
                "n": len(rows),
                "dice": float(np.mean([row["dice"] for row in rows])),
                "iou": float(np.mean([row["iou"] for row in rows])),
                "complete_misses": int(sum(row["complete_miss"] for row in rows)),
                "selector_regret": float(np.mean([row["selector_regret"] for row in rows])),
                "selected_source_counts": dict(sorted(Counter(row["selected_source"] for row in rows).items())),
                "selected_area_ratio_median": float(np.median([row["selected_area_ratio"] for row in rows])),
            }
    if abs(metrics["baseline"]["overall"]["dice"] - BASELINE_DICE) > 1.0e-12:
        raise ValueError("immutable baseline did not reproduce")
    best_variant = max(variants, key=lambda variant: metrics[variant]["overall"]["dice"])
    result = {
        "stage": "rich_gallery_cross_view_control_contrast_stage_b_v1",
        "inputs": {
            "stage_a_freeze_sha256": args.expected_stage_a_freeze_sha256,
            "per_candidate_sha256": args.expected_per_candidate_sha256,
        },
        "cohort": {"tumor": 184, "small": 94, "medium": 72, "large": 18},
        "actual_binary_mask_metrics": metrics,
        "decision": {
            "best_variant": best_variant,
            "best_overall_dice": metrics[best_variant]["overall"]["dice"],
            "beats_immutable_baseline": metrics[best_variant]["overall"]["dice"] > BASELINE_DICE,
            "delta_vs_immutable_baseline": metrics[best_variant]["overall"]["dice"] - BASELINE_DICE,
            "selection_status": "exploratory_validation_global_policy",
        },
        "academic_status": {
            "candidate_choices_frozen_before_validation_gt": True,
            "validation_gt_read_only_after_freeze": True,
            "spatial_ground_truth_used_for_training_or_selection": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "summary_sha256": sha256_file(summary_path),
        "per_image_sha256": sha256_file(per_image_path),
        "stage_a_freeze_sha256": args.expected_stage_a_freeze_sha256,
        "tumor_images": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
