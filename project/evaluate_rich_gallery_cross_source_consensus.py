from __future__ import annotations

"""Evaluate frozen cross-source-consensus choices on canonical validation only."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_cross_source_consensus import VARIANTS
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_g2_selector_pair import canonical_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 1.0


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def verify_stage_a(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, object]]:
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("consensus prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_cross_source_consensus_freeze_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("validation_images") != 371
        or set(freeze.get("variants", [])) != set(VARIANTS)
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("consensus freeze contract mismatch")
    manifest = args.prediction_root / "selection_manifest.csv"
    if sha256_file(manifest) != freeze["selection_manifest_sha256"]:
        raise ValueError("consensus selection manifest changed")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cohort = {row["image_id"] for row in val_rows}
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["variant"], row["image_id"])
        if row["variant"] not in VARIANTS or row["image_id"] not in cohort or key in indexed:
            raise ValueError("invalid consensus selection identity")
        indexed[key] = row
    if len(indexed) != 371 * len(VARIANTS):
        raise ValueError("consensus selection cohort is incomplete")
    return indexed, freeze


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("canonical validation cohort mismatch")
    selections, freeze = verify_stage_a(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("consensus evaluation requires the full validation gallery")

    # Annotation boundary: every choice was made and hash-frozen above.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_id = dataset[index]
        image_id = str(image_id)
        if selections[(VARIANTS[0], image_id)]["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
        candidate_dice = np.asarray([dice(mask, target) for mask in proposals])
        oracle_dice = float(candidate_dice.max())
        area = float(target.mean())
        subgroup = size_group(area)
        for variant in VARIANTS:
            selection = selections[(variant, image_id)]
            selected_index = int(selection["selected_candidate_index"])
            prediction = proposals[selected_index]
            selected_dice = dice(prediction, target)
            per_image.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": selection["group_id"],
                    "size_group": subgroup,
                    "gt_area_ratio": area,
                    "dice": selected_dice,
                    "iou": iou(prediction, target),
                    "complete_miss": int(not np.logical_and(prediction, target).any()),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_gt_area_ratio": float(prediction.mean() / area),
                    "selected_source": canonical_source(sources[selected_index]),
                    "selected_consensus_iou": float(selection["selected_consensus_iou"]),
                    "oracle_dice": oracle_dice,
                    "selector_regret": oracle_dice - selected_dice,
                }
            )
    if len(per_image) != 184 * len(VARIANTS):
        raise RuntimeError("consensus tumor evaluation count mismatch")
    subgroup_counts = Counter(
        row["size_group"] for row in per_image if row["variant"] == VARIANTS[0]
    )
    if subgroup_counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"consensus subgroup mismatch: {subgroup_counts}")

    summary: dict[str, dict[str, dict[str, object]]] = {}
    baseline = "g1_upstream_baseline"
    baseline_records = [row for row in per_image if row["variant"] == baseline]
    baseline_dice = {
        subgroup: float(
            np.mean(
                [
                    row["dice"]
                    for row in baseline_records
                    if subgroup == "overall" or row["size_group"] == subgroup
                ]
            )
        )
        for subgroup in ("overall", "small", "medium", "large")
    }
    for variant in VARIANTS:
        records = [row for row in per_image if row["variant"] == variant]
        summary[variant] = {}
        for subgroup in ("overall", "small", "medium", "large"):
            selected = [
                row for row in records if subgroup == "overall" or row["size_group"] == subgroup
            ]
            mean_dice = float(np.mean([row["dice"] for row in selected]))
            summary[variant][subgroup] = {
                "n": len(selected),
                "dice": mean_dice,
                "iou": float(np.mean([row["iou"] for row in selected])),
                "complete_misses": int(sum(row["complete_miss"] for row in selected)),
                "oracle_dice": float(np.mean([row["oracle_dice"] for row in selected])),
                "selector_regret": float(np.mean([row["selector_regret"] for row in selected])),
                "selected_gt_area_ratio_median": float(
                    np.median([row["selected_gt_area_ratio"] for row in selected])
                ),
                "selected_consensus_iou_median": float(
                    np.median([row["selected_consensus_iou"] for row in selected])
                ),
                "selected_source_counts": dict(
                    sorted(Counter(row["selected_source"] for row in selected).items())
                ),
                "delta_vs_g1_upstream_baseline": mean_dice - baseline_dice[subgroup],
            }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    result = {
        "stage": "rich_gallery_cross_source_consensus_post_freeze_evaluation_v1",
        "cohort": {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            "small": 94,
            "medium": 72,
            "large": 18,
        },
        "actual_binary_mask_metrics": summary,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "spatial_ground_truth_used_for_selection": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "selection_manifest_sha256": freeze["selection_manifest_sha256"],
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
