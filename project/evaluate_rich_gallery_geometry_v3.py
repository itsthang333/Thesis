from __future__ import annotations

"""Minimal post-freeze Dice/IoU evaluator for rich-gallery Geometry-v3."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate_rad_dino_mask_bag_mil_probe import (
    _dice,
    _load_and_verify_predictions,
    _size_group,
)
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


FULLY_DICE = {
    "overall": 0.49513170,
    "small": 0.32895493,
    "medium": 0.66244178,
    "large": 0.69370336,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = int(np.logical_or(prediction, target).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    predictions, freeze = _load_and_verify_predictions(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("Rich-gallery evaluator requires the full validation cohort")
    if (
        freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
    ):
        raise ValueError("Prediction freeze is not bound to this merged gallery")

    # Boundary: polygons are opened only after all 371 maps and the merged
    # candidate gallery have been hash-verified above.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    prediction_by_id = {row["image_id"]: row for row in predictions}
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_name = dataset[index]
        prediction = prediction_by_id[str(image_name)]
        if prediction["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        values = np.load(
            args.prediction_root / "predictions" / prediction["map_path"],
            allow_pickle=False,
        ).astype(np.float32)
        selected = values > 0.0
        candidate_row = candidate_rows[Path(str(image_name)).stem]
        with np.load(
            args.val_candidate_root / candidate_row["diagnostic_path"],
            allow_pickle=False,
        ) as payload:
            proposals = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
        proposal_dice = np.asarray([_dice(mask, target) for mask in proposals])
        oracle_index = int(np.argmax(proposal_dice)) if len(proposal_dice) else -1
        area_ratio = float(target.mean())
        overlap = np.logical_and(selected, target)
        per_image.append(
            {
                "image_id": str(image_name),
                "group_id": prediction["group_id"],
                "gt_area_ratio": area_ratio,
                "size_group": _size_group(area_ratio),
                "dice": _dice(selected, target),
                "iou": iou(selected, target),
                "complete_miss": int(not overlap.any()),
                "selected_area_ratio": float(selected.mean()),
                "candidate_count": len(proposals),
                "oracle_dice": float(proposal_dice[oracle_index]) if oracle_index >= 0 else 0.0,
                "oracle_source": str(sources[oracle_index]) if oracle_index >= 0 else "none",
            }
        )
    if len(per_image) != 184:
        raise RuntimeError(f"Expected 184 tumor images, got {len(per_image)}")
    subgroup_counts = {
        name: sum(row["size_group"] == name for row in per_image)
        for name in ("small", "medium", "large")
    }
    if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"Subgroup mismatch: {subgroup_counts}")

    metrics: dict[str, dict[str, float | int]] = {}
    for subgroup in ("overall", "small", "medium", "large"):
        rows = [
            row
            for row in per_image
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        dice = float(np.mean([float(row["dice"]) for row in rows]))
        metrics[subgroup] = {
            "n": len(rows),
            "dice": dice,
            "iou": float(np.mean([float(row["iou"]) for row in rows])),
            "gap_to_fully_dice": dice - FULLY_DICE[subgroup],
            "fully_dice": FULLY_DICE[subgroup],
            "complete_misses": int(sum(int(row["complete_miss"]) for row in rows)),
            "oracle_dice": float(np.mean([float(row["oracle_dice"]) for row in rows])),
        }
    image_auroc = float(
        roc_auc_score(
            [int(row["tumor"]) for row in predictions],
            [float(row["bag_probability"]) for row in predictions],
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary = {
        "stage": "rich_gallery_geometry_v3_post_freeze_evaluation",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187, **subgroup_counts},
        "image_level_auroc": image_auroc,
        "tumor_segmentation": metrics,
        "complete_misses_included": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "spatial_ground_truth_used_for_training": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "audit_pass": True,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "validation_gt_read_only_after_prediction_freeze": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
