from __future__ import annotations

"""Evaluate frozen candidate diagnostics after prediction generation completes.

Ordering is a protocol invariant: pseudo masks and all candidate NPZ files are
hash-verified before the segmentation dataset is constructed or any GT mask is
read.  Test is intentionally unsupported.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.factory import build_classification_dataset, build_segmentation_dataset
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from pseudo.manifest import sha256_file, validate_pseudo_mask_manifest
from pseudo.oracle_diagnostics import binary_overlap_metrics, oracle_vs_selected_metrics
from pseudo.prompt_metrics import (
    binary_mask_localization_metrics,
    box_prompt_localization_metrics,
    cam_localization_metrics,
    negative_point_rejection_rate,
    point_prompt_hit_rate,
)


FIELDS = [
    "image_name",
    "tumor_type",
    "tumor_area_ratio",
    "size_group",
    "generation_status",
    "foreground_iou",
    "foreground_recall",
    "foreground_precision",
    "point_hit_rate",
    "num_points",
    "num_hits",
    "negative_rejection_rate",
    "num_negative_points",
    "num_false_negative_points",
    "box_recall",
    "box_precision",
    "oracle_best_single_dice",
    "oracle_best_single_dice_clipped",
    "selected_dice",
    "oracle_gap_dice",
    "support_loss_dice",
    "selection_loss_dice",
    "final_dice",
    "final_iou",
    "postprocess_delta_dice",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hash-locked prediction-first candidate diagnostics"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pseudo-output-dir", type=Path, required=True)
    parser.add_argument("--expected-pseudo-manifest-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _finite_mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    pseudo_output = args.pseudo_output_dir.resolve()
    pseudo_manifest = pseudo_output / "pseudo_mask_manifest.csv"
    if not pseudo_manifest.is_file():
        raise FileNotFoundError(f"Missing pseudo-mask manifest: {pseudo_manifest}")
    actual_pseudo_hash = sha256_file(pseudo_manifest)
    if actual_pseudo_hash != args.expected_pseudo_manifest_sha256:
        raise ValueError("Pseudo-mask manifest differs from the caller-locked hash")

    # Image labels and split membership are allowed WSL inputs. No polygon/GT
    # mask is constructed or loaded during this entire integrity phase.
    classification_dataset = build_classification_dataset(
        root=args.data_root,
        split=args.split,
        target_columns=["tumor"],
        image_size=32,
        augment=False,
        split_manifest=args.split_manifest,
    )
    pseudo_audit = validate_pseudo_mask_manifest(
        pseudo_output / "masks",
        classification_dataset.samples,
        split=args.split,
        image_size=None,
    )
    if pseudo_audit["manifest_sha256"] != actual_pseudo_hash:
        raise ValueError("Pseudo-mask validator returned an inconsistent manifest hash")
    expected_tumor_names = [
        str(sample["image_id"])
        for sample in classification_dataset.samples
        if bool(sample.get("tumor", 0))
    ]
    diagnostic_rows, diagnostic_audit = validate_candidate_diagnostics_manifest(
        pseudo_output,
        expected_image_names=expected_tumor_names,
        split=args.split,
        expected_pseudo_manifest_sha256=actual_pseudo_hash,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
    )

    # Protocol boundary: GT becomes accessible only after every prediction
    # artifact and cohort/hash contract above has passed.
    image_size = int(diagnostic_audit["image_size"])
    segmentation_dataset = build_segmentation_dataset(
        root=args.data_root,
        split=args.split,
        image_size=image_size,
        augment=False,
        split_manifest=args.split_manifest,
    )
    gt_by_stem: dict[str, np.ndarray] = {}
    for index in range(len(segmentation_dataset)):
        _, mask_tensor, image_name = segmentation_dataset[index]
        stem = Path(str(image_name)).stem
        if stem in diagnostic_rows:
            gt_by_stem[stem] = mask_tensor[0].numpy() > 0.5
    if set(gt_by_stem) != set(diagnostic_rows):
        missing = sorted(set(diagnostic_rows) - set(gt_by_stem))
        raise RuntimeError(f"GT diagnostic cohort is incomplete after freeze: {missing[:5]}")

    selection_method = str(diagnostic_audit["selection_method"])
    support_clip_kernel = int(diagnostic_audit["support_clip_kernel"])
    cam_percentile = float(diagnostic_audit["cam_percentile"])
    evaluated: list[dict[str, object]] = []
    for stem in sorted(diagnostic_rows):
        manifest_row = diagnostic_rows[stem]
        gt_mask = gt_by_stem[stem]
        path = pseudo_output / manifest_row["diagnostic_path"]
        with np.load(path, allow_pickle=False) as payload:
            sam_masks = payload["sam_masks"].astype(bool)
            refined = payload["refined_mask"].astype(bool)
            final_mask = payload["final_mask"].astype(bool)
            support_present = bool(int(payload["bone_support_present"][0]))
            bone_support = payload["bone_support"].astype(bool) if support_present else None
            if bone_support is not None:
                foreground = binary_mask_localization_metrics(bone_support, gt_mask)
            else:
                cam_metrics = cam_localization_metrics(
                    payload["prompt_map"], gt_mask, percentile=cam_percentile
                )
                foreground = {
                    "iou": cam_metrics["cam_iou"],
                    "recall": cam_metrics["cam_recall"],
                    "precision": cam_metrics["cam_precision"],
                }
            positive = [tuple(map(int, point)) for point in payload["positive_points"]]
            negative = [tuple(map(int, point)) for point in payload["negative_points"]]
            boxes = [tuple(map(int, box)) for box in payload["boxes"]]
            point_metrics = point_prompt_hit_rate(positive, gt_mask)
            negative_metrics = negative_point_rejection_rate(negative, gt_mask)
            box_metrics = box_prompt_localization_metrics(boxes, gt_mask)
            oracle = oracle_vs_selected_metrics(
                sam_masks,
                refined,
                gt_mask,
                bone_support=bone_support,
                selection_method=selection_method,
                support_clip_kernel=support_clip_kernel,
            )
            final = binary_overlap_metrics(final_mask, gt_mask)
            selected = binary_overlap_metrics(refined, gt_mask)

        area_ratio = float(gt_mask.mean())
        evaluated.append(
            {
                "image_name": manifest_row["image_name"],
                "tumor_type": manifest_row.get("tumor_type", ""),
                "tumor_area_ratio": area_ratio,
                "size_group": _size_group(area_ratio),
                "generation_status": manifest_row.get("generation_status", ""),
                "foreground_iou": foreground["iou"],
                "foreground_recall": foreground["recall"],
                "foreground_precision": foreground["precision"],
                "point_hit_rate": point_metrics["point_hit_rate"],
                "num_points": point_metrics["num_points"],
                "num_hits": point_metrics["num_hits"],
                "negative_rejection_rate": negative_metrics["negative_rejection_rate"],
                "num_negative_points": negative_metrics["num_negative_points"],
                "num_false_negative_points": negative_metrics["num_false_negatives"],
                "box_recall": box_metrics["box_recall"],
                "box_precision": box_metrics["box_precision"],
                "oracle_best_single_dice": oracle["best_single_dice"],
                "oracle_best_single_dice_clipped": oracle["best_single_dice_clipped"],
                "selected_dice": selected["dice"],
                "oracle_gap_dice": (
                    oracle["best_single_dice"] - selected["dice"]
                    if np.isfinite(oracle["best_single_dice"])
                    else float("nan")
                ),
                "support_loss_dice": oracle["support_loss_dice"],
                "selection_loss_dice": (
                    oracle["best_single_dice_clipped"] - selected["dice"]
                    if np.isfinite(oracle["best_single_dice_clipped"])
                    else float("nan")
                ),
                "final_dice": final["dice"],
                "final_iou": final["iou"],
                "postprocess_delta_dice": final["dice"] - selected["dice"],
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "prompt_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(evaluated)

    subgroup = {}
    for group in ("small", "medium", "large"):
        group_rows = [row for row in evaluated if row["size_group"] == group]
        subgroup[group] = {
            "n": len(group_rows),
            "mean_final_dice": _finite_mean(group_rows, "final_dice"),
            "mean_foreground_recall": _finite_mean(group_rows, "foreground_recall"),
            "mean_oracle_best_single_dice": _finite_mean(
                group_rows, "oracle_best_single_dice"
            ),
            "mean_selected_dice": _finite_mean(group_rows, "selected_dice"),
        }
    summary = {
        "schema_version": 1,
        "protocol": "prediction_first_gt_after_hash_freeze",
        "split": args.split,
        "test_accessed": False,
        "tumor_images": len(evaluated),
        "pseudo_manifest_sha256": actual_pseudo_hash,
        "candidate_manifest_sha256": diagnostic_audit["manifest_sha256"],
        "candidate_summary_sha256": diagnostic_audit["summary_sha256"],
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "prompt_quality_sha256": sha256_file(csv_path),
        "means": {
            key: _finite_mean(evaluated, key)
            for key in (
                "foreground_iou",
                "foreground_recall",
                "point_hit_rate",
                "oracle_best_single_dice",
                "selected_dice",
                "final_dice",
                "postprocess_delta_dice",
            )
        },
        "subgroups": subgroup,
    }
    summary_path = output_dir / "prompt_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "status": "PASS",
        "prediction_artifacts_verified_before_gt_load": True,
        "complete_misses_included": True,
        "test_accessed": False,
        "summary_sha256": sha256_file(summary_path),
    }
    (output_dir / "prediction_first_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: evaluated {len(evaluated)} frozen tumor predictions; "
        f"mean final Dice={summary['means']['final_dice']:.6f}; "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
