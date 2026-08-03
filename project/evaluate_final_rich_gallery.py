from __future__ import annotations

"""Spatial-GT evaluator for already frozen validation or final-test choices."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from evaluation.frozen_test_guard import verify_frozen_test_config


EXPECTED_DICE = 0.28872948670665205
EXPECTED_VAL_COUNTS = {"overall": 184, "small": 94, "medium": 72, "large": 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--expected-selection-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-overall-dice", type=float)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / max(1, int(prediction.sum()) + int(target.sum())))


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    return float(intersection / max(1, int(np.logical_or(prediction, target).sum())))


def _size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for group in ("overall", "small", "medium", "large"):
        selected = [row for row in rows if group == "overall" or row["size_group"] == group]
        summary[group] = {
            "n": len(selected),
            "dice": float(np.mean([row["dice"] for row in selected])),
            "iou": float(np.mean([row["iou"] for row in selected])),
            "precision": float(np.mean([row["precision"] for row in selected])),
            "recall": float(np.mean([row["recall"] for row in selected])),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in selected)),
            "median_selected_gt_area_ratio": float(
                np.median([row["selected_gt_area_ratio"] for row in selected])
            ),
            "source_counts": dict(sorted(Counter(str(row["source"]) for row in selected).items())),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
        allow_test=args.split == "test",
    )
    split_by_id = {row["image_id"]: row for row in split_rows}
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
    )
    if args.split == "test" and args.expected_overall_dice is not None:
        raise ValueError("test Dice must not be pre-targeted; omit --expected-overall-dice")

    freeze_path = args.selection_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_selection_freeze_sha256:
        raise ValueError("selection freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "final_rich_gallery_choice_freeze_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("cohort_split", "val") != args.split
        or freeze.get("images") != len(split_rows)
        or freeze.get("tumor_images") != sum(int(row["tumor"]) for row in split_rows)
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != (len(split_rows) if args.split == "test" else 0)
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("selection freeze violates the validation boundary")
    selection_path = args.selection_root / "selection_manifest.csv"
    if sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("selection manifest changed after freezing")
    selection_rows = _read_csv(selection_path)
    selections = {row["image_id"]: row for row in selection_rows}
    if len(selections) != len(split_rows) or set(selections) != set(split_by_id):
        raise ValueError(f"selection cohort differs from canonical {args.split}")

    # Annotation boundary: every candidate choice is immutable above this line.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split=args.split,
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    per_image: list[dict[str, object]] = []
    opened_annotations = 0
    for index in range(len(dataset)):
        _image, mask_tensor, image_id = dataset[index]
        image_id = str(image_id)
        if split_by_id[image_id]["tumor"] != "1":
            continue
        opened_annotations += 1
        target = mask_tensor[0].numpy() > 0.5
        selection = selections[image_id]
        candidate_path = args.candidate_root / "candidate_diagnostics" / f"{Path(image_id).stem}.npz"
        if sha256_file(candidate_path) != selection["candidate_payload_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            prediction = payload["sam_masks"][int(selection["selected_candidate_index"])].astype(bool)
        intersection = int(np.logical_and(prediction, target).sum())
        pred_area = int(prediction.sum())
        gt_area = int(target.sum())
        per_image.append(
            {
                "image_id": image_id,
                "group_id": split_by_id[image_id]["group_id"],
                "size_group": _size_group(float(target.mean())),
                "dice": _dice(prediction, target),
                "iou": _iou(prediction, target),
                "precision": float(intersection / max(1, pred_area)),
                "recall": float(intersection / max(1, gt_area)),
                "complete_miss": int(intersection == 0),
                "selected_gt_area_ratio": float(pred_area / max(1, gt_area)),
                "source": selection["selected_source"],
                "selected_candidate_index": int(selection["selected_candidate_index"]),
            }
        )
    expected_tumor = sum(int(row["tumor"]) for row in split_rows)
    if opened_annotations != expected_tumor:
        raise ValueError(f"opened {opened_annotations} tumor annotations, expected {expected_tumor}")

    summary = _summarize(per_image)
    counts = {group: int(summary[group]["n"]) for group in ("overall", "small", "medium", "large")}
    if args.split == "val" and counts != EXPECTED_VAL_COUNTS:
        raise ValueError(f"validation subgroup counts differ: {counts}")
    expected_dice = EXPECTED_DICE if args.split == "val" and args.expected_overall_dice is None else args.expected_overall_dice
    if expected_dice is not None and abs(float(summary["overall"]["dice"]) - expected_dice) > 1.0e-12:
        raise ValueError("final validation Dice did not reproduce the frozen result")

    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    report = {
        "method": "G1 + fixed equal percentile-rank fusion",
        "split": args.split,
        "selection_freeze_sha256": args.expected_selection_freeze_sha256,
        "split_sha256": args.expected_split_sha256,
        "summary": summary,
        "candidate_choices_frozen_before_spatial_gt": True,
        "candidate_choices_frozen_before_validation_gt": args.split == "val",
        "candidate_choices_frozen_before_test_gt": args.split == "test",
        "spatial_annotations_opened": opened_annotations,
        "test_images_read": len(split_rows) if args.split == "test" else 0,
        "test_evaluated": args.split == "test",
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "pass": True,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(report_path),
        "overall_dice_reproduced": args.split == "val",
        "test_evaluated": args.split == "test",
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
