from __future__ import annotations

"""Spatial validation endpoint for an independently frozen HR-CBPMIL-IE+ run."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import _decode_labelme_polygon_mask, resolve_btxrd_root
from frozen_io import load_split_rows_without_annotations, sha256_file
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPECTED_GROUPS = {"small": 94, "medium": 72, "large": 18}
BASELINE = {"overall": 0.2887294867, "small": 0.157723, "medium": 0.435229, "large": 0.386874}


def metric_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    return {
        "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "complete_miss": int(tp == 0),
    }


def size_group(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for group in ("overall", "small", "medium", "large"):
        selected = [row for row in rows if group == "overall" or row["size_group"] == group]
        output[group] = {
            "n": len(selected),
            "dice": float(np.mean([row["dice"] for row in selected])),
            "iou": float(np.mean([row["iou"] for row in selected])),
            "precision": float(np.mean([row["precision"] for row in selected])),
            "recall": float(np.mean([row["recall"] for row in selected])),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in selected)),
            "candidate_oracle_dice": float(np.mean([row["candidate_oracle_dice"] for row in selected])),
            "selector_regret": float(np.mean([row["selector_regret"] for row in selected])),
            "delta_vs_baseline": float(np.mean([row["dice"] for row in selected]) - BASELINE[group]),
            "source_counts": dict(sorted(Counter(str(row["selected_source"]) for row in selected).items())),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--pseudo-manifest-sha256", required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--expected-selection-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    candidate_rows, _ = validate_candidate_diagnostics_manifest(
        args.candidate_root,
        expected_image_names=[row["image_id"] for row in rows],
        split="val",
        expected_pseudo_manifest_sha256=args.pseudo_manifest_sha256,
        expected_manifest_sha256=args.candidate_manifest_sha256,
    )
    freeze_path = args.selection_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_selection_freeze_sha256:
        raise ValueError("Prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "hr_cbpmil_ie_plus_prediction_freeze_v1"
        or freeze.get("images") != 371
        or freeze.get("tumor_images") != 184
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
    ):
        raise ValueError("Prediction freeze violates the annotation boundary")
    manifest_path = args.selection_root / "selection_manifest.csv"
    if sha256_file(manifest_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("Selection manifest changed after freeze")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        selections = {row["image_id"]: row for row in csv.DictReader(handle)}
    if set(selections) != {row["image_id"] for row in rows}:
        raise ValueError("Selection cohort differs from canonical validation")

    # Annotation boundary: all 371 choices and hashes are immutable above here.
    root = resolve_btxrd_root(args.dataset_root)
    per_image: list[dict[str, object]] = []
    opened = 0
    for row in rows:
        if row["tumor"] != "1":
            continue
        opened += 1
        image_id = row["image_id"]
        selection = selections[image_id]
        mask_path = args.selection_root / selection["mask_path"]
        if sha256_file(mask_path) != selection["mask_sha256"]:
            raise ValueError(f"Frozen selected mask changed: {image_id}")
        prediction = np.load(mask_path, allow_pickle=False).astype(bool)
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != selection["candidate_payload_sha256"]:
            raise ValueError(f"Candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            candidate_masks = payload["sam_masks"].astype(bool)
        with Image.open(root / "images" / image_id) as image:
            width, height = image.size
        native_target = _decode_labelme_polygon_mask(
            root / "Annotations" / f"{Path(image_id).stem}.json", height=height, width=width
        )
        target = np.asarray(
            Image.fromarray(native_target.astype(np.uint8) * 255).resize((320, 320), Image.Resampling.NEAREST)
        ) > 0
        metrics = metric_counts(prediction, target)
        candidate_dice = [metric_counts(mask, target)["dice"] for mask in candidate_masks]
        oracle = float(max(candidate_dice))
        per_image.append({
            "image_id": image_id,
            "size_group": size_group(float(target.mean())),
            **metrics,
            "candidate_count": len(candidate_masks),
            "candidate_oracle_dice": oracle,
            "selector_regret": oracle - float(metrics["dice"]),
            "selected_source": selection["selected_source"],
            "selected_candidate_index": int(selection["selected_candidate_index"]),
        })
    counts = Counter(str(row["size_group"]) for row in per_image)
    if opened != 184 or counts != Counter(EXPECTED_GROUPS):
        raise ValueError(f"Canonical tumor/subgroup counts differ: opened={opened}, groups={counts}")
    summary = {
        "stage": "hr_cbpmil_ie_plus_spatial_validation_v1",
        "metrics": summarize(per_image),
        "baseline": BASELINE,
        "validation_annotations_opened": opened,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_path = args.output_dir / "per_image.csv"
    with per_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary["per_image_sha256"] = sha256_file(per_path)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
