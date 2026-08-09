from __future__ import annotations

"""Validation-only spatial endpoint for the frozen DSLL experiment."""

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


DSLL_SOURCES = {
    "tumor_logodds_320",
    "disease_top1_320",
    "disease_top2_320",
    "disease_top3_320",
    "disease_latefusion_320",
}
GENERIC_SOURCES = {"tumor_logodds_320"}
EXPECTED_SUBGROUPS = {"small": 94, "medium": 72, "large": 18}


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / max(1, int(prediction.sum()) + int(target.sum())))


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    return float(intersection / max(1, int(np.logical_or(prediction, target).sum())))


def group(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_by_group(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    return {
        label: float(np.mean([
            float(row[field]) for row in rows
            if label == "overall" or row["size_group"] == label
        ]))
        for label in ("overall", "small", "medium", "large")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-pseudo-manifest-sha256", required=True)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--expected-selection-freeze-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.selection_root is None) != (args.expected_selection_freeze_sha256 is None):
        raise ValueError("Selection root/freeze must be supplied together")
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    candidate_rows, _ = validate_candidate_diagnostics_manifest(
        args.candidate_root,
        expected_image_names=[row["image_id"] for row in split_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
    )
    selections = None
    if args.selection_root is not None:
        freeze_path = args.selection_root / "prediction_freeze.json"
        if sha256_file(freeze_path) != args.expected_selection_freeze_sha256:
            raise ValueError("Selection freeze hash mismatch")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
            or freeze.get("spatial_ground_truth_used") is not False
            or freeze.get("test_evaluated") is not False
        ):
            raise ValueError("Selection was not frozen before GT")
        selection_path = args.selection_root / "selection_manifest.csv"
        if sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
            raise ValueError("Selection manifest changed")
        selections = {row["image_id"]: row for row in read_csv(selection_path)}

    root = resolve_btxrd_root(args.dataset_root)
    per_image: list[dict[str, object]] = []
    opened = 0
    for split_row in split_rows:
        if split_row["tumor"] != "1":
            continue
        opened += 1
        image_id = split_row["image_id"]
        with Image.open(root / "images" / image_id) as image:
            width, height = image.size
        native_target = _decode_labelme_polygon_mask(
            root / "Annotations" / f"{Path(image_id).stem}.json",
            height=height,
            width=width,
        )
        candidate_row = candidate_rows[Path(image_id).stem]
        payload_path = args.candidate_root / candidate_row["diagnostic_path"]
        with np.load(payload_path, allow_pickle=False) as payload:
            masks = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
            maps = payload["dsll_source_maps"].astype(np.float32)
            map_ids = payload["dsll_source_map_ids"].astype(str)
        if len(maps) != 5 or set(map_ids) != DSLL_SOURCES:
            raise ValueError(f"Frozen DSLL map set differs: {image_id}")
        target = np.asarray(
            Image.fromarray(native_target.astype(np.uint8) * 255).resize(
                (masks.shape[2], masks.shape[1]), Image.Resampling.NEAREST
            )
        ) > 0
        area_group = group(float(target.mean()))
        candidate_dice = np.asarray([dice(mask, target) for mask in masks], dtype=np.float64)

        def oracle(allowed: set[str] | None) -> float:
            indices = np.arange(len(sources)) if allowed is None else np.flatnonzero(np.isin(sources, list(allowed)))
            return float(candidate_dice[indices].max()) if len(indices) else 0.0

        record: dict[str, object] = {
            "image_id": image_id,
            "size_group": area_group,
            "candidate_count": len(masks),
            "oracle_generic": oracle(GENERIC_SOURCES),
            "oracle_generic_plus_dsll": oracle(DSLL_SOURCES),
            "oracle_full_7_source": oracle(None),
        }
        for map_values, map_id in zip(maps, map_ids, strict=True):
            for percentile in (85, 90, 95):
                threshold = float(np.percentile(map_values, percentile))
                record[f"cam_{map_id}_p{percentile}_dice"] = dice(
                    map_values >= threshold, target
                )
        if selections is not None:
            selection = selections[image_id]
            prediction = masks[int(selection["selected_candidate_index"])]
            record.update({
                "selected_dice": dice(prediction, target),
                "selected_iou": iou(prediction, target),
                "selected_source": selection["selected_source"],
                "selector_regret": oracle(None) - dice(prediction, target),
            })
        per_image.append(record)
    if opened != 184 or Counter(row["size_group"] for row in per_image) != Counter(EXPECTED_SUBGROUPS):
        raise ValueError("Validation tumor/subgroup cohort differs")
    summary: dict[str, object] = {
        "images": 184,
        "test_images_read": 0,
        "oracle_generic": mean_by_group(per_image, "oracle_generic"),
        "oracle_generic_plus_dsll": mean_by_group(per_image, "oracle_generic_plus_dsll"),
        "oracle_full_7_source": mean_by_group(per_image, "oracle_full_7_source"),
        "pre_sam_dice": {},
    }
    for map_id in sorted(DSLL_SOURCES):
        summary["pre_sam_dice"][map_id] = {
            f"p{percentile}": mean_by_group(per_image, f"cam_{map_id}_p{percentile}_dice")
            for percentile in (85, 90, 95)
        }
    if selections is not None:
        summary["selected_dice"] = mean_by_group(per_image, "selected_dice")
        summary["selected_iou"] = mean_by_group(per_image, "selected_iou")
        summary["selector_regret"] = mean_by_group(per_image, "selector_regret")
        summary["selected_source_counts"] = dict(sorted(Counter(
            str(row["selected_source"]) for row in per_image
        ).items()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_path = args.output_dir / "per_image.csv"
    with per_path.open("w", encoding="utf-8", newline="") as handle:
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
