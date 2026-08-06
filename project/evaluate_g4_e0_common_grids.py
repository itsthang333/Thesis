from __future__ import annotations

"""E0: compare frozen WSSS and fully masks on native, 320, and 448 grids."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.btxrd import _decode_labelme_polygon_mask, resolve_btxrd_root
from evaluation.segmentation_metrics import (
    bootstrap_group_confidence_intervals,
    json_safe,
    segmentation_metrics,
    summarize_segmentation_rows,
)
from frozen_io import load_split_rows_without_annotations, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--wsss-choice-root", type=Path, required=True)
    parser.add_argument("--expected-wsss-freeze-sha256", required=True)
    parser.add_argument("--fully-mask-root", type=Path, required=True)
    parser.add_argument("--expected-fully-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resize(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return np.asarray(mask, dtype=bool)
    height, width = shape
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0


def size_group(ratio: float) -> str:
    return "small_lt_1pct" if ratio < 0.01 else (
        "medium_1_to_5pct" if ratio < 0.05 else "large_ge_5pct"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("E0 requires canonical 371/184 validation")
    split_by_id = {row["image_id"]: row for row in split_rows}

    candidate_manifest_path = args.candidate_root / "candidate_diagnostics_manifest.csv"
    if sha256_file(candidate_manifest_path) != args.expected_candidate_manifest_sha256:
        raise ValueError("candidate manifest mismatch")
    candidate_by_id = {row["image_name"]: row for row in read_csv(candidate_manifest_path)}

    wsss_freeze_path = args.wsss_choice_root / "prediction_freeze.json"
    if sha256_file(wsss_freeze_path) != args.expected_wsss_freeze_sha256:
        raise ValueError("WSSS freeze mismatch")
    wsss_freeze = json.loads(wsss_freeze_path.read_text(encoding="utf-8"))
    wsss_manifest_path = args.wsss_choice_root / "selection_manifest.csv"
    if sha256_file(wsss_manifest_path) != wsss_freeze["selection_manifest_sha256"]:
        raise ValueError("WSSS selection manifest changed")
    wsss_by_id = {row["image_id"]: row for row in read_csv(wsss_manifest_path)}

    fully_freeze_path = args.fully_mask_root / "mask_freeze.json"
    if sha256_file(fully_freeze_path) != args.expected_fully_freeze_sha256:
        raise ValueError("fully mask freeze mismatch")
    fully_freeze = json.loads(fully_freeze_path.read_text(encoding="utf-8"))
    if (
        fully_freeze.get("split_sha256") != args.expected_split_sha256
        or fully_freeze.get("masks_frozen_before_spatial_ground_truth") is not True
        or fully_freeze.get("spatial_ground_truth_used") is not False
        or fully_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("fully mask freeze violates E0 boundary")
    fully_manifest_path = args.fully_mask_root / "mask_manifest.csv"
    if sha256_file(fully_manifest_path) != fully_freeze["mask_manifest_sha256"]:
        raise ValueError("fully mask manifest changed")
    fully_by_id = {row["image_id"]: row for row in read_csv(fully_manifest_path)}
    if not (set(candidate_by_id) == set(wsss_by_id) == set(fully_by_id) == set(split_by_id)):
        raise ValueError("E0 frozen cohorts differ")

    # Verify every prediction byte before opening one spatial annotation.
    candidate_paths: dict[str, Path] = {}
    fully_paths: dict[str, Path] = {}
    for image_id in sorted(split_by_id):
        candidate_path = args.candidate_root / candidate_by_id[image_id]["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_by_id[image_id]["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        candidate_paths[image_id] = candidate_path
        fully_path = args.fully_mask_root / fully_by_id[image_id]["mask_path"]
        if sha256_file(fully_path) != fully_by_id[image_id]["mask_sha256"]:
            raise ValueError(f"fully mask changed: {image_id}")
        fully_paths[image_id] = fully_path

    btxrd_root = resolve_btxrd_root(args.dataset_root)
    results: dict[str, list[dict[str, object]]] = defaultdict(list)
    migration: dict[tuple[str, str], int] = defaultdict(int)
    oracle_dice: list[float] = []
    opened = 0
    for split_row in split_rows:
        image_id = split_row["image_id"]
        tumor = int(split_row["tumor"]) == 1
        with Image.open(btxrd_root / "images" / image_id) as handle:
            native_width, native_height = handle.size
        native_shape = (native_height, native_width)
        if tumor:
            target_native = _decode_labelme_polygon_mask(
                btxrd_root / "Annotations" / f"{Path(image_id).stem}.json",
                height=native_height,
                width=native_width,
            ).astype(bool)
            opened += 1
        else:
            target_native = np.zeros(native_shape, dtype=bool)
        native_ratio = float(target_native.mean())
        native_group = size_group(native_ratio) if tumor else "normal"

        with np.load(candidate_paths[image_id], allow_pickle=False) as payload:
            masks = payload["sam_masks"].astype(bool)
            selected = int(wsss_by_id[image_id]["selected_candidate_index"])
            wsss_320 = masks[selected] if tumor and selected >= 0 else np.zeros((320, 320), bool)
            if tumor:
                target_320 = resize(target_native, (320, 320))
                intersections = np.logical_and(masks, target_320[None]).sum(axis=(1, 2))
                denominators = masks.sum(axis=(1, 2)) + int(target_320.sum())
                dice = np.divide(
                    2 * intersections,
                    denominators,
                    out=np.zeros(len(masks), dtype=np.float64),
                    where=denominators > 0,
                )
                oracle_dice.append(float(dice.max()))
                resized_group = size_group(float(target_320.mean()))
                migration[(resized_group, native_group)] += 1
        with Image.open(fully_paths[image_id]) as handle:
            fully_448 = np.asarray(handle.convert("L")) > 0

        for method, base_mask in (("wsss", wsss_320), ("fully", fully_448)):
            for grid, shape in (("native", native_shape), ("320", (320, 320)), ("448", (448, 448))):
                target = resize(target_native, shape)
                prediction = resize(base_mask, shape)
                metrics = segmentation_metrics(prediction, target, compute_boundary=True)
                results[f"{method}__{grid}"].append({
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "method": method,
                    "grid": grid,
                    "native_height": native_height,
                    "native_width": native_width,
                    "native_gt_area_ratio": native_ratio,
                    "native_size_group": native_group,
                    **metrics,
                })

    if opened != 184:
        raise ValueError(f"opened {opened} annotations instead of 184")
    summaries: dict[str, object] = {}
    for key, rows in results.items():
        summary = summarize_segmentation_rows(rows)
        tumor_rows = [row for row in rows if bool(row["gt_positive"])]
        summary["native_subgroups"] = {
            label: summarize_segmentation_rows(
                [row for row in tumor_rows if row["native_size_group"] == label]
            )
            for label in ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")
        }
        summary["group_bootstrap_ci95"] = bootstrap_group_confidence_intervals(
            rows, iterations=args.bootstrap_iterations, seed=20260806
        )
        summaries[key] = summary

    all_rows = [row for key in sorted(results) for row in results[key]]
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(all_rows))
    report = {
        "schema_version": 1,
        "study": "G4 E0 coordinate and evaluator reconciliation",
        "split_sha256": args.expected_split_sha256,
        "images": 371,
        "tumor_images": 184,
        "validation_annotations_opened": opened,
        "candidate_oracle_mean_dice_common320_all_payload_candidates": float(np.mean(oracle_dice)),
        "size_group_migration_resized320_to_native": {
            f"{old}__to__{new}": count for (old, new), count in sorted(migration.items())
        },
        "summaries": summaries,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "pass": True,
        "prediction_bytes_verified_before_annotations": True,
        "wsss_freeze_sha256": args.expected_wsss_freeze_sha256,
        "fully_freeze_sha256": args.expected_fully_freeze_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(report_path),
        "validation_annotations_opened": opened,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    audit_path = args.output_dir / "evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "evaluation_audit_sha256": sha256_file(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
