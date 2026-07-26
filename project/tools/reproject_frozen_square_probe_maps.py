from __future__ import annotations

"""Correct square-padded RAD-DINO maps back to the original image frame.

The dense-MIL and INSIGHT runners generated predictions in the square-padded
448-pixel frame but evaluated their 320-pixel square maps directly against
ground truth resized from the original rectangular image.  This tool applies
the unique inverse geometry to already frozen maps, freezes the derived maps,
and only then opens validation annotations through the existing evaluator.

No threshold, model weight, or prediction value is selected or fitted here.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rad_dino_dense_mil_probe as base
from mae_reconstruction_io import load_split_rows_without_annotations
from models.mae_reconstruction import pad_to_square, project_square_map


ARMS = ("single_scale", "multiscale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_original_run(
    run_root: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_split_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    run_manifest_path = run_root / "run_manifest.json"
    freeze_path = run_root / "prediction_freeze.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        run_manifest["checkpoint_sha256"] != expected_checkpoint_sha256
        or freeze["checkpoint_sha256"] != expected_checkpoint_sha256
        or run_manifest["split_sha256"] != expected_split_sha256
        or freeze["split_sha256"] != expected_split_sha256
        or freeze["validation_gt_read"]
        or freeze["test_evaluated"]
        or run_manifest["test_evaluated"]
    ):
        raise RuntimeError("Original frozen-run contract mismatch")
    for arm in ARMS:
        manifest = run_root / "predictions" / arm / "prediction_manifest.csv"
        if sha256(manifest) != freeze["prediction_manifests"][arm]:
            raise RuntimeError(f"Original prediction manifest mismatch: {arm}")
        for row in read_csv(manifest):
            source = run_root / "predictions" / arm / row["map_path"]
            if sha256(source) != row["map_sha256"]:
                raise RuntimeError(f"Original frozen map mismatch: {arm}/{row['image_id']}")
    return run_manifest, freeze


def reproject_arm(
    *,
    run_root: Path,
    output_root: Path,
    arm: str,
    dataset_root: Path,
    validation_rows: list[dict[str, str]],
    protocol_sha256: str,
) -> str:
    source_dir = run_root / "predictions" / arm
    source_manifest = read_csv(source_dir / "prediction_manifest.csv")
    source_by_id = {row["image_id"]: row for row in source_manifest}
    if len(source_by_id) != 371 or len(source_manifest) != 371:
        raise RuntimeError(f"Expected 371 unique frozen predictions for {arm}")

    destination = output_root / "predictions" / arm
    maps_dir = destination / "maps"
    maps_dir.mkdir(parents=True, exist_ok=False)
    corrected_rows: list[dict[str, object]] = []
    for row in validation_rows:
        original = source_by_id[row["image_id"]]
        source_path = source_dir / original["map_path"]
        values = np.load(source_path, allow_pickle=False).astype(np.float32)
        if values.shape != (320, 320) or not np.isfinite(values).all():
            raise RuntimeError(f"Unexpected frozen map geometry: {row['image_id']}")

        image_path = base.locate_verified_image(dataset_root, row)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            _square, projection = pad_to_square(image, fill=0)
        corrected = project_square_map(
            values,
            projection,
            output_height=320,
            output_width=320,
        )
        corrected = np.clip(corrected, 0.0, 1.0).astype(np.float16)
        relative = Path("maps") / f"{Path(row['image_id']).stem}.npy"
        map_path = destination / relative
        np.save(map_path, corrected, allow_pickle=False)
        corrected_rows.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "map_path": relative.as_posix(),
                "map_sha256": sha256(map_path),
                "source_map_sha256": original["map_sha256"],
                "original_width": width,
                "original_height": height,
                "content_fraction": float(min(width, height) / max(width, height)),
                "raw_p99": float(np.percentile(corrected.astype(np.float32), 99)),
                "raw_max": float(corrected.max()),
            }
        )
    manifest_path = destination / "prediction_manifest.csv"
    write_csv(manifest_path, corrected_rows)
    manifest_sha = sha256(manifest_path)
    (destination / "generation_metadata.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "operation": (
                    "crop the content box from the frozen square-padded map, "
                    "then bilinear-resize to 320x320"
                ),
                "source_prediction_manifest_sha256": sha256(
                    source_dir / "prediction_manifest.csv"
                ),
                "prediction_manifest_sha256": manifest_sha,
                "protocol_sha256": protocol_sha256,
                "cohort": 371,
                "parameters_fitted": False,
                "threshold_selected": False,
                "validation_gt_read": False,
                "test_evaluated": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_sha


def load_evaluation(path: Path) -> list[dict[str, object]]:
    return [dict(row) for row in read_csv(path)]


def summarize_aspect_ratios(
    aspect_ratios: list[float],
) -> dict[str, int | float]:
    if not aspect_ratios or any(
        not np.isfinite(value) or value <= 0.0 or value > 1.0
        for value in aspect_ratios
    ):
        raise ValueError("Aspect ratios must be finite values in (0,1]")
    return {
        "square": int(sum(np.isclose(value, 1.0) for value in aspect_ratios)),
        "below_0_90": int(sum(value < 0.90 for value in aspect_ratios)),
        "below_0_75": int(sum(value < 0.75 for value in aspect_ratios)),
        "minimum": float(min(aspect_ratios)),
        "mean": float(np.mean(aspect_ratios)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    run_manifest, original_freeze = validate_original_run(
        args.run_root,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_split_sha256=args.expected_split_sha256,
    )
    validation_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(validation_rows) != 371:
        raise RuntimeError("Frozen validation cohort mismatch")
    args.output_dir.mkdir(parents=True)
    manifest_hashes = {
        arm: reproject_arm(
            run_root=args.run_root,
            output_root=args.output_dir,
            arm=arm,
            dataset_root=args.dataset_root,
            validation_rows=validation_rows,
            protocol_sha256=args.protocol_sha256,
        )
        for arm in ARMS
    }
    freeze = {
        "probe_id": args.probe_id,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "source_run_manifest_sha256": sha256(args.run_root / "run_manifest.json"),
        "source_prediction_freeze_sha256": sha256(
            args.run_root / "prediction_freeze.json"
        ),
        "source_checkpoint_sha256": args.expected_checkpoint_sha256,
        "source_prediction_manifests": original_freeze["prediction_manifests"],
        "derived_prediction_manifests": manifest_hashes,
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    corrected: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    effects: dict[str, object] = {}
    for arm in ARMS:
        evaluated, summary = base.evaluate_arm(
            args.output_dir / "predictions" / arm,
            args.dataset_root,
            args.split_manifest,
        )
        corrected[arm] = evaluated
        summaries[arm] = summary
        original = load_evaluation(
            args.run_root
            / "predictions"
            / arm
            / "evaluation"
            / "per_image.csv"
        )
        effects[arm] = base.bootstrap_compare(original, evaluated)
        (args.output_dir / f"{arm}_geometry_effect.json").write_text(
            json.dumps(effects[arm], indent=2) + "\n",
            encoding="utf-8",
        )
    corrected_comparison = base.bootstrap_compare(
        corrected["single_scale"], corrected["multiscale"]
    )
    comparison_path = args.output_dir / "paired_comparison.json"
    comparison_path.write_text(
        json.dumps(corrected_comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    aspect_ratios = [
        min(
            int(row["original_width"]),
            int(row["original_height"]),
        )
        / max(
            int(row["original_width"]),
            int(row["original_height"]),
        )
        for row in read_csv(
            args.output_dir
            / "predictions"
            / "single_scale"
            / "prediction_manifest.csv"
        )
    ]
    correction_manifest = {
        "probe_id": args.probe_id,
        "operation": "deterministic inverse square-padding geometry only",
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "source_run_id": run_manifest["run_id"],
        "source_checkpoint_sha256": args.expected_checkpoint_sha256,
        "source_run_manifest_sha256": freeze["source_run_manifest_sha256"],
        "prediction_freeze_sha256": sha256(freeze_path),
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "aspect_ratio": summarize_aspect_ratios(aspect_ratios),
        "summaries": summaries,
        "geometry_effect": effects,
        "corrected_multiscale_minus_single": corrected_comparison,
        "parameters_fitted": False,
        "threshold_selected": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(correction_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(correction_manifest, indent=2))


if __name__ == "__main__":
    main()
