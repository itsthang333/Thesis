from __future__ import annotations

"""Freeze control-subtracted co-witness selections before validation GT."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from models.rich_gallery_g2_objective import average_percentile_rank, stable_select


RAW_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
LOCKED_RANK_WEIGHTS = (0.05, 0.10, 0.20, 0.35, 0.50)
SOURCE_NAMES = {0: "classifier448", 1: "layercam320", 2: "external_saliency"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variants() -> list[str]:
    return [
        "baseline",
        *(f"raw_control_contrast_x{value:g}" for value in RAW_MULTIPLIERS),
        *(f"source_locked_rank_control_contrast_x{value:g}" for value in LOCKED_RANK_WEIGHTS),
    ]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("control-contrast Stage-A output must not exist")
    input_freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(input_freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("input prediction freeze SHA-256 mismatch")
    input_freeze = json.loads(input_freeze_path.read_text(encoding="utf-8"))
    if (
        input_freeze.get("validation_images") != 371
        or input_freeze.get("spatial_ground_truth_used") is not False
        or input_freeze.get("validation_gt_read") is not False
        or input_freeze.get("test_images_read") != 0
        or input_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("input prediction freeze contract mismatch")

    baseline_rows: dict[str, dict[str, str]] = {}
    selection_manifest = args.prediction_root / "stage_a_selection_manifest.csv"
    if sha256_file(selection_manifest) != input_freeze["selection_manifest_sha256"]:
        raise ValueError("input selection manifest changed")
    with selection_manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "baseline":
                baseline_rows[Path(row["image_id"]).stem] = row
    if len(baseline_rows) != 371:
        raise ValueError("expected exact 371-image baseline selection cohort")

    frozen_rows: list[dict[str, object]] = []
    score_hashes: list[str] = []
    for score_path in sorted((args.prediction_root / "stage_a_scores").glob("*.npz")):
        image_id = score_path.stem
        if image_id not in baseline_rows:
            raise ValueError(f"score payload not in baseline cohort: {image_id}")
        score_sha = sha256_file(score_path)
        score_hashes.append(score_sha)
        with np.load(score_path, allow_pickle=False) as payload:
            source_ids = np.asarray(payload["source_ids"], dtype=np.int64).reshape(-1)
            candidate_indices = np.asarray(payload["candidate_indices"], dtype=np.int64).reshape(-1)
            g1 = np.asarray(payload["g1_logits"], dtype=np.float64).reshape(-1)
            baseline_fusion = np.asarray(payload["baseline_fusion"], dtype=np.float64).reshape(-1)
            baseline_scores = np.asarray(payload["baseline_scores"], dtype=np.float64).reshape(-1)
            control = np.asarray(payload["control_residual"], dtype=np.float64).reshape(-1)
            full = np.asarray(payload["full_residual"], dtype=np.float64).reshape(-1)
        lengths = {len(array) for array in (source_ids, candidate_indices, g1, baseline_fusion, baseline_scores, control, full)}
        if len(lengths) != 1 or lengths == {0}:
            raise ValueError(f"score payload arrays misaligned: {image_id}")
        contrast = full - control
        baseline_local = stable_select(baseline_fusion, g1)
        if int(baseline_rows[image_id]["selected_local_index"]) != baseline_local:
            raise ValueError(f"baseline selection did not reproduce: {image_id}")

        choices: dict[str, int] = {"baseline": baseline_local}
        for multiplier in RAW_MULTIPLIERS:
            choices[f"raw_control_contrast_x{multiplier:g}"] = stable_select(
                baseline_scores + float(multiplier) * contrast,
                g1,
            )
        baseline_source = int(source_ids[baseline_local])
        source_members = np.flatnonzero(source_ids == baseline_source)
        source_rank = average_percentile_rank(contrast[source_members])
        for weight in LOCKED_RANK_WEIGHTS:
            local_scores = baseline_fusion[source_members] + float(weight) * (source_rank - 0.5)
            local_member = stable_select(local_scores, g1[source_members])
            choices[f"source_locked_rank_control_contrast_x{weight:g}"] = int(source_members[local_member])

        baseline = baseline_rows[image_id]
        for variant in variants():
            selected = int(choices[variant])
            frozen_rows.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": baseline["group_id"],
                    "tumor": int(baseline["tumor"]),
                    "candidate_count": len(source_ids),
                    "selected_local_index": selected,
                    "selected_candidate_index": int(candidate_indices[selected]),
                    "selected_source": SOURCE_NAMES[int(source_ids[selected])],
                    "input_score_path": str(score_path.relative_to(args.prediction_root)).replace("\\", "/"),
                    "input_score_sha256": score_sha,
                }
            )

    if len(score_hashes) != 371 or len(frozen_rows) != 371 * len(variants()):
        raise ValueError("control-contrast frozen cohort incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frozen_rows[0]))
        writer.writeheader()
        writer.writerows(frozen_rows)
    score_set_sha = hashlib.sha256("\n".join(sorted(score_hashes)).encode("utf-8")).hexdigest()
    freeze = {
        "stage": "rich_gallery_cross_view_control_contrast_stage_a_v1",
        "input_prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "input_score_set_sha256": score_set_sha,
        "selection_manifest_sha256": sha256_file(manifest_path),
        "validation_images": 371,
        "selection_rows": len(frozen_rows),
        "variants": variants(),
        "raw_multipliers": list(RAW_MULTIPLIERS),
        "source_locked_rank_weights": list(LOCKED_RANK_WEIGHTS),
        "formula": {
            "raw": "centered_baseline + alpha * (full_residual - control_residual)",
            "source_locked_rank": "select only baseline source by baseline_fusion + gamma * (within_source_rank(delta)-0.5)",
        },
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
