from __future__ import annotations

"""Run the N1 normal-only direct-anomaly selector on the frozen gallery."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, save_float_map, sha256_file
from models.mask_bag_normal_anomaly import (
    DirectNormalAnomalyConfig,
    fit_direct_normal_anomaly_bank,
    score_direct_normal_anomaly,
)
from models.mask_bag_score_evidence import save_candidate_score_evidence, write_candidate_score_manifest
from models.mask_bag_selector_cache import unpack_candidate_masks
from run_mask_bag_normal_prototype_arm import _load_cache_records, _verify_cache_freeze


EXPECTED_BASELINE_COUNT_PROBABILITY_SPEARMAN = 0.48137777593654113


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prototype-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _absolute_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="stable")
        result = np.empty(len(array), dtype=np.float64)
        start = 0
        while start < len(array):
            stop = start + 1
            while stop < len(array) and array[order[stop]] == array[order[start]]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + 1 + stop)
            start = stop
        return result

    if len(first) != len(second) or len(first) < 2:
        raise ValueError("N1 Spearman inputs must be aligned and nontrivial")
    left, right = ranks(first), ranks(second)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("N1 Spearman inputs must be nonconstant")
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        raise ValueError("N1 Spearman is not finite")
    return abs(value)


def _verify_baseline(
    args: argparse.Namespace,
    validation_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    freeze_path = args.baseline_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_baseline_freeze_sha256:
        raise ValueError("N1 baseline freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("checkpoint_sha256") != args.expected_baseline_checkpoint_sha256
        or freeze.get("source_commit") != args.expected_baseline_source_commit
        or freeze.get("protocol_sha256") != args.expected_baseline_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("validation_predictions") != 371
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("N1 baseline freeze provenance/safety mismatch")
    manifest = args.baseline_root / "predictions" / "prediction_manifest.csv"
    if sha256_file(manifest) != freeze.get("prediction_manifest_sha256"):
        raise ValueError("N1 baseline prediction manifest hash mismatch")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["image_id"]: row for row in validation_rows}
    by_id = {row["image_id"]: row for row in rows}
    if len(rows) != 371 or set(by_id) != set(expected):
        raise ValueError("N1 baseline validation cohort mismatch")
    for image_id, row in by_id.items():
        source = expected[image_id]
        map_path = args.baseline_root / "predictions" / row["map_path"]
        if (
            row["group_id"] != source["group_id"]
            or row["tumor"] != source["tumor"]
            or not map_path.is_file()
            or sha256_file(map_path) != row["map_sha256"]
            or not math.isfinite(float(row["bag_logit"]))
            or not 0.0 <= float(row["bag_probability"]) <= 1.0
        ):
            raise ValueError(f"N1 baseline prediction mismatch: {image_id}")
    return by_id


def _write_outputs(
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, str]],
    prototypes: np.ndarray,
) -> tuple[str, str, str, float]:
    prediction_root = args.output_dir / "predictions"
    map_root = prediction_root / "maps"
    score_root = args.output_dir / "candidate_scores"
    score_payload_root = score_root / "scores"
    evidence_root = args.output_dir / "normal_anomaly_evidence"
    map_root.mkdir(parents=True, exist_ok=False)
    score_payload_root.mkdir(parents=True, exist_ok=False)
    evidence_root.mkdir(parents=True, exist_ok=False)
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    counts: list[int] = []
    probabilities: list[float] = []
    for index, record in enumerate(records):
        image_id = str(record["image_id"])
        result = score_direct_normal_anomaly(
            record["descriptors"], record["flipped_descriptors"], prototypes
        )
        scores = np.asarray(result["candidate_scores"], dtype=np.float32)
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        if scores.shape != indices.shape:
            raise RuntimeError(f"N1 candidate score count mismatch: {image_id}")
        selected_position = int(result["selected_candidate_position"])
        selected_original_index = int(indices[selected_position])
        stem = f"{index:04d}_{Path(image_id).stem}"
        score_relative = Path("scores") / f"{stem}.npz"
        saved_score = save_candidate_score_evidence(
            score_root / score_relative,
            candidate_indices=indices,
            candidate_logits=scores,
        )
        evidence_path = evidence_root / f"{stem}.npz"
        np.savez_compressed(
            evidence_path,
            schema_version=np.asarray(1, dtype=np.int32),
            candidate_indices=indices,
            original_normal_distance=np.asarray(result["original_normal_distance"], dtype=np.float32),
            flipped_normal_distance=np.asarray(result["flipped_normal_distance"], dtype=np.float32),
            candidate_scores=scores,
        )
        base = baseline[image_id]
        probability = float(base["bag_probability"])
        masks = unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        map_path = map_root / f"{stem}.npy"
        save_float_map(map_path, masks[selected_position] * probability)
        score_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                **saved_score,
                "score_path": str(score_relative),
            }
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "candidate_count": len(indices),
                "selected_candidate_index": selected_original_index,
                "view_selected_agreement": int(result["view_selected_agreement"]),
                "evidence_path": evidence_path.name,
                "evidence_sha256": sha256_file(evidence_path),
            }
        )
        prediction_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                "candidate_count": len(indices),
                "selected_candidate_index": selected_original_index,
                "selected_candidate_logit": saved_score["selected_candidate_logit"],
                "candidate_logit_tta": "mean_original_aligned_horizontal_flip",
                "bag_logit": base["bag_logit"],
                "bag_probability": base["bag_probability"],
                "selected_area_ratio": float(masks[selected_position].mean()),
                "fallback_count": int(np.asarray(record["fallback_flags"]).sum()),
                "map_path": str(Path("maps") / map_path.name),
                "map_sha256": sha256_file(map_path),
            }
        )
        counts.append(len(indices))
        probabilities.append(probability)
    prediction_manifest = prediction_root / "prediction_manifest.csv"
    with prediction_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    score_manifest = write_candidate_score_manifest(score_root, score_rows)
    evidence_manifest = args.output_dir / "normal_anomaly_evidence_manifest.csv"
    with evidence_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence_rows)
    association = _absolute_spearman(counts, probabilities)
    if abs(association - EXPECTED_BASELINE_COUNT_PROBABILITY_SPEARMAN) > 1.0e-12:
        raise RuntimeError("N1 did not preserve exact baseline count/probability association")
    return (
        sha256_file(prediction_manifest),
        str(score_manifest["manifest_sha256"]),
        sha256_file(evidence_manifest),
        association,
    )


def main() -> None:
    args = parse_args()
    if args.prototype_count != 32 or args.seed != 42:
        raise ValueError("N1 frozen controls require K=32 and seed=42")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("N1 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"N1 requires Tesla T4 x2, got {device_names}")
    if args.output_dir.exists():
        raise FileExistsError(f"N1 output already exists: {args.output_dir}")
    started = datetime.now(timezone.utc)

    cache_freeze, cache_manifest = _verify_cache_freeze(args)
    split_rows = {
        split: load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("N1 frozen cohort mismatch")
    cache, _validated = _load_cache_records(args, split_rows, cache_manifest)
    baseline = _verify_baseline(args, split_rows["val"])
    args.output_dir.mkdir(parents=True)
    normal_records = [
        {
            "image_id": record["image_id"],
            "image_label": record["label"],
            "descriptors": record["descriptors"],
            "flipped_descriptors": record["flipped_descriptors"],
            "family_ids": record["family_ids"],
        }
        for record in cache["train"]
        if int(record["label"]) == 0
    ]
    if len(normal_records) != 1493:
        raise RuntimeError("N1 normal training cohort mismatch")
    config = DirectNormalAnomalyConfig(prototype_count=32, seed=42)
    prototypes, bank_audit = fit_direct_normal_anomaly_bank(normal_records, config=config)
    bank_path = args.output_dir / "normal_anomaly_bank.npz"
    np.savez_compressed(
        bank_path,
        schema_version=np.asarray(1, dtype=np.int32),
        prototypes=prototypes.astype(np.float32),
    )
    bank_audit_path = args.output_dir / "normal_anomaly_bank_audit.json"
    bank_audit_path.write_text(
        json.dumps(
            {
                **bank_audit,
                "config": asdict(config),
                "bank_sha256": sha256_file(bank_path),
                "training_labels": "image_level_normal_only",
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_sha, score_sha, evidence_sha, association = _write_outputs(
        args, cache["val"], baseline, prototypes
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "selector_cache_manifest_sha256": cache_freeze["selector_cache_manifest_sha256"],
        "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "normal_anomaly_bank_sha256": sha256_file(bank_path),
        "normal_anomaly_bank_audit_sha256": sha256_file(bank_audit_path),
        "normal_anomaly_evidence_manifest_sha256": evidence_sha,
        "candidate_score_manifest_sha256": score_sha,
        "prediction_manifest_sha256": prediction_sha,
        "absolute_candidate_count_probability_spearman": association,
        "ranking_semantics": "direct_normal_anomaly_distance_not_classification_logit",
        "classification_probabilities": "exact_accepted_geometry_v3",
        "validation_predictions": 371,
        "training_labels": "image_level_normal_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_manifest = {
        "run_id": "btxrd_mask_bag_normal_only_direct_anomaly_n1_v1",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device_names": device_names,
        "config": asdict(config),
        "normal_images": len(normal_records),
        "validation_predictions": 371,
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prediction_freeze": freeze, "run_manifest": run_manifest}, indent=2))


if __name__ == "__main__":
    main()
