from __future__ import annotations

"""Fast independent audit for matched-normal transplant Stage A."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.matched_normal_candidate_transplant import (
    DENSENET_DIAGNOSTIC_STAGES,
    frozen_selector_panel,
    select_normal_reference_pairs,
    select_random_normal_reference_pairs,
)
from models.rich_gallery_g2_objective import average_percentile_rank, stable_select


VARIANTS = (
    "g1_upstream_baseline",
    "transplant_only",
    "baseline_transplant_equal",
    "baseline_transplant_three_to_one",
    "baseline_random_control_three_to_one",
)
STAGE_METRICS = (
    "feature_l2_inside",
    "feature_l2_ring",
    "feature_l2_contrast",
    "relative_feature_l2_inside",
    "relative_feature_l2_ring",
    "relative_feature_l2_contrast",
    "cosine_inside",
    "cosine_ring",
    "delta_energy_inside_fraction",
    "mask_mass",
    "ring_mass",
)
CLASS_METRICS = (
    "class_response_delta_inside",
    "class_response_delta_ring",
    "class_response_delta_contrast",
    "class_response_delta_global",
    "class_response_logit_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _require_array(
    payload: np.lib.npyio.NpzFile,
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"missing Stage-A array: {key}")
    value = np.asarray(payload[key])
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid Stage-A array {key}: {value.shape}")
    return value


def main() -> None:
    args = parse_args()
    freeze_path = args.output_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("prediction-freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_matched_normal_transplant_stage_a_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("classifier_checkpoint_sha256") != args.expected_classifier_sha256
        or freeze.get("validation_images") != 371
        or freeze.get("tumor_validation_images") != 184
        or freeze.get("train_normal_references") != 1493
        or freeze.get("reference_rows") != 1484
        or freeze.get("selection_rows") != 371 * len(VARIANTS)
        or tuple(freeze.get("variants", [])) != VARIANTS
        or tuple(freeze.get("layerwise_stages", [])) != DENSENET_DIAGNOSTIC_STAGES
        or freeze.get("baseline_reproduction_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("prediction-freeze contract mismatch")
    train_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="train",
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    normal_rows = [row for row in train_rows if row["tumor"] == "0"]
    if len(normal_rows) != 1493 or len(val_rows) != 371:
        raise ValueError("canonical cohort mismatch")

    reference_path = args.output_root / "reference_manifest.csv"
    selection_path = args.output_root / "selection_manifest.csv"
    if sha256_file(reference_path) != freeze["reference_manifest_sha256"]:
        raise ValueError("reference manifest changed")
    if sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("selection manifest changed")
    references = _read_csv(reference_path)
    selections = _read_csv(selection_path)
    if len(references) != 1484 or len(selections) != 371 * len(VARIANTS):
        raise ValueError("Stage-A manifest row counts mismatch")
    by_reference = {
        (row["image_id"], row["arm"], int(row["pair_index"])): row
        for row in references
    }
    if len(by_reference) != len(references):
        raise ValueError("duplicate reference identity")
    for query in val_rows:
        expected = {
            "matched": select_normal_reference_pairs(query, normal_rows, pair_count=2),
            "random": select_random_normal_reference_pairs(
                query, normal_rows, pair_count=2, seed=20260802
            ),
        }
        for arm, pairs in expected.items():
            for pair_index, pair in enumerate(pairs):
                row = by_reference[(query["image_id"], arm, pair_index)]
                if (
                    row["recipient_image_id"] != pair.recipient_image_id
                    or row["recipient_group_id"] != pair.recipient_group_id
                    or row["sham_image_id"] != pair.sham_image_id
                    or row["sham_group_id"] != pair.sham_group_id
                ):
                    raise ValueError("reference assignment does not reproduce")

    indexed: dict[tuple[str, str], dict[str, str]] = {}
    score_cache: dict[str, dict[str, np.ndarray]] = {}
    candidate_hashes = {
        row["image_id"]: row["candidate_payload_sha256"] for row in selections
    }
    max_decomposition_residual = 0.0
    for row in selections:
        key = (row["variant"], row["image_id"])
        if row["variant"] not in VARIANTS or key in indexed:
            raise ValueError("selection identity mismatch")
        candidate_path = args.candidate_root / "candidate_diagnostics" / (
            Path(row["image_id"]).stem + ".npz"
        )
        if not candidate_path.is_file() or sha256_file(candidate_path) != row["candidate_payload_sha256"]:
            raise ValueError(f"candidate payload changed: {row['image_id']}")
        score_path = args.output_root / row["score_path"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"score payload changed: {row['image_id']}")
        if row["score_path"] not in score_cache:
            with np.load(score_path, allow_pickle=False) as payload:
                score_cache[row["score_path"]] = {
                    name: np.asarray(payload[name]) for name in payload.files
                }
        payload = score_cache[row["score_path"]]
        count = int(row["candidate_count"])
        if payload["candidate_indices"].shape != (count,):
            raise ValueError("candidate index shape mismatch")
        if tuple(payload["matched_stage_names"].astype(str)) != DENSENET_DIAGNOSTIC_STAGES:
            raise ValueError("matched stage order mismatch")
        if tuple(payload["random_stage_names"].astype(str)) != DENSENET_DIAGNOSTIC_STAGES:
            raise ValueError("random stage order mismatch")
        for arm in ("matched", "random"):
            _require_array(payload, f"{arm}_score", (count,))
            _require_array(payload, f"{arm}_recipient_std", (count,))
            for metric in STAGE_METRICS:
                _require_array(payload, f"{arm}_{metric}_mean", (count, 5))
                _require_array(payload, f"{arm}_{metric}_recipient_std", (count, 5))
            for metric in CLASS_METRICS:
                value = _require_array(payload, f"{arm}_{metric}_mean", (count,))
                spread = _require_array(
                    payload, f"{arm}_{metric}_recipient_std", (count,)
                )
                if metric == "class_response_logit_residual":
                    max_decomposition_residual = max(
                        max_decomposition_residual,
                        float(np.max(np.abs(value))),
                        float(np.max(np.abs(spread))),
                    )
        g1 = payload["g1_logits"].astype(np.float64)
        upstream = payload["upstream_scores"].astype(np.float64)
        baseline = 0.5 * (
            average_percentile_rank(g1) + average_percentile_rank(upstream)
        )
        panel = frozen_selector_panel(
            baseline,
            payload["matched_score"],
            payload["random_score"],
        )
        local = stable_select(panel[row["variant"]], g1)
        if (
            local != int(row["selected_local_index"])
            or int(payload["candidate_indices"][local])
            != int(row["selected_candidate_index"])
        ):
            raise ValueError(f"frozen selection does not reproduce: {key}")
        indexed[key] = row
    if len(indexed) != 371 * len(VARIANTS) or len(score_cache) != 371:
        raise ValueError("Stage-A score cohort incomplete")
    if max_decomposition_residual > 1.0e-4:
        raise ValueError(
            f"class-map/logit decomposition residual too large: {max_decomposition_residual}"
        )
    audit = {
        "pass": True,
        "stage": "rich_gallery_matched_normal_transplant_stage_a_audit_v1",
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "split_sha256": args.expected_split_sha256,
        "classifier_checkpoint_sha256": args.expected_classifier_sha256,
        "validation_images_verified": 371,
        "score_payloads_verified": 371,
        "selection_rows_verified": 371 * len(VARIANTS),
        "reference_rows_verified": 1484,
        "layerwise_stages_verified": list(DENSENET_DIAGNOSTIC_STAGES),
        "max_class_response_logit_residual": max_decomposition_residual,
        "candidate_payloads_verified": len(candidate_hashes),
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**audit, "audit_sha256": sha256_file(args.audit_output)}, indent=2))


if __name__ == "__main__":
    main()
