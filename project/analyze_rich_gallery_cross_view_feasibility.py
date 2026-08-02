from __future__ import annotations

"""GT-only feasibility diagnosis for cross-view candidate supervision.

The analysis consumes only the canonical split, audited G1 descriptor freeze,
and audited Stage-B candidate table.  It never creates a selector.  Validation
Dice is used only to ask whether already-frozen cross-view descriptor support
contains candidate identity beyond G1/upstream/area/source.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from analyze_rich_gallery_g1_conditional_information import (
    partial_rank_correlation,
)
from mae_reconstruction_io import sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank


CONTRAST_SLICE = slice(768, 1152)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--expected-g1-freeze-sha256", required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--expected-stage-b-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cosine_max_support(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference descriptors must be compatible matrices")
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1.0e-8)
    reference = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), 1.0e-8
    )
    return (query @ reference.T).max(axis=1)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(np.asarray(left, dtype=np.float64))
    right_rank = average_percentile_rank(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 1.0e-12 or np.std(right_rank) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _source_controls(sources: list[str]) -> np.ndarray:
    levels = sorted(set(sources))
    if len(levels) <= 1:
        return np.empty((len(sources), 0), dtype=np.float64)
    return np.column_stack(
        [np.asarray([float(value == level) for value in sources]) for level in levels[1:]]
    )


def _load_descriptor(
    root: Path,
    row: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    path = root / "descriptor_evidence" / row["evidence_path"]
    if sha256_file(path) != row["evidence_sha256"]:
        raise ValueError(f"descriptor evidence hash mismatch: {row['image_id']}")
    with np.load(path, allow_pickle=False) as payload:
        descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
        indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
    if descriptors.ndim != 2 or descriptors.shape[1] != 1156:
        raise ValueError("unexpected frozen G1 descriptor layout")
    if len(indices) != len(descriptors) or len(set(indices.tolist())) != len(indices):
        raise ValueError("candidate-index/descriptor contract mismatch")
    return indices, descriptors[:, CONTRAST_SLICE]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("cross-view analysis output must not exist")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")

    freeze_path = args.g1_root / "diagnostic_freeze.json"
    if sha256_file(freeze_path) != args.expected_g1_freeze_sha256:
        raise ValueError("G1 diagnostic-freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("validation_images") != 371
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
        or freeze.get("test_images_read") != 0
    ):
        raise ValueError("G1 descriptor scientific boundary mismatch")
    descriptor_manifest_path = args.g1_root / "descriptor_evidence_manifest.csv"
    if sha256_file(descriptor_manifest_path) != freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("G1 descriptor manifest SHA-256 mismatch")
    descriptor_rows = _read_csv(descriptor_manifest_path)
    descriptor_by_image = {row["image_id"]: row for row in descriptor_rows}
    if len(descriptor_by_image) != 371:
        raise ValueError("G1 descriptor manifest population mismatch")

    summary_path = args.stage_b_root / "evaluation_summary.json"
    if sha256_file(summary_path) != args.expected_stage_b_summary_sha256:
        raise ValueError("Stage-B summary SHA-256 mismatch")
    stage_b = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        stage_b.get("tumor_images_evaluated") != 184
        or stage_b.get("test_evaluated") is not False
        or stage_b.get("candidate_scores_frozen_before_validation_gt") is not True
    ):
        raise ValueError("Stage-B scientific boundary mismatch")
    candidate_path = args.stage_b_root / "per_candidate_layerwise.csv"
    if sha256_file(candidate_path) != stage_b["per_candidate_layerwise_sha256"]:
        raise ValueError("Stage-B candidate table SHA-256 mismatch")

    split_rows = _read_csv(args.split_manifest)
    by_group: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in split_rows:
        if row["split"] == "val" and row["tumor"] == "1":
            by_group[row["group_id"]].append(row)
    oriented_pairs: list[dict[str, dict[str, str] | str]] = []
    for group_id, rows in sorted(by_group.items()):
        if len(rows) != 2 or len({row["view"] for row in rows}) != 2:
            continue
        oriented_pairs.extend(
            [
                {"group_id": group_id, "query": rows[0], "partner": rows[1]},
                {"group_id": group_id, "query": rows[1], "partner": rows[0]},
            ]
        )
    if len(oriented_pairs) != 26:
        raise ValueError("expected 13 distinct-view validation tumor pairs")

    candidate_by_image: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    required_candidate_columns = {
        "image_id",
        "candidate_index",
        "candidate_dice",
        "is_eligible_oracle",
        "is_baseline_selected",
        "source",
        "candidate_area_ratio",
        "g1_logit",
        "upstream_score",
    }
    with candidate_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_candidate_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError("Stage-B candidate columns missing")
        paired_images = {
            str(pair["query"]["image_id"]) for pair in oriented_pairs
        }
        for row in reader:
            if row["image_id"] in paired_images:
                candidate_by_image[row["image_id"]].append(row)

    descriptor_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def descriptor(image_id: str) -> tuple[np.ndarray, np.ndarray]:
        if image_id not in descriptor_cache:
            descriptor_cache[image_id] = _load_descriptor(
                args.g1_root, descriptor_by_image[image_id]
            )
        return descriptor_cache[image_id]

    raw_records: list[dict[str, object]] = []
    controlled_records: list[dict[str, object]] = []
    for pair in oriented_pairs:
        query = pair["query"]
        partner = pair["partner"]
        image_id = query["image_id"]
        query_indices, query_descriptor = descriptor(image_id)
        _, partner_descriptor = descriptor(partner["image_id"])
        same_support = cosine_max_support(query_descriptor, partner_descriptor)
        index_lookup = {
            int(candidate_index): index
            for index, candidate_index in enumerate(query_indices)
        }
        candidate_rows = candidate_by_image[image_id]
        order = [index_lookup[int(row["candidate_index"])] for row in candidate_rows]
        same_support = same_support[order]
        quality = np.asarray(
            [float(row["candidate_dice"]) for row in candidate_rows]
        )
        oracle_index = next(
            index
            for index, row in enumerate(candidate_rows)
            if row["is_eligible_oracle"].lower() in {"1", "true"}
        )
        baseline_index = next(
            index
            for index, row in enumerate(candidate_rows)
            if row["is_baseline_selected"].lower() in {"1", "true"}
        )
        raw_records.append(
            {
                "image_id": image_id,
                "group_id": pair["group_id"],
                "quality_rank_correlation": _rank_correlation(same_support, quality),
                "oracle_percentile": float(
                    average_percentile_rank(same_support)[oracle_index]
                ),
                "baseline_percentile": float(
                    average_percentile_rank(same_support)[baseline_index]
                ),
                "oracle_above_baseline": float(
                    same_support[oracle_index] > same_support[baseline_index]
                ),
            }
        )

        alternatives = [
            other
            for other in oriented_pairs
            if other["group_id"] != pair["group_id"]
            and other["query"]["anatomy"] == query["anatomy"]
            and other["query"]["tumor_type"] == query["tumor_type"]
            and other["partner"]["view"] == partner["view"]
        ]
        if not alternatives:
            continue
        control_support = []
        for alternative in alternatives:
            _, alternative_descriptor = descriptor(
                alternative["partner"]["image_id"]
            )
            control_support.append(
                cosine_max_support(query_descriptor, alternative_descriptor)
            )
        mean_control = np.mean(control_support, axis=0)[order]
        matched_residual = same_support - mean_control
        area = average_percentile_rank(
            np.log1p(
                np.asarray(
                    [float(row["candidate_area_ratio"]) for row in candidate_rows]
                )
            )
        )
        controls = np.column_stack(
            [
                average_percentile_rank(
                    np.asarray([float(row["g1_logit"]) for row in candidate_rows])
                ),
                average_percentile_rank(
                    np.asarray(
                        [float(row["upstream_score"]) for row in candidate_rows]
                    )
                ),
                area,
                _source_controls([row["source"] for row in candidate_rows]),
            ]
        )
        controlled_records.append(
            {
                "image_id": image_id,
                "group_id": pair["group_id"],
                "control_partner_count": len(alternatives),
                "matched_quality_rank_correlation": _rank_correlation(
                    same_support, quality
                ),
                "control_quality_rank_correlation": _rank_correlation(
                    mean_control, quality
                ),
                "matched_minus_control_quality_rank_correlation": _rank_correlation(
                    matched_residual, quality
                ),
                "matched_minus_control_partial_rank_correlation": (
                    partial_rank_correlation(quality, matched_residual, controls)
                ),
                "oracle_above_baseline": float(
                    matched_residual[oracle_index] > matched_residual[baseline_index]
                ),
            }
        )

    train_tumor_by_group: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in split_rows:
        if row["split"] == "train" and row["tumor"] == "1":
            train_tumor_by_group[row["group_id"]].append(row)
    train_distinct_view_groups = [
        rows
        for rows in train_tumor_by_group.values()
        if len(rows) >= 2 and len({row["view"] for row in rows}) >= 2
    ]
    raw_summary = {
        key: _summary([float(row[key]) for row in raw_records])
        for key in (
            "quality_rank_correlation",
            "oracle_percentile",
            "baseline_percentile",
            "oracle_above_baseline",
        )
    }
    controlled_summary = {
        key: _summary([float(row[key]) for row in controlled_records])
        for key in (
            "matched_quality_rank_correlation",
            "control_quality_rank_correlation",
            "matched_minus_control_quality_rank_correlation",
            "matched_minus_control_partial_rank_correlation",
            "oracle_above_baseline",
        )
    }
    result = {
        "stage": "rich_gallery_cross_view_feasibility_v1",
        "split_sha256": args.expected_split_sha256,
        "g1_freeze_sha256": args.expected_g1_freeze_sha256,
        "stage_b_summary_sha256": args.expected_stage_b_summary_sha256,
        "descriptor_family": "frozen_rad_dino_candidate_contrast_384d",
        "training_signal_coverage": {
            "canonical_train_tumor_images": 1488,
            "distinct_view_tumor_groups": len(train_distinct_view_groups),
            "images_in_distinct_view_tumor_groups": int(
                sum(len(rows) for rows in train_distinct_view_groups)
            ),
            "group_id_limitation": (
                "heuristic_consecutive_id_and_stable_metadata_not_published_patient_id"
            ),
        },
        "validation_distinct_view_groups": 13,
        "validation_oriented_queries": len(raw_records),
        "matched_control_queries": len(controlled_records),
        "raw_same_group_support": raw_summary,
        "matched_minus_different_group_control": controlled_summary,
        "frozen_descriptor_decision": (
            "do_not_add_cross_view_similarity_to_g1_fusion"
        ),
        "representation_learning_hypothesis_status": (
            "cross_view_is_available_as_new_training_supervision_but_unproven"
        ),
        "validation_gt_used_only_for_retrospective_information_diagnosis": True,
        "selector_or_prediction_created": False,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "cross_view_feasibility.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = args.output_dir / "per_query_cross_view.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in raw_records + controlled_records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in raw_records:
            writer.writerow(row)
        for row in controlled_records:
            writer.writerow(row)
    print(
        json.dumps(
            {
                "pass": True,
                "cross_view_feasibility_sha256": sha256_file(json_path),
                "per_query_sha256": sha256_file(csv_path),
                "frozen_descriptor_decision": result["frozen_descriptor_decision"],
                "test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
