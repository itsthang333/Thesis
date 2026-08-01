from __future__ import annotations

"""Freeze cross-source-consensus choices before validation annotations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_cross_source_consensus import (
    VARIANTS,
    consensus_selector_scores,
    cross_source_max_iou,
    freeze_consensus_choices,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--expected-g2-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(rows) != 371:
        raise RuntimeError("consensus freeze requires canonical validation")
    g2_freeze_path = args.g2_root / "prediction_freeze.json"
    if sha256_file(g2_freeze_path) != args.expected_g2_freeze_sha256:
        raise ValueError("G2 prediction freeze SHA-256 mismatch")
    g2_freeze = json.loads(g2_freeze_path.read_text(encoding="utf-8"))
    if (
        g2_freeze.get("validation_images") != 371
        or g2_freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or g2_freeze.get("validation_gt_read") is not False
        or g2_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G2 freeze safety contract mismatch")
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("consensus diagnostic requires complete candidates")
    selection_path = args.g2_root / "stage_a_selection_manifest.csv"
    if sha256_file(selection_path) != g2_freeze["selection_manifest_sha256"]:
        raise ValueError("G2 selection manifest changed")
    with selection_path.open("r", newline="", encoding="utf-8-sig") as handle:
        g2_rows = list(csv.DictReader(handle))
    baseline = {
        row["image_id"]: int(row["selected_candidate_index"])
        for row in g2_rows
        if row["variant"] == "g1_frozen__rank_fusion"
    }
    if len(baseline) != 371:
        raise ValueError("G1 fusion baseline cohort mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, object]] = []
    consensus_hashes: list[str] = []
    for row in rows:
        image_id = row["image_id"]
        stem = Path(image_id).stem
        score_path = args.g2_root / "stage_a_scores" / f"{stem}.npz"
        candidate_row = candidate_rows[stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        with np.load(score_path, allow_pickle=False) as score:
            kept = score["candidate_indices"].astype(np.int64)
            sources = score["source_ids"].astype(np.int64)
            upstream = score["upstream_scores"].astype(np.float64)
            g1 = score["g1_frozen_candidate_logits"].astype(np.float64)
        with np.load(candidate_path, allow_pickle=False) as candidate:
            masks = candidate["sam_masks"][kept].astype(bool)
        if masks.shape[0] != len(kept) or sources.shape != kept.shape:
            raise ValueError(f"candidate alignment mismatch: {image_id}")
        consensus = cross_source_max_iou(masks, sources)
        scores = consensus_selector_scores(g1, upstream, consensus)
        choices = freeze_consensus_choices(g1, upstream, consensus)
        if int(kept[choices["g1_upstream_baseline"]]) != baseline[image_id]:
            raise RuntimeError(f"G1 fusion baseline did not reproduce: {image_id}")
        score_hash = hashlib.sha256(consensus.astype("<f4").tobytes()).hexdigest()
        consensus_hashes.append(score_hash)
        for variant in VARIANTS:
            local = choices[variant]
            output_rows.append(
                {
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "tumor": row["tumor"],
                    "variant": variant,
                    "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                    "candidate_count": len(kept),
                    "selected_local_index": local,
                    "selected_candidate_index": int(kept[local]),
                    "selected_source_id": int(sources[local]),
                    "selected_g1_logit": float(g1[local]),
                    "selected_upstream_score": float(upstream[local]),
                    "selected_consensus_iou": float(consensus[local]),
                    "selected_rule_score": float(scores[variant][local]),
                    "consensus_vector_sha256": score_hash,
                }
            )
    manifest = args.output_dir / "selection_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    freeze = {
        "stage": "rich_gallery_cross_source_consensus_freeze_v1",
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "g2_prediction_freeze_sha256": args.expected_g2_freeze_sha256,
        "val_candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.expected_val_pseudo_manifest_sha256,
        "selection_manifest_sha256": sha256_file(manifest),
        "consensus_set_sha256": hashlib.sha256(
            "\n".join(sorted(consensus_hashes)).encode()
        ).hexdigest(),
        "validation_images": 371,
        "variants": list(VARIANTS),
        "selection_rows": len(output_rows),
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
