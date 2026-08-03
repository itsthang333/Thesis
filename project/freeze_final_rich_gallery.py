from __future__ import annotations

"""Freeze final rich-gallery choices before opening validation polygons."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from final_selector import select_candidate
from evaluation.frozen_test_guard import verify_frozen_test_config
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPECTED_COUNTS = {
    "val": {"images": 371, "tumor": 184, "normal": 187},
    "test": {"images": 373, "tumor": 187, "normal": 186},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--g1-diagnostic-root", type=Path, required=True)
    parser.add_argument("--expected-g1-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
        allow_test=args.split == "test",
    )
    counts = {
        "images": len(split_rows),
        "tumor": sum(int(row["tumor"]) for row in split_rows),
        "normal": sum(1 - int(row["tumor"]) for row in split_rows),
    }
    if counts != EXPECTED_COUNTS[args.split]:
        raise ValueError(f"canonical {args.split} counts differ: {counts}")
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
    )

    freeze_path = args.g1_diagnostic_root / "diagnostic_freeze.json"
    if sha256_file(freeze_path) != args.expected_g1_freeze_sha256:
        raise ValueError("G1 diagnostic freeze SHA-256 mismatch")
    g1_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        g1_freeze.get("stage") != "rich_gallery_g1_all_candidate_score_freeze_v1"
        or g1_freeze.get("split_sha256") != args.expected_split_sha256
        or int(g1_freeze.get("images", g1_freeze.get("validation_images", -1))) != counts["images"]
        or g1_freeze.get("cohort_split", "val") != args.split
        or g1_freeze.get("validation_gt_read") is not False
        or g1_freeze.get("spatial_ground_truth_used") is not False
        or int(g1_freeze.get("test_images_read", -1)) != (counts["images"] if args.split == "test" else 0)
        or g1_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1 diagnostic freeze violates the annotation-free contract")

    evidence_manifest = args.g1_diagnostic_root / "descriptor_evidence_manifest.csv"
    if sha256_file(evidence_manifest) != g1_freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("G1 evidence manifest changed")
    evidence_rows = _read_csv(evidence_manifest)
    evidence_by_image = {row["image_id"]: row for row in evidence_rows}
    expected_ids = {row["image_id"] for row in split_rows}
    if len(evidence_by_image) != counts["images"] or set(evidence_by_image) != expected_ids:
        raise ValueError(f"G1 evidence cohort differs from canonical {args.split}")

    candidate_rows, candidate_summary = validate_candidate_diagnostics_manifest(
        args.candidate_root,
        expected_image_names=expected_ids,
        split=args.split,
        expected_pseudo_manifest_sha256=args.expected_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
    )

    selections: list[dict[str, object]] = []
    for split_row in split_rows:
        image_id = split_row["image_id"]
        evidence_row = evidence_by_image[image_id]
        evidence_path = args.g1_diagnostic_root / "descriptor_evidence" / evidence_row["evidence_path"]
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise ValueError(f"G1 evidence payload changed: {image_id}")
        candidate_row = candidate_rows[Path(image_id).stem]
        if candidate_row["diagnostic_sha256"] != evidence_row["candidate_payload_sha256"]:
            raise ValueError(f"candidate/G1 payload mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            candidate_indices = payload["candidate_indices"].astype(np.int64)
            g1_logits = payload["candidate_logits"].astype(np.float64)
            upstream_scores = payload["selection_scores"].astype(np.float64)
            sources = payload["proposal_source_ids"].astype(str)
        if not (len(candidate_indices) == len(g1_logits) == len(upstream_scores) == len(sources)):
            raise ValueError(f"selector arrays differ: {image_id}")
        local_index, fused = select_candidate(g1_logits, upstream_scores)
        selections.append(
            {
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "tumor": split_row["tumor"],
                "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                "candidate_count": len(candidate_indices),
                "selected_local_index": local_index,
                "selected_candidate_index": int(candidate_indices[local_index]),
                "selected_source": str(sources[local_index]),
                "selected_g1_logit": float(g1_logits[local_index]),
                "selected_upstream_score": float(upstream_scores[local_index]),
                "selected_fused_rank": float(fused[local_index]),
            }
        )

    manifest_path = args.output_dir / "selection_manifest.csv"
    manifest_sha = _write_csv(manifest_path, selections)
    freeze = {
        "stage": "final_rich_gallery_choice_freeze_v1",
        "cohort_split": args.split,
        "method": "G1 + 0.5/0.5 percentile-rank fusion",
        "split_sha256": args.expected_split_sha256,
        "g1_diagnostic_freeze_sha256": args.expected_g1_freeze_sha256,
        "g1_checkpoint_sha256": g1_freeze["baseline_checkpoint_sha256"],
        "g1_prediction_manifest_sha256": g1_freeze.get("baseline_prediction_manifest_sha256"),
        "candidate_manifest_sha256": candidate_summary["manifest_sha256"],
        "pseudo_manifest_sha256": args.expected_pseudo_manifest_sha256,
        "selection_manifest_sha256": manifest_sha,
        "images": counts["images"],
        "tumor_images": counts["tumor"],
        "normal_images": counts["normal"],
        "candidate_choices_frozen_before_spatial_gt": True,
        "candidate_choices_frozen_before_validation_gt": args.split == "val",
        "candidate_choices_frozen_before_test_gt": args.split == "test",
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": counts["images"] if args.split == "test" else 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
