"""Independent GT-blind auditor for collaborator rich-gallery Stage-A transport."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from models.rich_gallery_bas_residual import (
    RichGalleryAlignedPayload,
    align_transport_payloads,
    score_rich_gallery_bas_pair,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from pseudo.manifest import sha256_file


FORBIDDEN_TRANSPORT_PARTS = {
    "annotation",
    "annotations",
    "evaluation",
    "ground_truth",
    "per_image",
    "polygon",
    "polygons",
    "stage_b",
}
EXPECTED_FREEZE_STAGE = "rich_gallery_g2_selector_pair_stage_a_v1"
EXPECTED_BASELINE_VARIANT = "g1_frozen__rank_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def find_forbidden_transport_paths(root: Path) -> list[str]:
    """Reject Stage-B/GT-derived material from the descriptor-side transport."""

    if not root.is_dir():
        raise FileNotFoundError("rich-gallery transport root is missing")
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        lowered_parts = {part.lower() for part in relative.parts}
        stem_tokens = set(relative.stem.lower().replace("-", "_").split("_"))
        if lowered_parts & FORBIDDEN_TRANSPORT_PARTS or stem_tokens & FORBIDDEN_TRANSPORT_PARTS:
            forbidden.append(relative.as_posix())
    return sorted(forbidden)


def safe_transport_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("absolute path in Stage-A selection manifest")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError("Stage-A score path escapes transport root")
    return resolved


def load_npz_mapping(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def audit_g1_baseline_row(
    row: Mapping[str, str],
    candidate_payload: Mapping[str, object],
    stage_a_payload: Mapping[str, object],
) -> RichGalleryAlignedPayload:
    """Reproduce one frozen G1+upstream choice without validation GT."""

    if row.get("variant") != EXPECTED_BASELINE_VARIANT:
        raise ValueError("transport row is not the frozen G1 rank-fusion baseline")
    aligned = align_transport_payloads(candidate_payload, stage_a_payload)
    # A zero BAS vector leaves the two-rank control unchanged; only that control
    # is inspected at transport time. BAS is fitted later under a separate claim.
    pair = score_rich_gallery_bas_pair(
        aligned.g1_logits,
        aligned.upstream_scores,
        np.zeros_like(aligned.g1_logits),
    )
    local = pair.baseline_local_index
    if (
        int(row["selected_local_index"]) != local
        or int(row["selected_candidate_index"])
        != int(aligned.candidate_indices[local])
    ):
        raise ValueError(f"frozen G1 rank-fusion choice does not reproduce: {row['image_id']}")
    return aligned


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _score_set_sha256(score_hashes: list[str]) -> str:
    return sha256("\n".join(sorted(score_hashes)).encode()).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("rich-gallery transport split SHA-256 mismatch")
    forbidden = find_forbidden_transport_paths(args.transport_root)
    if forbidden:
        raise ValueError(f"GT/Stage-B paths present in transport: {forbidden[:5]}")

    freeze_path = args.transport_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_freeze_sha256:
        raise ValueError("rich-gallery Stage-A freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != EXPECTED_FREEZE_STAGE
        or freeze.get("source_commit") != args.expected_source_commit
        or freeze.get("protocol_sha256") != args.expected_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("g1_checkpoint_sha256")
        != args.expected_g1_checkpoint_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
        or freeze.get("g1_reproduction_max_selected_index_delta") != 0
        or freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("rich-gallery Stage-A safety/provenance contract mismatch")

    selection_path = args.transport_root / "stage_a_selection_manifest.csv"
    if sha256_file(selection_path) != freeze.get("selection_manifest_sha256"):
        raise ValueError("rich-gallery Stage-A selection manifest changed")
    rows = _load_csv(selection_path)
    if len(rows) != int(freeze.get("selection_rows", -1)):
        raise ValueError("rich-gallery Stage-A selection row count mismatch")
    baseline_rows = [row for row in rows if row.get("variant") == EXPECTED_BASELINE_VARIANT]
    if len(baseline_rows) != 371:
        raise ValueError("rich-gallery G1 rank-fusion baseline cohort is incomplete")
    image_ids = [row["image_id"] for row in baseline_rows]
    if len(set(image_ids)) != 371:
        raise ValueError("rich-gallery G1 rank-fusion image ids are duplicated")
    labels = [int(row["tumor"]) for row in baseline_rows]
    if labels.count(1) != 184 or labels.count(0) != 187:
        raise ValueError("rich-gallery Stage-A image-label cohort mismatch")

    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=image_ids,
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all" or len(candidate_rows) != 371:
        raise ValueError("rich-gallery candidate transport does not cover all validation images")

    unique_score_hashes: dict[str, str] = {}
    total_candidates = 0
    source_counts = {"0": 0, "1": 0, "2": 0}
    for row in baseline_rows:
        image_id = row["image_id"]
        stem = Path(image_id).stem
        candidate_row = candidate_rows[stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if (
            candidate_row["diagnostic_sha256"] != row["candidate_payload_sha256"]
            or sha256_file(candidate_path) != row["candidate_payload_sha256"]
        ):
            raise ValueError(f"rich-gallery candidate payload drift: {image_id}")
        score_path = safe_transport_path(args.transport_root, row["score_path"])
        if sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"rich-gallery Stage-A score payload drift: {image_id}")
        prior = unique_score_hashes.setdefault(row["score_path"], row["score_sha256"])
        if prior != row["score_sha256"]:
            raise ValueError(f"conflicting Stage-A score hash: {row['score_path']}")
        aligned = audit_g1_baseline_row(
            row,
            load_npz_mapping(candidate_path),
            load_npz_mapping(score_path),
        )
        total_candidates += len(aligned.candidate_indices)
        for source_id in aligned.source_ids:
            key = str(int(source_id))
            if key not in source_counts:
                raise ValueError(f"unknown Stage-A source id: {source_id}")
            source_counts[key] += 1

    if len(unique_score_hashes) != 371:
        raise ValueError("rich-gallery Stage-A score payload cohort is incomplete")
    score_set = _score_set_sha256(list(unique_score_hashes.values()))
    if score_set != freeze.get("score_set_sha256"):
        raise ValueError("rich-gallery Stage-A score-set hash mismatch")

    return {
        "stage": "rich_gallery_stage_a_transport_independent_audit_v1",
        "audit_pass": True,
        "prediction_freeze_sha256": args.expected_freeze_sha256,
        "selection_manifest_sha256": sha256_file(selection_path),
        "score_set_sha256": score_set,
        "candidate_manifest_sha256": candidate_audit["manifest_sha256"],
        "candidate_summary_sha256": candidate_audit["summary_sha256"],
        "g1_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "kept_candidates": total_candidates,
        "kept_source_counts": source_counts,
        "g1_rank_fusion_choices_reproduced": 371,
        "forbidden_transport_paths": [],
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("transport audit output already exists")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "audit_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
