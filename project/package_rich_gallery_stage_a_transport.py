"""Create an allow-listed, GT-blind transport for the rich-gallery B2 successor."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from pseudo.candidate_diagnostics import (
    MANIFEST_NAME,
    SUMMARY_NAME,
    validate_candidate_diagnostics_manifest,
)
from pseudo.manifest import sha256_file


EXPECTED_STAGE = "rich_gallery_g2_selector_pair_stage_a_v1"
EXPECTED_VARIANT = "g1_frozen__rank_fusion"
FORBIDDEN_TOKENS = {
    "annotation",
    "annotations",
    "evaluation",
    "ground_truth",
    "mask_gt",
    "per_image",
    "polygon",
    "polygons",
    "stage_b",
    "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def path_tokens(path: Path | str) -> set[str]:
    relative = Path(path)
    tokens: set[str] = set()
    for part in relative.parts:
        lowered = part.lower().replace("-", "_")
        stem = lowered.rsplit(".", 1)[0]
        tokens.add(lowered)
        tokens.add(stem)
        tokens.update(stem.split("_"))
    return tokens


def reject_forbidden_path(path: Path | str) -> None:
    overlap = path_tokens(path) & FORBIDDEN_TOKENS
    if overlap:
        raise ValueError(f"forbidden GT/Stage-B/test token in transport path: {path}")


def safe_child(root: Path, relative: Path | str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("absolute transport path")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError("transport path escapes its source root")
    return resolved


def inspect_npz_keys(path: Path) -> tuple[str, ...]:
    with np.load(path, allow_pickle=False) as payload:
        keys = tuple(sorted(payload.files))
        for key in keys:
            reject_forbidden_path(key)
            value = payload[key]
            if value.dtype.hasobject:
                raise ValueError(f"object array is forbidden in GT-blind transport: {path}")
    if not keys:
        raise ValueError(f"empty NPZ payload in GT-blind transport: {path}")
    return keys


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty transport inventory")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def inventory(root: Path, relative_paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for relative in sorted(relative_paths, key=lambda path: path.as_posix()):
        reject_forbidden_path(relative)
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate transport inventory path: {normalized}")
        seen.add(normalized)
        path = safe_child(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"transport inventory file is missing: {normalized}")
        rows.append(
            {
                "path": normalized,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _verify_stage_a(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, str]]]:
    freeze_path = args.stage_a_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_freeze_sha256:
        raise ValueError("Stage-A freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != EXPECTED_STAGE
        or freeze.get("source_commit") != args.expected_source_commit
        or freeze.get("protocol_sha256") != args.expected_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("g1_checkpoint_sha256") != args.expected_g1_checkpoint_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Stage-A freeze provenance/safety mismatch")
    selection_path = args.stage_a_root / "stage_a_selection_manifest.csv"
    if sha256_file(selection_path) != freeze.get("selection_manifest_sha256"):
        raise ValueError("Stage-A selection manifest SHA-256 mismatch")
    rows = _load_csv(selection_path)
    baseline = [row for row in rows if row.get("variant") == EXPECTED_VARIANT]
    if (
        len(rows) != int(freeze.get("selection_rows", -1))
        or len(baseline) != 371
        or len({row["image_id"] for row in baseline}) != 371
        or sum(int(row["tumor"]) for row in baseline) != 184
    ):
        raise ValueError("Stage-A selection cohort is incomplete")
    return freeze, baseline


def package(args: argparse.Namespace) -> dict[str, object]:
    if args.output_root.exists():
        raise FileExistsError("GT-blind transport output already exists")
    freeze, baseline = _verify_stage_a(args)
    image_ids = [row["image_id"] for row in baseline]
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=image_ids,
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all" or len(candidate_rows) != 371:
        raise ValueError("complete all-image validation gallery is required")

    source_plan: dict[Path, Path] = {
        args.stage_a_root / "prediction_freeze.json": Path("prediction_freeze.json"),
        args.stage_a_root / "stage_a_selection_manifest.csv": Path(
            "stage_a_selection_manifest.csv"
        ),
        args.val_candidate_root / MANIFEST_NAME: Path("val_candidates") / MANIFEST_NAME,
        args.val_candidate_root / SUMMARY_NAME: Path("val_candidates") / SUMMARY_NAME,
    }
    score_hashes: set[str] = set()
    for row in baseline:
        relative = Path(row["score_path"])
        reject_forbidden_path(relative)
        source = safe_child(args.stage_a_root, relative)
        if sha256_file(source) != row["score_sha256"]:
            raise ValueError(f"Stage-A score hash mismatch: {row['image_id']}")
        inspect_npz_keys(source)
        source_plan[source] = relative
        score_hashes.add(row["score_sha256"])
        candidate_row = candidate_rows[Path(row["image_id"]).stem]
        candidate_relative = Path(candidate_row["diagnostic_path"])
        reject_forbidden_path(candidate_relative)
        candidate_source = safe_child(args.val_candidate_root, candidate_relative)
        if sha256_file(candidate_source) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload hash mismatch: {row['image_id']}")
        inspect_npz_keys(candidate_source)
        source_plan[candidate_source] = Path("val_candidates") / candidate_relative
    if len(score_hashes) != 371 or len(source_plan) != 746:
        raise ValueError("GT-blind transport copy plan is incomplete")
    score_set = sha256("\n".join(sorted(score_hashes)).encode()).hexdigest()
    if score_set != freeze.get("score_set_sha256"):
        raise ValueError("Stage-A score-set SHA-256 mismatch")

    for source, relative in source_plan.items():
        reject_forbidden_path(relative)
        destination = args.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    copied_paths = list(source_plan.values())
    inventory_rows = inventory(args.output_root, copied_paths)
    inventory_path = args.output_root / "transport_inventory.csv"
    inventory_sha256 = _write_csv(inventory_path, inventory_rows)
    transport_freeze = {
        "stage": "rich_gallery_b2_gt_blind_transport_v1",
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "g1_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "stage_a_prediction_freeze_sha256": args.expected_freeze_sha256,
        "stage_a_selection_manifest_sha256": freeze["selection_manifest_sha256"],
        "stage_a_score_set_sha256": score_set,
        "candidate_manifest_sha256": candidate_audit["manifest_sha256"],
        "candidate_summary_sha256": candidate_audit["summary_sha256"],
        "candidate_pseudo_manifest_sha256": candidate_audit[
            "pseudo_manifest_sha256"
        ],
        "transport_inventory_sha256": inventory_sha256,
        "inventory_rows": len(inventory_rows),
        "validation_images": 371,
        "tumor_image_labels": 184,
        "normal_image_labels": 187,
        "validation_gt_included": False,
        "spatial_ground_truth_included": False,
        "stage_b_included": False,
        "test_included": False,
        "consumer_included": False,
    }
    freeze_output = args.output_root / "transport_freeze.json"
    freeze_output.write_text(
        json.dumps(transport_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **transport_freeze,
        "transport_freeze_sha256": sha256_file(freeze_output),
    }


def main() -> None:
    result = package(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
