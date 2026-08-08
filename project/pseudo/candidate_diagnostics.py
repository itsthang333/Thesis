from __future__ import annotations

"""Prediction-first candidate diagnostics with an immutable integrity manifest.

This module deliberately has no dependency on segmentation ground truth.  The
generation stage serializes every candidate/prompt/selected mask first, then a
separate evaluator may load GT only after this manifest has been verified.
"""

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .manifest import sha256_file


MANIFEST_NAME = "candidate_diagnostics_manifest.csv"
SUMMARY_NAME = "candidate_diagnostics_summary.json"


def save_candidate_diagnostics(
    path: str | Path,
    *,
    sam_masks: np.ndarray,
    refined_mask: np.ndarray,
    final_mask: np.ndarray,
    bone_support: np.ndarray | None,
    prompt_map: np.ndarray,
    positive_points: Iterable[tuple[int, int]],
    negative_points: Iterable[tuple[int, int]],
    boxes: Iterable[tuple[int, int, int, int]],
    sam_scores: np.ndarray,
    selection_scores: np.ndarray,
    classifier_causal_scores: np.ndarray | None,
    component_ids: np.ndarray | None,
    prompt_modes: Iterable[str],
    proposal_source_ids: Iterable[str] | None = None,
    cam_levels: Iterable[float] | None = None,
    prompt_ids: Iterable[str] | None = None,
    multimask_indices: Iterable[int] | None = None,
) -> dict[str, object]:
    """Save one pickle-free NPZ without ever consulting segmentation GT."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(value) for value in final_mask.shape)
    if len(shape) != 2:
        raise ValueError(f"final_mask must be 2-D, got {shape}")

    candidates = np.asarray(sam_masks, dtype=np.uint8)
    if candidates.ndim != 3 or tuple(candidates.shape[1:]) != shape:
        raise ValueError(
            f"sam_masks must have shape [N,{shape[0]},{shape[1]}], got {candidates.shape}"
        )
    scores = np.asarray(sam_scores, dtype=np.float32).reshape(-1)
    selection = np.asarray(selection_scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(candidates) or len(selection) != len(candidates):
        raise ValueError("Candidate, SAM-score and selection-score counts differ")

    positive = np.asarray(list(positive_points), dtype=np.int32).reshape(-1, 2)
    negative = np.asarray(list(negative_points), dtype=np.int32).reshape(-1, 2)
    prompt_boxes = np.asarray(list(boxes), dtype=np.int32).reshape(-1, 4)
    modes = np.asarray(list(prompt_modes), dtype="U32").reshape(-1)
    if len(modes) != len(candidates):
        raise ValueError("Candidate and prompt-mode counts differ")
    sources_present = proposal_source_ids is not None
    sources = (
        np.asarray(list(proposal_source_ids), dtype="U32").reshape(-1)
        if sources_present
        else np.full(len(candidates), "unassigned", dtype="U32")
    )
    if len(sources) != len(candidates):
        raise ValueError("Candidate and proposal-source counts differ")
    provenance_present = any(
        value is not None for value in (cam_levels, prompt_ids, multimask_indices)
    )
    if provenance_present and not all(
        value is not None for value in (cam_levels, prompt_ids, multimask_indices)
    ):
        raise ValueError("Exact candidate provenance must be supplied as one complete set")
    levels = (
        np.asarray(list(cam_levels), dtype=np.float32).reshape(-1)
        if cam_levels is not None
        else np.full(len(candidates), -1.0, dtype=np.float32)
    )
    exact_prompt_ids = (
        np.asarray(list(prompt_ids), dtype="U128").reshape(-1)
        if prompt_ids is not None
        else np.full(len(candidates), "", dtype="U128")
    )
    multimask = (
        np.asarray(list(multimask_indices), dtype=np.int16).reshape(-1)
        if multimask_indices is not None
        else np.full(len(candidates), -1, dtype=np.int16)
    )
    if not (
        len(levels) == len(exact_prompt_ids) == len(multimask) == len(candidates)
    ):
        raise ValueError("Candidate and exact-provenance counts differ")

    support_present = bone_support is not None
    support = (
        np.asarray(bone_support, dtype=np.uint8)
        if support_present
        else np.zeros(shape, dtype=np.uint8)
    )
    causal = (
        np.asarray(classifier_causal_scores, dtype=np.float32).reshape(-1)
        if classifier_causal_scores is not None
        else np.zeros(len(candidates), dtype=np.float32)
    )
    components = (
        np.asarray(component_ids, dtype=np.int32).reshape(-1)
        if component_ids is not None
        else np.full(len(candidates), -1, dtype=np.int32)
    )
    if len(causal) != len(candidates) or len(components) != len(candidates):
        raise ValueError("Candidate diagnostic vectors have inconsistent lengths")

    np.savez_compressed(
        path,
        schema_version=np.asarray(
            [3 if provenance_present else (2 if sources_present else 1)],
            dtype=np.int32,
        ),
        sam_masks=candidates,
        refined_mask=np.asarray(refined_mask, dtype=np.uint8),
        final_mask=np.asarray(final_mask, dtype=np.uint8),
        bone_support=support,
        bone_support_present=np.asarray([int(support_present)], dtype=np.uint8),
        prompt_map=np.asarray(prompt_map, dtype=np.float32),
        positive_points=positive,
        negative_points=negative,
        boxes=prompt_boxes,
        sam_scores=scores,
        selection_scores=selection,
        classifier_causal_scores=causal,
        component_ids=components,
        prompt_modes=modes,
        proposal_source_ids=sources,
        cam_levels=levels,
        prompt_ids=exact_prompt_ids,
        multimask_indices=multimask,
    )
    return {
        "diagnostic_path": str(Path("candidate_diagnostics") / path.name),
        "diagnostic_sha256": sha256_file(path),
        "diagnostic_bytes": path.stat().st_size,
        "candidate_count": len(candidates),
        "positive_point_count": len(positive),
        "negative_point_count": len(negative),
        "box_count": len(prompt_boxes),
        "bone_support_present": int(support_present),
    }


def write_candidate_diagnostics_manifest(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    *,
    expected_image_names: Iterable[str],
    split: str,
    image_size: int,
    pseudo_manifest_sha256: str,
    selection_method: str,
    support_clip_kernel: int,
    cam_percentile: float,
    cohort: str = "tumor",
) -> dict[str, object]:
    if cohort not in {"tumor", "all"}:
        raise ValueError("Candidate diagnostic cohort must be 'tumor' or 'all'")
    output_dir = Path(output_dir)
    expected = {Path(str(name)).stem for name in expected_image_names}
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        stem = Path(str(row.get("image_name", ""))).stem
        if not stem or stem in indexed:
            raise ValueError(f"Duplicate/empty candidate diagnostic image id: {stem!r}")
        indexed[stem] = row
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected:
        cohort_label = "tumor cohort" if cohort == "tumor" else "image cohort"
        raise RuntimeError(
            f"Candidate diagnostics do not cover the complete {cohort_label}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for stem, row in indexed.items():
        path = output_dir / str(row["diagnostic_path"])
        if not path.is_file() or sha256_file(path) != row["diagnostic_sha256"]:
            raise ValueError(f"Candidate diagnostic file/hash mismatch for {stem}")

    manifest_path = output_dir / MANIFEST_NAME
    fieldnames = sorted({key for row in rows for key in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "complete": True,
        "prediction_first": True,
        "ground_truth_loaded_during_generation": False,
        "split": split,
        "cohort": cohort,
        "expected_images": len(expected),
        "expected_tumor_images": len(expected) if cohort == "tumor" else None,
        "manifest_rows": len(rows),
        "image_size": int(image_size),
        "pseudo_manifest_sha256": pseudo_manifest_sha256,
        "selection_method": selection_method,
        "support_clip_kernel": int(support_clip_kernel),
        "cam_percentile": float(cam_percentile),
        "manifest_sha256": sha256_file(manifest_path),
    }
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**summary, "summary_sha256": sha256_file(summary_path)}


def validate_candidate_diagnostics_manifest(
    output_dir: str | Path,
    *,
    expected_image_names: Iterable[str],
    split: str,
    expected_pseudo_manifest_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    summary_path = output_dir / SUMMARY_NAME
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Frozen candidate diagnostic manifest/summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(manifest_path)
    if (
        int(summary.get("schema_version", -1)) != 1
        or not summary.get("complete")
        or not summary.get("prediction_first")
        or summary.get("ground_truth_loaded_during_generation") is not False
    ):
        raise ValueError("Candidate diagnostic summary is not a complete prediction-first artifact")
    if summary.get("split") != split:
        raise ValueError(f"Candidate diagnostics split={summary.get('split')!r}, expected {split!r}")
    if summary.get("cohort", "tumor") not in {"tumor", "all"}:
        raise ValueError("Candidate diagnostics contain an unsupported cohort")
    if summary.get("manifest_sha256") != actual_manifest_hash:
        raise ValueError("Candidate diagnostic manifest hash differs from its frozen summary")
    if expected_manifest_sha256 and actual_manifest_hash != expected_manifest_sha256:
        raise ValueError("Candidate diagnostic manifest differs from the caller-locked hash")
    if (
        expected_pseudo_manifest_sha256
        and summary.get("pseudo_manifest_sha256") != expected_pseudo_manifest_sha256
    ):
        raise ValueError("Candidate diagnostics were not frozen against this pseudo-mask manifest")

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        stem = Path(row.get("image_name", "")).stem
        if not stem or stem in indexed:
            raise ValueError(f"Duplicate/empty candidate diagnostic image id: {stem!r}")
        indexed[stem] = row
    expected = {Path(str(name)).stem for name in expected_image_names}
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected:
        raise ValueError(
            f"Candidate diagnostic cohort mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    expected_count = int(
        summary.get("expected_images", summary.get("expected_tumor_images", -1))
    )
    if (
        int(summary.get("manifest_rows", -1)) != len(rows)
        or expected_count != len(expected)
        or len(rows) != len(expected)
    ):
        raise ValueError("Candidate diagnostic summary counts are inconsistent")

    for stem, row in indexed.items():
        path = output_dir / row["diagnostic_path"]
        if not path.is_file() or sha256_file(path) != row["diagnostic_sha256"]:
            raise ValueError(f"Candidate diagnostic file/hash mismatch for {stem}")
        with np.load(path, allow_pickle=False) as payload:
            schema_version = int(payload["schema_version"][0])
            if schema_version not in {1, 2}:
                raise ValueError(f"Unsupported candidate diagnostic schema for {stem}")
            expected_shape = (int(summary["image_size"]),) * 2
            if tuple(payload["final_mask"].shape) != expected_shape:
                raise ValueError(f"Candidate diagnostic shape mismatch for {stem}")
            if int(row["candidate_count"]) != int(payload["sam_masks"].shape[0]):
                raise ValueError(f"Candidate diagnostic count mismatch for {stem}")
            if schema_version >= 2:
                if (
                    "proposal_source_ids" not in payload
                    or len(payload["proposal_source_ids"]) != len(payload["sam_masks"])
                ):
                    raise ValueError(
                        f"Candidate proposal-source provenance mismatch for {stem}"
                    )

    return indexed, {
        **summary,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_hash,
        "summary_sha256": sha256_file(summary_path),
    }
