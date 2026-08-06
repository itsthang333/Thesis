from __future__ import annotations

"""Freeze G4 E4/E5/E6/E8 choices without opening spatial annotations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from final_selector import stable_select
from frozen_io import load_split_rows_without_annotations, sha256_file
from g4_ablation import (
    ALL_SOURCES,
    FUSION_ARMS,
    candidate_filter,
    deterministic_random_candidate,
    fusion_score,
    select_score_only,
)


SOURCE_ARMS = {
    "L320": ("layercam",),
    "C448": ("classifier448:layercam",),
    "External": ("external_saliency",),
    "L320+C448": ("layercam", "classifier448:layercam"),
    "L320+External": ("layercam", "external_saliency"),
    "C448+External": ("classifier448:layercam", "external_saliency"),
    "All": ALL_SOURCES,
}
CAP_ARMS = {"27": 9, "81": 27, "162": 54, "243": 81}
RANDOM_SEED = 20260806


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--expected-g1-freeze-sha256", required=True)
    parser.add_argument("--baseline-selection-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
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


def _candidate_manifest(candidate_root: Path, expected_sha: str) -> dict[str, dict[str, str]]:
    path = candidate_root / "candidate_diagnostics_manifest.csv"
    if sha256_file(path) != expected_sha:
        raise ValueError("candidate manifest SHA-256 mismatch")
    rows = _read_csv(path)
    indexed = {str(row["image_name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("candidate manifest image IDs are not unique")
    return indexed


def _choose(
    arm: str,
    indices: np.ndarray,
    g1: np.ndarray,
    upstream: np.ndarray,
    sam: np.ndarray,
    *,
    image_id: str,
) -> tuple[int, np.ndarray]:
    if arm == "random":
        selected = deterministic_random_candidate(image_id, indices, RANDOM_SEED)
        return selected, np.zeros(len(indices), dtype=np.float64)
    if arm == "sam_only":
        return select_score_only(sam, indices), sam
    if arm == "upstream_only":
        return select_score_only(upstream, indices), upstream
    if arm == "g1_only":
        return select_score_only(g1, indices), g1
    if arm.startswith("R"):
        scores = fusion_score(g1, upstream, arm)
        if arm in {"R0", "R1"}:
            return select_score_only(scores, indices), scores
        local = stable_select(scores, g1)
        return int(indices[local]), scores
    raise ValueError(f"unknown selector arm: {arm}")


def _filter_or_empty(*args, **kwargs) -> np.ndarray:
    try:
        return candidate_filter(*args, **kwargs)
    except ValueError as error:
        if str(error) != "candidate filter produced an empty gallery":
            raise
        return np.zeros(0, dtype=np.int64)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    split_by_id = {str(row["image_id"]): row for row in split_rows}
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("G4 offline ablations require the canonical 371/184 validation cohort")

    candidate_by_id = _candidate_manifest(
        args.candidate_root, args.expected_candidate_manifest_sha256
    )
    if set(candidate_by_id) != set(split_by_id):
        raise ValueError("candidate cohort differs from canonical validation")

    g1_freeze_path = args.g1_root / "diagnostic_freeze.json"
    if sha256_file(g1_freeze_path) != args.expected_g1_freeze_sha256:
        raise ValueError("G1 freeze SHA-256 mismatch")
    g1_freeze = json.loads(g1_freeze_path.read_text(encoding="utf-8"))
    if (
        g1_freeze.get("spatial_ground_truth_used") is not False
        or g1_freeze.get("validation_gt_read") is not False
        or g1_freeze.get("test_evaluated") is not False
        or g1_freeze.get("split_sha256") != args.expected_split_sha256
    ):
        raise ValueError("G1 freeze violates the annotation-free validation contract")
    evidence_manifest_path = args.g1_root / "descriptor_evidence_manifest.csv"
    if sha256_file(evidence_manifest_path) != g1_freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("G1 evidence manifest changed after freezing")
    evidence_by_id = {row["image_id"]: row for row in _read_csv(evidence_manifest_path)}
    if set(evidence_by_id) != set(split_by_id):
        raise ValueError("G1 evidence cohort differs from canonical validation")

    baseline_freeze_path = args.baseline_selection_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze_path) != args.expected_baseline_freeze_sha256:
        raise ValueError("baseline choice freeze SHA-256 mismatch")
    baseline_freeze = json.loads(baseline_freeze_path.read_text(encoding="utf-8"))
    baseline_manifest_path = args.baseline_selection_root / "selection_manifest.csv"
    if sha256_file(baseline_manifest_path) != baseline_freeze["selection_manifest_sha256"]:
        raise ValueError("baseline selection manifest changed after freezing")
    baseline_by_id = {row["image_id"]: row for row in _read_csv(baseline_manifest_path)}

    output_rows: list[dict[str, object]] = []
    arm_names: list[str] = []
    for selector in ("random", "sam_only", "upstream_only", "g1_only"):
        arm_names.append(f"E6__{selector}")
    arm_names.extend(f"E8__{arm}" for arm in FUSION_ARMS)
    arm_names.extend(f"E4__{label}" for label in SOURCE_ARMS)
    arm_names.extend(f"E5_cap__{label}" for label in CAP_ARMS)
    arm_names.extend(f"E5_prompt_mode__{mode}" for mode in ("point", "box", "box_point"))

    baseline_matches = 0
    u5_residuals: list[float] = []
    for split_row in split_rows:
        image_id = str(split_row["image_id"])
        candidate_row = candidate_by_id[image_id]
        candidate_path = args.candidate_root / str(candidate_row["diagnostic_path"])
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        evidence_row = evidence_by_id[image_id]
        evidence_path = args.g1_root / "descriptor_evidence" / evidence_row["evidence_path"]
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise ValueError(f"G1 evidence payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as candidate, np.load(
            evidence_path, allow_pickle=False
        ) as evidence:
            masks_count = len(candidate["sam_masks"])
            candidate_indices = evidence["candidate_indices"].astype(np.int64)
            if (
                candidate_indices.ndim != 1
                or len(candidate_indices) == 0
                or len(np.unique(candidate_indices)) != len(candidate_indices)
                or candidate_indices.min() < 0
                or candidate_indices.max() >= masks_count
            ):
                raise ValueError(f"invalid G1 candidate-index subset: {image_id}")
            all_sources = candidate["proposal_source_ids"].astype(str)
            all_prompt_modes = candidate["prompt_modes"].astype(str)
            all_sam_scores = candidate["sam_scores"].astype(np.float64)
            all_upstream = candidate["selection_scores"].astype(np.float64)
            sources = all_sources[candidate_indices]
            prompt_modes = all_prompt_modes[candidate_indices]
            sam_scores = all_sam_scores[candidate_indices]
            stored_upstream = all_upstream[candidate_indices]
            g1 = evidence["candidate_logits"].astype(np.float64)
            evidence_upstream = evidence["selection_scores"].astype(np.float64)
            if not np.array_equal(sources, evidence["proposal_source_ids"].astype(str)):
                raise ValueError(f"G1 candidate sources differ: {image_id}")
            residual = float(np.max(np.abs(stored_upstream - evidence_upstream)))
            u5_residuals.append(residual)
            if residual > 1e-7:
                raise ValueError(f"frozen upstream scores differ: {image_id}: {residual}")

            selections: dict[str, tuple[int, np.ndarray]] = {}
            eligible_by_arm: dict[str, np.ndarray] = {}
            full_indices = candidate_indices
            for selector in ("random", "sam_only", "upstream_only", "g1_only"):
                arm_name = f"E6__{selector}"
                selections[arm_name] = _choose(
                    selector,
                    full_indices,
                    g1,
                    stored_upstream,
                    sam_scores,
                    image_id=image_id,
                )
                eligible_by_arm[arm_name] = full_indices
            for arm in FUSION_ARMS:
                arm_name = f"E8__{arm}"
                selections[arm_name] = _choose(
                    arm,
                    full_indices,
                    g1,
                    stored_upstream,
                    sam_scores,
                    image_id=image_id,
                )
                eligible_by_arm[arm_name] = full_indices
            for label, allowed_sources in SOURCE_ARMS.items():
                local_indices = _filter_or_empty(
                    sources, stored_upstream, allowed_sources=allowed_sources
                )
                indices = candidate_indices[local_indices]
                if len(indices):
                    selected, scores = _choose(
                        "R7",
                        indices,
                        g1[local_indices],
                        stored_upstream[local_indices],
                        sam_scores[local_indices],
                        image_id=image_id,
                    )
                else:
                    selected, scores = -1, np.zeros(0, dtype=np.float64)
                arm_name = f"E4__{label}"
                selections[arm_name] = selected, scores
                eligible_by_arm[arm_name] = indices
            for label, per_source_cap in CAP_ARMS.items():
                local_indices = _filter_or_empty(
                    sources,
                    stored_upstream,
                    allowed_sources=ALL_SOURCES,
                    per_source_cap=per_source_cap,
                )
                indices = candidate_indices[local_indices]
                selected, scores = _choose(
                    "R7",
                    indices,
                    g1[local_indices],
                    stored_upstream[local_indices],
                    sam_scores[local_indices],
                    image_id=image_id,
                )
                arm_name = f"E5_cap__{label}"
                selections[arm_name] = selected, scores
                eligible_by_arm[arm_name] = indices
            for mode in ("point", "box", "box_point"):
                local_indices = _filter_or_empty(
                    sources,
                    stored_upstream,
                    allowed_sources=ALL_SOURCES,
                    prompt_mode=mode,
                    prompt_modes=prompt_modes,
                )
                indices = candidate_indices[local_indices]
                if len(indices):
                    selected, scores = _choose(
                        "R7",
                        indices,
                        g1[local_indices],
                        stored_upstream[local_indices],
                        sam_scores[local_indices],
                        image_id=image_id,
                    )
                else:
                    selected, scores = -1, np.zeros(0, dtype=np.float64)
                arm_name = f"E5_prompt_mode__{mode}"
                selections[arm_name] = selected, scores
                eligible_by_arm[arm_name] = indices

        baseline_selected = int(baseline_by_id[image_id]["selected_candidate_index"])
        if selections["E8__R7"][0] == baseline_selected:
            baseline_matches += 1
        for arm in arm_names:
            selected, scores = selections[arm]
            eligible_indices = np.asarray(eligible_by_arm[arm], dtype=np.int64)
            if selected >= 0 and selected not in set(eligible_indices.tolist()):
                raise ValueError(f"selected candidate is not eligible: {image_id}/{arm}")
            selected_local = (
                int(np.flatnonzero(candidate_indices == selected)[0]) if selected >= 0 else -1
            )
            output_rows.append(
                {
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "tumor": split_row["tumor"],
                    "arm": arm,
                    "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                    "gallery_candidate_count": masks_count,
                    "g1_eligible_candidate_count": len(candidate_indices),
                    "eligible_candidate_count": len(eligible_indices),
                    "eligible_candidate_indices": ";".join(
                        str(int(index)) for index in eligible_indices
                    ),
                    "selected_candidate_index": selected,
                    "selected_source": (
                        str(all_sources[selected]) if selected >= 0 else "empty_no_eligible_candidate"
                    ),
                    "selected_prompt_mode": (
                        str(all_prompt_modes[selected]) if selected >= 0 else ""
                    ),
                    "selected_sam_score": (
                        float(all_sam_scores[selected]) if selected >= 0 else ""
                    ),
                    "selected_upstream_score": (
                        float(all_upstream[selected]) if selected >= 0 else ""
                    ),
                    "selected_g1_logit": float(
                        g1[selected_local]
                    ) if selected_local >= 0 else "",
                }
            )

    if baseline_matches != len(split_rows):
        raise ValueError(
            f"E8 R7 reproduced only {baseline_matches}/{len(split_rows)} frozen choices"
        )
    choices_path = args.output_dir / "g4_choices.csv"
    choices_sha = _write_csv(choices_path, output_rows)
    freeze = {
        "schema_version": 1,
        "stage": "g4_offline_ablation_choice_freeze_v1",
        "cohort_split": "val",
        "split_sha256": args.expected_split_sha256,
        "candidate_manifest_sha256": args.expected_candidate_manifest_sha256,
        "g1_freeze_sha256": args.expected_g1_freeze_sha256,
        "baseline_freeze_sha256": args.expected_baseline_freeze_sha256,
        "choices_sha256": choices_sha,
        "images": len(split_rows),
        "tumor_images": sum(int(row["tumor"]) for row in split_rows),
        "arms": arm_names,
        "selection_rows": len(output_rows),
        "random_seed": RANDOM_SEED,
        "baseline_r7_exact_matches": baseline_matches,
        "stored_upstream_g1_max_abs_residual": max(u5_residuals),
        "eligible_candidate_indices_frozen_per_image_arm": True,
        "candidate_choices_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "limitations": {
            "E5_prompt_mode_arms": "one prompt mode across the gallery, not one exact prompt ID",
            "E5_dedup_and_exact_single_prompt": "require regenerated payloads with prompt IDs",
            "E7": "requires source-specific prompt maps; merged artifact intentionally cannot reconstruct it",
        },
    }
    freeze_path = args.output_dir / "g4_choice_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
