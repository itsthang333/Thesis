from __future__ import annotations

"""Freeze all exact G4 E5 gallery-necessity choices before spatial GT."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from final_selector import select_candidate
from frozen_io import load_split_rows_without_annotations, sha256_file
from g4_ablation import ALL_SOURCES, candidate_filter
from g4_e5_exact import (
    concatenate_payloads,
    normalized_payload,
    project_payload_masks_to_grid,
    verify_post_dedup_reproduction,
)
from pseudo.candidate_diagnostics import (
    save_candidate_diagnostics,
    validate_candidate_diagnostics_manifest,
    write_candidate_diagnostics_manifest,
)


ARMS = (
    "E5_exact__upstream_top1",
    "E5_exact__single_prompt_single_mask",
    "E5_exact__single_prompt_multimask",
    "E5_exact__full_pre_dedup",
    "E5_exact__full_post_dedup",
    "E5_exact__cap243",
)
BASELINE_ARM = "E5_exact__cap243"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    for label in ("pre-dedup", "post-dedup", "single-anchor", "single-addition"):
        parser.add_argument(f"--{label}-root", type=Path, required=True)
        parser.add_argument(f"--expected-{label}-manifest-sha256", required=True)
        parser.add_argument(f"--expected-{label}-pseudo-sha256", required=True)
    parser.add_argument("--expected-pre-dedup-contract-sha256", required=True)
    parser.add_argument("--pre-g1-root", type=Path, required=True)
    parser.add_argument("--expected-pre-g1-freeze-sha256", required=True)
    parser.add_argument("--post-g1-root", type=Path, required=True)
    parser.add_argument("--expected-post-g1-freeze-sha256", required=True)
    parser.add_argument("--baseline-choice-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-choice-freeze-sha256", required=True)
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


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {field: np.asarray(payload[field]) for field in payload.files}


def _validate_candidates(
    root: Path,
    *,
    image_ids: list[str],
    manifest_sha256: str,
    pseudo_sha256: str,
) -> dict[str, dict[str, str]]:
    rows, _summary = validate_candidate_diagnostics_manifest(
        root,
        expected_image_names=image_ids,
        split="val",
        expected_pseudo_manifest_sha256=pseudo_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    return rows


def _load_g1(
    root: Path,
    *,
    expected_freeze_sha256: str,
    split_sha256: str,
    image_ids: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    freeze_path = root / "diagnostic_freeze.json"
    if sha256_file(freeze_path) != expected_freeze_sha256:
        raise ValueError("G1 diagnostic freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_g1_all_candidate_score_freeze_v1"
        or freeze.get("split_sha256") != split_sha256
        or int(freeze.get("images", freeze.get("validation_images", -1))) != 371
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1 diagnostic freeze violates the E5 boundary")
    manifest = root / "descriptor_evidence_manifest.csv"
    if sha256_file(manifest) != freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("G1 evidence manifest changed")
    rows = _read_csv(manifest)
    indexed = {row["image_id"]: row for row in rows}
    if len(indexed) != 371 or set(indexed) != image_ids:
        raise ValueError("G1 evidence cohort differs from canonical validation")
    return indexed, freeze


def _g1_payload(
    root: Path,
    row: dict[str, str],
    *,
    expected_candidate_sha256: str,
) -> dict[str, np.ndarray]:
    path = root / "descriptor_evidence" / row["evidence_path"]
    if sha256_file(path) != row["evidence_sha256"]:
        raise ValueError(f"G1 evidence payload changed: {row['image_id']}")
    if row["candidate_payload_sha256"] != expected_candidate_sha256:
        raise ValueError(f"G1/candidate payload mismatch: {row['image_id']}")
    return _load(path)


def _candidate_path(root: Path, row: dict[str, str]) -> Path:
    path = root / row["diagnostic_path"]
    if sha256_file(path) != row["diagnostic_sha256"]:
        raise ValueError(f"candidate payload changed: {path.name}")
    return path


def _points(*values: np.ndarray, columns: int) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.int32).reshape(-1, columns) for value in values]
    return np.concatenate(arrays, axis=0)


def _validate_g1_alignment(
    candidate_indices: np.ndarray,
    g1_logits: np.ndarray,
    g1_upstream: np.ndarray,
    full_upstream: np.ndarray,
    *,
    image_id: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the sparse G1 bag against its full frozen candidate bank.

    G1 deliberately drops masks without sufficient token-grid support.  Its
    evidence arrays are therefore indexed by ``candidate_indices`` and must
    never be treated as dense candidate-order arrays.
    """

    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    logits = np.asarray(g1_logits, dtype=np.float64).reshape(-1)
    upstream = np.asarray(g1_upstream, dtype=np.float64).reshape(-1)
    full = np.asarray(full_upstream, dtype=np.float32).reshape(-1)
    if not (len(indices) == len(logits) == len(upstream)) or not len(indices):
        raise ValueError(f"{label} G1 arrays differ or are empty: {image_id}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{label} G1 candidate indices repeat: {image_id}")
    if np.any(indices < 0) or np.any(indices >= len(full)):
        raise ValueError(f"{label} G1 candidate indices are out of range: {image_id}")
    if not np.array_equal(upstream.astype(np.float32), full[indices]):
        raise ValueError(f"{label} G1 upstream scores differ: {image_id}")
    return indices, logits, upstream


def _select_indexed(
    candidate_indices: np.ndarray,
    g1_logits: np.ndarray,
    upstream: np.ndarray,
    eligible_indices: np.ndarray,
) -> tuple[int, float, float, float]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    eligible = set(np.asarray(eligible_indices, dtype=np.int64).reshape(-1).tolist())
    local_indices = np.asarray(
        [index for index, candidate in enumerate(candidate_indices) if int(candidate) in eligible],
        dtype=np.int64,
    )
    if not len(local_indices):
        raise ValueError("selector has no G1-scoreable eligible candidate")
    local, fused = select_candidate(g1_logits[local_indices], upstream[local_indices])
    evidence_local = int(local_indices[local])
    selected = int(candidate_indices[evidence_local])
    return (
        selected,
        float(g1_logits[evidence_local]),
        float(upstream[evidence_local]),
        float(fused[local]),
    )


def _optional_g1_value(
    candidate_indices: np.ndarray,
    g1_logits: np.ndarray,
    candidate_index: int,
) -> float | str:
    locations = np.flatnonzero(np.asarray(candidate_indices, dtype=np.int64) == candidate_index)
    if not len(locations):
        return ""
    return float(np.asarray(g1_logits, dtype=np.float64)[int(locations[0])])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gallery_root = args.output_dir / "candidate_gallery"
    choice_root = args.output_dir / "choices"
    gallery_root.mkdir()
    choice_root.mkdir()
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    image_ids = [str(row["image_id"]) for row in split_rows]
    image_id_set = set(image_ids)
    if len(image_ids) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("G4 E5 requires the canonical 371/184 validation cohort")

    pre_contract_path = args.pre_dedup_root / "g4_e5_pre_dedup_contract.json"
    if sha256_file(pre_contract_path) != args.expected_pre_dedup_contract_sha256:
        raise ValueError("pre-dedup build contract changed")
    pre_contract = json.loads(pre_contract_path.read_text(encoding="utf-8"))
    if (
        pre_contract.get("stage") != "g4_e5_exact_pre_dedup_gallery_v1"
        or pre_contract.get("split_sha256") != args.expected_split_sha256
        or pre_contract.get("post_dedup_reproduced_exactly") is not True
        or pre_contract.get("validation_gt_read") is not False
        or pre_contract.get("test_images_read") != 0
    ):
        raise ValueError("pre-dedup build contract violates the E5 boundary")

    candidate_specs = {
        "pre": (
            args.pre_dedup_root,
            args.expected_pre_dedup_manifest_sha256,
            args.expected_pre_dedup_pseudo_sha256,
        ),
        "post": (
            args.post_dedup_root,
            args.expected_post_dedup_manifest_sha256,
            args.expected_post_dedup_pseudo_sha256,
        ),
        "single_anchor": (
            args.single_anchor_root,
            args.expected_single_anchor_manifest_sha256,
            args.expected_single_anchor_pseudo_sha256,
        ),
        "single_addition": (
            args.single_addition_root,
            args.expected_single_addition_manifest_sha256,
            args.expected_single_addition_pseudo_sha256,
        ),
    }
    candidates = {
        label: _validate_candidates(
            root,
            image_ids=image_ids,
            manifest_sha256=manifest,
            pseudo_sha256=pseudo,
        )
        for label, (root, manifest, pseudo) in candidate_specs.items()
    }
    pre_g1_rows, pre_g1_freeze = _load_g1(
        args.pre_g1_root,
        expected_freeze_sha256=args.expected_pre_g1_freeze_sha256,
        split_sha256=args.expected_split_sha256,
        image_ids=image_id_set,
    )
    post_g1_rows, post_g1_freeze = _load_g1(
        args.post_g1_root,
        expected_freeze_sha256=args.expected_post_g1_freeze_sha256,
        split_sha256=args.expected_split_sha256,
        image_ids=image_id_set,
    )
    if pre_g1_freeze.get("baseline_checkpoint_sha256") != post_g1_freeze.get(
        "baseline_checkpoint_sha256"
    ):
        raise ValueError("pre/post G1 checkpoints differ")

    baseline_freeze_path = args.baseline_choice_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze_path) != args.expected_baseline_choice_freeze_sha256:
        raise ValueError("baseline choice freeze changed")
    baseline_freeze = json.loads(baseline_freeze_path.read_text(encoding="utf-8"))
    if (
        baseline_freeze.get("stage") != "final_rich_gallery_choice_freeze_v1"
        or baseline_freeze.get("cohort_split") != "val"
        or baseline_freeze.get("split_sha256") != args.expected_split_sha256
        or baseline_freeze.get("spatial_ground_truth_used") is not False
        or baseline_freeze.get("validation_gt_read") is not False
        or baseline_freeze.get("test_images_read") != 0
        or baseline_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("baseline choice freeze violates the E5 boundary")
    baseline_manifest = args.baseline_choice_root / "selection_manifest.csv"
    if sha256_file(baseline_manifest) != baseline_freeze["selection_manifest_sha256"]:
        raise ValueError("baseline selection manifest changed")
    baseline_rows = {row["image_id"]: row for row in _read_csv(baseline_manifest)}
    if len(baseline_rows) != 371 or set(baseline_rows) != image_id_set:
        raise ValueError("baseline selection cohort differs")

    gallery_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    prompt_matches = baseline_matches = 0
    for split_row in split_rows:
        image_id = str(split_row["image_id"])
        stem = Path(image_id).stem
        loaded = {
            label: _load(_candidate_path(candidate_specs[label][0], candidates[label][stem]))
            for label in candidate_specs
        }
        raw = normalized_payload(loaded["pre"])
        post = normalized_payload(loaded["post"])
        single_anchor = normalized_payload(loaded["single_anchor"])
        single_addition = normalized_payload(
            loaded["single_addition"], namespace="classifier448"
        )
        single_addition = project_payload_masks_to_grid(
            single_addition,
            tuple(int(value) for value in single_anchor["sam_masks"].shape[1:]),
        )
        single = concatenate_payloads(single_anchor, single_addition)
        if len(np.unique(single["prompt_ids"])) != len(single["prompt_ids"]):
            raise ValueError(f"single-mask prompt IDs are not unique: {image_id}")
        raw_first = verify_post_dedup_reproduction(raw, post)

        pre_evidence = _g1_payload(
            args.pre_g1_root,
            pre_g1_rows[image_id],
            expected_candidate_sha256=candidates["pre"][stem]["diagnostic_sha256"],
        )
        post_evidence = _g1_payload(
            args.post_g1_root,
            post_g1_rows[image_id],
            expected_candidate_sha256=candidates["post"][stem]["diagnostic_sha256"],
        )
        pre_indices, pre_logits, pre_upstream = _validate_g1_alignment(
            pre_evidence["candidate_indices"],
            pre_evidence["candidate_logits"],
            pre_evidence["selection_scores"],
            raw["selection_scores"],
            image_id=image_id,
            label="pre-dedup",
        )
        post_indices, post_logits, post_upstream = _validate_g1_alignment(
            post_evidence["candidate_indices"],
            post_evidence["candidate_logits"],
            post_evidence["selection_scores"],
            post["selection_scores"],
            image_id=image_id,
            label="post-dedup",
        )

        top_index = max(
            range(len(raw["sam_masks"])),
            key=lambda index: (float(pre_upstream[index]), -index),
        )
        exact_prompt_id = str(raw["prompt_ids"][top_index])
        multi_indices = np.flatnonzero(raw["prompt_ids"] == exact_prompt_id).astype(np.int64)
        single_local = np.flatnonzero(single["prompt_ids"] == exact_prompt_id).astype(np.int64)
        if len(multi_indices) != 3 or len(single_local) != 1:
            raise ValueError(f"exact prompt does not map 3-to-1: {image_id}/{exact_prompt_id}")
        if sorted(raw["multimask_indices"][multi_indices].tolist()) != [0, 1, 2]:
            raise ValueError(f"exact multimask IDs differ: {image_id}/{exact_prompt_id}")
        prompt_matches += 1

        pre_selected, pre_g1_value, pre_upstream_value, pre_fused = _select_indexed(
            pre_indices,
            pre_logits,
            pre_upstream,
            np.arange(len(raw["sam_masks"]), dtype=np.int64),
        )
        multi_selected, multi_g1_value, multi_upstream_value, multi_fused = _select_indexed(
            pre_indices, pre_logits, pre_upstream, multi_indices
        )
        post_selected_local, post_g1_value, post_upstream_value, post_fused = _select_indexed(
            post_indices,
            post_logits,
            post_upstream,
            np.arange(len(post["sam_masks"]), dtype=np.int64),
        )
        if post_selected_local != int(baseline_rows[image_id]["selected_candidate_index"]):
            raise ValueError(f"official baseline choice is not reproduced: {image_id}")
        baseline_matches += 1
        post_eligible_raw = raw_first.copy()
        post_selected_raw = int(raw_first[post_selected_local])
        cap_post = candidate_filter(
            post["proposal_source_ids"],
            post["selection_scores"],
            allowed_sources=ALL_SOURCES,
            per_source_cap=81,
        )
        cap_selected_local, cap_g1_value, cap_upstream_value, cap_fused = _select_indexed(
            post_indices, post_logits, post_upstream, cap_post
        )
        cap_eligible_raw = raw_first[cap_post]
        cap_selected_raw = int(raw_first[cap_selected_local])

        single_offset = len(raw["sam_masks"])
        unified = concatenate_payloads(raw, single)
        pre_payload = loaded["pre"]
        single_anchor_payload = loaded["single_anchor"]
        single_addition_payload = loaded["single_addition"]
        saved = save_candidate_diagnostics(
            gallery_root / "candidate_diagnostics" / f"{stem}.npz",
            sam_masks=unified["sam_masks"],
            refined_mask=np.asarray(pre_payload["refined_mask"], dtype=np.uint8),
            final_mask=np.asarray(pre_payload["final_mask"], dtype=np.uint8),
            bone_support=(
                np.asarray(pre_payload["bone_support"], dtype=np.uint8)
                if int(np.asarray(pre_payload["bone_support_present"]).reshape(-1)[0])
                else None
            ),
            prompt_map=np.asarray(pre_payload["prompt_map"], dtype=np.float32),
            positive_points=_points(
                pre_payload["positive_points"],
                single_anchor_payload["positive_points"],
                single_addition_payload["positive_points"],
                columns=2,
            ),
            negative_points=_points(
                pre_payload["negative_points"],
                single_anchor_payload["negative_points"],
                single_addition_payload["negative_points"],
                columns=2,
            ),
            boxes=_points(
                pre_payload["boxes"],
                single_anchor_payload["boxes"],
                single_addition_payload["boxes"],
                columns=4,
            ),
            sam_scores=unified["sam_scores"],
            selection_scores=unified["selection_scores"],
            classifier_causal_scores=unified["classifier_causal_scores"],
            component_ids=unified["component_ids"],
            prompt_modes=unified["prompt_modes"],
            proposal_source_ids=unified["proposal_source_ids"],
            cam_levels=unified["cam_levels"],
            prompt_ids=unified["prompt_ids"],
            multimask_indices=unified["multimask_indices"],
        )
        gallery_rows.append(
            {
                "image_name": image_id,
                "tumor_type": "tumor" if int(split_row["tumor"]) else "normal",
                "generation_status": "exact_e5_unified_gallery",
                **saved,
            }
        )

        selections = {
            "E5_exact__upstream_top1": (
                top_index,
                np.asarray([top_index], dtype=np.int64),
                _optional_g1_value(pre_indices, pre_logits, top_index),
                float(raw["selection_scores"][top_index]),
                float(raw["selection_scores"][top_index]),
            ),
            "E5_exact__single_prompt_single_mask": (
                single_offset + int(single_local[0]),
                single_offset + single_local,
                "",
                float(single["selection_scores"][single_local[0]]),
                float(single["selection_scores"][single_local[0]]),
            ),
            "E5_exact__single_prompt_multimask": (
                multi_selected,
                multi_indices,
                multi_g1_value,
                multi_upstream_value,
                multi_fused,
            ),
            "E5_exact__full_pre_dedup": (
                pre_selected,
                np.arange(len(raw["sam_masks"]), dtype=np.int64),
                pre_g1_value,
                pre_upstream_value,
                pre_fused,
            ),
            "E5_exact__full_post_dedup": (
                post_selected_raw,
                post_eligible_raw,
                post_g1_value,
                post_upstream_value,
                post_fused,
            ),
            "E5_exact__cap243": (
                cap_selected_raw,
                cap_eligible_raw,
                cap_g1_value,
                cap_upstream_value,
                cap_fused,
            ),
        }
        for arm in ARMS:
            selected, eligible, g1_value, upstream_value, fused_value = selections[arm]
            eligible = np.asarray(eligible, dtype=np.int64)
            if int(selected) not in set(eligible.tolist()):
                raise ValueError(f"selected candidate is ineligible: {image_id}/{arm}")
            choice_rows.append(
                {
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "tumor": split_row["tumor"],
                    "arm": arm,
                    "candidate_payload_sha256": saved["diagnostic_sha256"],
                    "selected_candidate_index": int(selected),
                    "eligible_candidate_count": len(eligible),
                    "eligible_candidate_indices": ";".join(str(int(value)) for value in eligible),
                    "selected_g1_logit": g1_value,
                    "selected_upstream_score": upstream_value,
                    "selected_fused_score": fused_value,
                    "exact_prompt_id": exact_prompt_id,
                    "raw_candidate_count": len(raw["sam_masks"]),
                    "single_mask_candidate_count": len(single["sam_masks"]),
                    "post_dedup_candidate_count": len(post["sam_masks"]),
                }
            )

    gallery_summary = write_candidate_diagnostics_manifest(
        gallery_root,
        gallery_rows,
        expected_image_names=image_ids,
        split="val",
        image_size=320,
        pseudo_manifest_sha256=args.expected_pre_dedup_pseudo_sha256,
        selection_method="G4 E5 exact gallery-necessity arms",
        support_clip_kernel=0,
        cam_percentile=90.0,
        cohort="all",
    )
    choices_path = choice_root / "g4_choices.csv"
    choices_sha = _write_csv(choices_path, choice_rows)
    freeze = {
        "schema_version": 1,
        "stage": "g4_e5_exact_choice_freeze_v1",
        "study": "G4 E5 exact gallery richness, deduplication, and cap necessity",
        "split_sha256": args.expected_split_sha256,
        "images": 371,
        "tumor_images": 184,
        "arms": list(ARMS),
        "baseline_arm": BASELINE_ARM,
        "choices_sha256": choices_sha,
        "selection_rows": len(choice_rows),
        "candidate_manifest_sha256": gallery_summary["manifest_sha256"],
        "candidate_summary_sha256": gallery_summary["summary_sha256"],
        "pseudo_manifest_sha256": args.expected_pre_dedup_pseudo_sha256,
        "pre_dedup_contract_sha256": args.expected_pre_dedup_contract_sha256,
        "pre_g1_freeze_sha256": args.expected_pre_g1_freeze_sha256,
        "post_g1_freeze_sha256": args.expected_post_g1_freeze_sha256,
        "baseline_choice_freeze_sha256": args.expected_baseline_choice_freeze_sha256,
        "g1_checkpoint_sha256": pre_g1_freeze["baseline_checkpoint_sha256"],
        "prompt_matches": prompt_matches,
        "baseline_exact_matches": baseline_matches,
        "candidate_choices_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = choice_root / "g4_choice_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {**freeze, "freeze_sha256": sha256_file(freeze_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
