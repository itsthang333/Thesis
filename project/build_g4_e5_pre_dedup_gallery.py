from __future__ import annotations

"""Build the exact pre-dedup G4 E5 gallery without opening spatial GT."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from frozen_io import load_split_rows_without_annotations, sha256_file
from g4_e5_exact import (
    attach_exact_multimask_provenance,
    concatenate_payloads,
    normalized_payload,
    verify_post_dedup_reproduction,
)
from pseudo.candidate_diagnostics import (
    save_candidate_diagnostics,
    validate_candidate_diagnostics_manifest,
    write_candidate_diagnostics_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    for label in ("multimask-anchor", "multimask-addition", "single-anchor", "single-addition", "post-dedup"):
        parser.add_argument(f"--{label}-root", type=Path, required=True)
        parser.add_argument(f"--expected-{label}-manifest-sha256", required=True)
        parser.add_argument(f"--expected-{label}-pseudo-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {field: np.asarray(payload[field]) for field in payload.files}


def _candidate_rows(
    root: Path,
    *,
    image_ids: list[str],
    expected_manifest: str,
    expected_pseudo: str,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    return validate_candidate_diagnostics_manifest(
        root,
        expected_image_names=image_ids,
        split="val",
        expected_pseudo_manifest_sha256=expected_pseudo,
        expected_manifest_sha256=expected_manifest,
    )


def _payload_path(root: Path, row: dict[str, str]) -> Path:
    path = root / row["diagnostic_path"]
    if sha256_file(path) != row["diagnostic_sha256"]:
        raise ValueError(f"candidate payload changed: {path.name}")
    return path


def _points(*values: np.ndarray, columns: int) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.int32).reshape(-1, columns) for value in values]
    return np.concatenate(arrays, axis=0) if arrays else np.empty((0, columns), dtype=np.int32)


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
        split="val",
        allow_test=False,
    )
    image_ids = [str(row["image_id"]) for row in split_rows]
    if len(image_ids) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("G4 E5 requires the canonical 371/184 validation cohort")

    specifications = {
        "multimask_anchor": (
            args.multimask_anchor_root,
            args.expected_multimask_anchor_manifest_sha256,
            args.expected_multimask_anchor_pseudo_sha256,
        ),
        "multimask_addition": (
            args.multimask_addition_root,
            args.expected_multimask_addition_manifest_sha256,
            args.expected_multimask_addition_pseudo_sha256,
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
        "post_dedup": (
            args.post_dedup_root,
            args.expected_post_dedup_manifest_sha256,
            args.expected_post_dedup_pseudo_sha256,
        ),
    }
    manifests: dict[str, tuple[dict[str, dict[str, str]], dict[str, object]]] = {}
    for label, (root, expected_manifest, expected_pseudo) in specifications.items():
        manifests[label] = _candidate_rows(
            root,
            image_ids=image_ids,
            expected_manifest=expected_manifest,
            expected_pseudo=expected_pseudo,
        )

    output_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    total_raw = total_post = total_prompts = 0
    raw_counts: list[int] = []
    post_counts: list[int] = []
    for split_row in split_rows:
        image_id = str(split_row["image_id"])
        stem = Path(image_id).stem
        loaded: dict[str, dict[str, np.ndarray]] = {}
        for label, (root, _manifest, _pseudo) in specifications.items():
            row = manifests[label][0][stem]
            loaded[label] = _load(_payload_path(root, row))

        anchor_multi = attach_exact_multimask_provenance(
            loaded["multimask_anchor"], loaded["single_anchor"]
        )
        addition_multi = attach_exact_multimask_provenance(
            normalized_payload(loaded["multimask_addition"], namespace="classifier448"),
            normalized_payload(loaded["single_addition"], namespace="classifier448"),
        )
        raw = concatenate_payloads(anchor_multi, addition_multi)
        post = normalized_payload(loaded["post_dedup"])
        kept = verify_post_dedup_reproduction(raw, post)
        if len(raw["sam_masks"]) > 243:
            raise ValueError(f"raw gallery exceeds theoretical cap for {image_id}")
        if len(np.unique(raw["prompt_ids"])) * 3 != len(raw["sam_masks"]):
            raise ValueError(f"multimask prompt cardinality differs for {image_id}")
        multimask_values = np.asarray(raw["multimask_indices"], dtype=np.int16)
        for prompt_id in np.unique(raw["prompt_ids"]):
            indices = np.flatnonzero(raw["prompt_ids"] == prompt_id)
            if sorted(multimask_values[indices].tolist()) != [0, 1, 2]:
                raise ValueError(f"multimask indices differ for {image_id}/{prompt_id}")

        anchor_original = loaded["multimask_anchor"]
        addition_original = loaded["multimask_addition"]
        saved = save_candidate_diagnostics(
            args.output_dir / "candidate_diagnostics" / f"{stem}.npz",
            sam_masks=raw["sam_masks"],
            refined_mask=np.asarray(anchor_original["refined_mask"], dtype=np.uint8),
            final_mask=np.asarray(anchor_original["final_mask"], dtype=np.uint8),
            bone_support=(
                np.asarray(anchor_original["bone_support"], dtype=np.uint8)
                if int(np.asarray(anchor_original["bone_support_present"]).reshape(-1)[0])
                else None
            ),
            prompt_map=np.asarray(anchor_original["prompt_map"], dtype=np.float32),
            positive_points=_points(
                anchor_original["positive_points"], addition_original["positive_points"], columns=2
            ),
            negative_points=_points(
                anchor_original["negative_points"], addition_original["negative_points"], columns=2
            ),
            boxes=_points(anchor_original["boxes"], addition_original["boxes"], columns=4),
            sam_scores=raw["sam_scores"],
            selection_scores=raw["selection_scores"],
            classifier_causal_scores=raw["classifier_causal_scores"],
            component_ids=raw["component_ids"],
            prompt_modes=raw["prompt_modes"],
            proposal_source_ids=raw["proposal_source_ids"],
            cam_levels=raw["cam_levels"],
            prompt_ids=raw["prompt_ids"],
            multimask_indices=raw["multimask_indices"],
        )
        output_rows.append(
            {
                "image_name": image_id,
                "tumor_type": "tumor" if int(split_row["tumor"]) else "normal",
                "generation_status": "exact_pre_dedup_replay",
                **saved,
            }
        )
        raw_count = len(raw["sam_masks"])
        post_count = len(post["sam_masks"])
        prompt_count = len(np.unique(raw["prompt_ids"]))
        audit_rows.append(
            {
                "image_id": image_id,
                "raw_candidate_count": raw_count,
                "exact_prompt_count": prompt_count,
                "post_dedup_candidate_count": post_count,
                "duplicate_candidate_count": raw_count - post_count,
                "post_raw_first_indices": ";".join(str(int(value)) for value in kept),
                "raw_payload_sha256": saved["diagnostic_sha256"],
                "post_payload_sha256": manifests["post_dedup"][0][stem]["diagnostic_sha256"],
            }
        )
        raw_counts.append(raw_count)
        post_counts.append(post_count)
        total_raw += raw_count
        total_post += post_count
        total_prompts += prompt_count

    output_summary = write_candidate_diagnostics_manifest(
        args.output_dir,
        output_rows,
        expected_image_names=image_ids,
        split="val",
        image_size=320,
        pseudo_manifest_sha256=args.expected_multimask_anchor_pseudo_sha256,
        selection_method="G4 E5 exact pre-dedup replay",
        support_clip_kernel=0,
        cam_percentile=90.0,
        cohort="all",
    )
    replay_path = args.output_dir / "pre_dedup_replay.csv"
    replay_sha = _write_csv(replay_path, audit_rows)
    contract = {
        "schema_version": 1,
        "stage": "g4_e5_exact_pre_dedup_gallery_v1",
        "split_sha256": args.expected_split_sha256,
        "images": 371,
        "tumor_images": 184,
        "raw_candidates": total_raw,
        "post_dedup_candidates": total_post,
        "duplicates_removed": total_raw - total_post,
        "exact_prompts": total_prompts,
        "raw_candidate_count_min": min(raw_counts),
        "raw_candidate_count_max": max(raw_counts),
        "post_candidate_count_min": min(post_counts),
        "post_candidate_count_max": max(post_counts),
        "output_manifest_sha256": output_summary["manifest_sha256"],
        "output_summary_sha256": output_summary["summary_sha256"],
        "output_pseudo_manifest_sha256": args.expected_multimask_anchor_pseudo_sha256,
        "pre_dedup_replay_sha256": replay_sha,
        "input_manifest_sha256": {
            label: expected_manifest
            for label, (_root, expected_manifest, _pseudo) in specifications.items()
        },
        "input_pseudo_manifest_sha256": {
            label: expected_pseudo
            for label, (_root, _manifest, expected_pseudo) in specifications.items()
        },
        "post_dedup_reproduced_exactly": True,
        "prompt_group_contract": "one exact single-mask candidate and three multimask candidates",
        "candidate_cap_theoretical": 243,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    contract_path = args.output_dir / "g4_e5_pre_dedup_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**contract, "contract_sha256": sha256_file(contract_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
