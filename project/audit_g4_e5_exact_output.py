from __future__ import annotations

"""Independent integrity audit for the exact G4 E5 output bundle."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from frozen_io import load_split_rows_without_annotations, sha256_file
from freeze_g4_e5_exact_choices import ARMS, BASELINE_ARM
from g4_e5_exact import first_unique_mask_indices
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--choice-root", type=Path, required=True)
    parser.add_argument("--expected-choice-freeze-sha256", required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _indices(text: str) -> np.ndarray:
    return np.asarray([int(value) for value in text.split(";") if value], dtype=np.int64)


def main() -> None:
    args = parse_args()
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    image_ids = [str(row["image_id"]) for row in split_rows]
    image_id_set = set(image_ids)
    if len(image_ids) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("G4 E5 audit requires the canonical 371/184 validation cohort")

    freeze_path = args.choice_root / "g4_choice_freeze.json"
    if sha256_file(freeze_path) != args.expected_choice_freeze_sha256:
        raise ValueError("choice freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "g4_e5_exact_choice_freeze_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or tuple(freeze.get("arms", ())) != ARMS
        or freeze.get("baseline_arm") != BASELINE_ARM
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
        or int(freeze.get("prompt_matches", -1)) != 371
        or int(freeze.get("baseline_exact_matches", -1)) != 371
    ):
        raise ValueError("choice freeze violates the exact E5 contract")

    candidate_rows, candidate_summary = validate_candidate_diagnostics_manifest(
        args.candidate_root,
        expected_image_names=image_ids,
        split="val",
        expected_pseudo_manifest_sha256=freeze["pseudo_manifest_sha256"],
        expected_manifest_sha256=freeze["candidate_manifest_sha256"],
    )
    if candidate_summary["summary_sha256"] != freeze["candidate_summary_sha256"]:
        raise ValueError("candidate summary changed")
    choices_path = args.choice_root / "g4_choices.csv"
    if sha256_file(choices_path) != freeze["choices_sha256"]:
        raise ValueError("choice manifest changed")
    choices = _read_csv(choices_path)
    if len(choices) != 371 * len(ARMS):
        raise ValueError("choice matrix row count differs")
    by_image: dict[str, dict[str, dict[str, str]]] = {}
    for row in choices:
        by_image.setdefault(row["image_id"], {})[row["arm"]] = row
    if set(by_image) != image_id_set or any(set(rows) != set(ARMS) for rows in by_image.values()):
        raise ValueError("choice matrix cohort/arms differ")

    total_raw = total_single = total_post = total_cap = 0
    for image_id in image_ids:
        stem = Path(image_id).stem
        row = candidate_rows[stem]
        path = args.candidate_root / row["diagnostic_path"]
        if sha256_file(path) != row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["schema_version"][0]) != 3:
                raise ValueError(f"candidate provenance schema differs: {image_id}")
            masks = payload["sam_masks"].astype(np.uint8)
            sources = payload["proposal_source_ids"].astype(str)
            prompt_ids = payload["prompt_ids"].astype(str)
            multimask = payload["multimask_indices"].astype(np.int16)
            upstream = payload["selection_scores"].astype(np.float64)
        rows = by_image[image_id]
        raw_count = int(rows[ARMS[0]]["raw_candidate_count"])
        single_count = int(rows[ARMS[0]]["single_mask_candidate_count"])
        post_count = int(rows[ARMS[0]]["post_dedup_candidate_count"])
        if len(masks) != raw_count + single_count or raw_count > 243:
            raise ValueError(f"unified/raw candidate counts differ: {image_id}")
        if any(
            int(item["raw_candidate_count"]) != raw_count
            or int(item["single_mask_candidate_count"]) != single_count
            or int(item["post_dedup_candidate_count"]) != post_count
            for item in rows.values()
        ):
            raise ValueError(f"per-arm candidate counts differ: {image_id}")
        first_unique = first_unique_mask_indices(masks[:raw_count])
        if len(first_unique) != post_count:
            raise ValueError(f"post-dedup count differs: {image_id}")

        top = rows["E5_exact__upstream_top1"]
        top_eligible = _indices(top["eligible_candidate_indices"])
        top_index = max(range(raw_count), key=lambda index: (upstream[index], -index))
        if top_eligible.tolist() != [top_index] or int(top["selected_candidate_index"]) != top_index:
            raise ValueError(f"upstream top-1 is not exact: {image_id}")
        exact_prompt = top["exact_prompt_id"]
        if prompt_ids[top_index] != exact_prompt:
            raise ValueError(f"top-1 prompt provenance differs: {image_id}")

        single = rows["E5_exact__single_prompt_single_mask"]
        single_eligible = _indices(single["eligible_candidate_indices"])
        if (
            len(single_eligible) != 1
            or single_eligible[0] < raw_count
            or prompt_ids[single_eligible[0]] != exact_prompt
        ):
            raise ValueError(f"single-mask arm is not the same exact prompt: {image_id}")
        multi = rows["E5_exact__single_prompt_multimask"]
        multi_eligible = _indices(multi["eligible_candidate_indices"])
        if (
            len(multi_eligible) != 3
            or np.any(multi_eligible >= raw_count)
            or set(prompt_ids[multi_eligible]) != {exact_prompt}
            or sorted(multimask[multi_eligible].tolist()) != [0, 1, 2]
        ):
            raise ValueError(f"multimask arm is not the same exact prompt: {image_id}")

        pre = _indices(rows["E5_exact__full_pre_dedup"]["eligible_candidate_indices"])
        post = _indices(rows["E5_exact__full_post_dedup"]["eligible_candidate_indices"])
        cap = _indices(rows["E5_exact__cap243"]["eligible_candidate_indices"])
        if not np.array_equal(pre, np.arange(raw_count)):
            raise ValueError(f"pre-dedup eligibility differs: {image_id}")
        if not np.array_equal(post, first_unique):
            raise ValueError(f"post-dedup eligibility differs: {image_id}")
        if len(cap) > 243 or not set(cap.tolist()).issubset(set(post.tolist())):
            raise ValueError(f"cap-243 eligibility differs: {image_id}")
        for source in set(sources[cap].tolist()):
            if int(np.sum(sources[cap] == source)) > 81:
                raise ValueError(f"per-source cap exceeds 81: {image_id}/{source}")
        for arm, item in rows.items():
            eligible = _indices(item["eligible_candidate_indices"])
            if len(eligible) != int(item["eligible_candidate_count"]):
                raise ValueError(f"eligible count differs: {image_id}/{arm}")
            if int(item["selected_candidate_index"]) not in set(eligible.tolist()):
                raise ValueError(f"selected candidate is ineligible: {image_id}/{arm}")
            if item["candidate_payload_sha256"] != row["diagnostic_sha256"]:
                raise ValueError(f"choice/candidate hash differs: {image_id}/{arm}")
        total_raw += raw_count
        total_single += single_count
        total_post += post_count
        total_cap += len(cap)

    evaluation_audit_path = args.evaluation_root / "evaluation_audit.json"
    evaluation_summary_path = args.evaluation_root / "summary.json"
    evaluation_audit = json.loads(evaluation_audit_path.read_text(encoding="utf-8"))
    evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
    if (
        evaluation_audit.get("pass") is not True
        or evaluation_audit.get("choices_frozen_before_annotations") is not True
        or evaluation_audit.get("choice_freeze_sha256") != args.expected_choice_freeze_sha256
        or evaluation_audit.get("summary_sha256") != sha256_file(evaluation_summary_path)
        or int(evaluation_audit.get("images", -1)) != 371
        or int(evaluation_audit.get("tumor_images", -1)) != 184
        or int(evaluation_audit.get("arms", -1)) != len(ARMS)
        or int(evaluation_audit.get("validation_annotations_opened", -1)) != 184
        or evaluation_audit.get("test_images_read") != 0
        or evaluation_audit.get("test_evaluated") is not False
        or evaluation_summary.get("baseline_arm") != BASELINE_ARM
        or set(evaluation_summary.get("summaries", {})) != set(ARMS)
        or evaluation_summary.get("test_images_read") != 0
        or evaluation_summary.get("test_evaluated") is not False
    ):
        raise ValueError("spatial evaluation violates the exact E5 contract")

    result = {
        "schema_version": 1,
        "pass": True,
        "stage": "g4_e5_exact_output_audit_v1",
        "split_sha256": args.expected_split_sha256,
        "choice_freeze_sha256": args.expected_choice_freeze_sha256,
        "candidate_manifest_sha256": candidate_summary["manifest_sha256"],
        "candidate_summary_sha256": candidate_summary["summary_sha256"],
        "evaluation_audit_sha256": sha256_file(evaluation_audit_path),
        "evaluation_summary_sha256": sha256_file(evaluation_summary_path),
        "images": 371,
        "tumor_images": 184,
        "arms": len(ARMS),
        "raw_candidates": total_raw,
        "single_mask_candidates": total_single,
        "post_dedup_candidates": total_post,
        "cap243_candidates": total_cap,
        "same_prompt_single_vs_multimask_verified": True,
        "post_dedup_replay_verified": True,
        "per_source_cap81_verified": True,
        "choices_frozen_before_annotations": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "audit_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
