from __future__ import annotations

"""Freeze source-correct G4 E7 upstream choices without spatial annotations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from final_selector import stable_select
from frozen_io import load_split_rows_without_annotations, sha256_file
from g4_ablation import (
    ALL_SOURCES,
    UPSTREAM_ARMS,
    fusion_score,
    select_score_only,
    source_correct_upstream_components_by_source,
    upstream_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--expected-g1-freeze-sha256", required=True)
    parser.add_argument("--addition-root", type=Path, required=True)
    parser.add_argument("--external-saliency-root", type=Path, required=True)
    parser.add_argument("--expected-external-manifest-sha256", required=True)
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


def resize_float_map_bilinear(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Deterministically resize a normalized map, without spatial labels."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("saliency map must be one finite HxW array")
    if values.shape == shape:
        result = values.copy()
    else:
        height, width = shape
        result = np.asarray(
            Image.fromarray(values, mode="F").resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
    if float(result.min()) < -1e-6 or float(result.max()) > 1.0 + 1e-6:
        raise ValueError("saliency map must remain normalized to [0,1]")
    return np.clip(result, 0.0, 1.0)


def _select_fused(
    g1: np.ndarray, upstream: np.ndarray, candidate_indices: np.ndarray
) -> int:
    fused = fusion_score(g1, upstream, "R7")
    local = stable_select(fused, g1)
    return int(candidate_indices[local])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    evidence_out = args.output_dir / "component_evidence"
    evidence_out.mkdir()

    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
        allow_test=False,
    )
    split_by_id = {str(row["image_id"]): row for row in split_rows}
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("E7 requires the canonical 371/184 validation cohort")

    candidate_manifest_path = args.candidate_root / "candidate_diagnostics_manifest.csv"
    if sha256_file(candidate_manifest_path) != args.expected_candidate_manifest_sha256:
        raise ValueError("candidate manifest SHA-256 mismatch")
    candidates = {row["image_name"]: row for row in _read_csv(candidate_manifest_path)}
    if set(candidates) != set(split_by_id):
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
        raise ValueError("G1 freeze violates annotation-free validation")
    g1_manifest_path = args.g1_root / "descriptor_evidence_manifest.csv"
    if sha256_file(g1_manifest_path) != g1_freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("G1 evidence manifest changed after freezing")
    g1_rows = {row["image_id"]: row for row in _read_csv(g1_manifest_path)}

    external_manifest_path = args.external_saliency_root / "saliency_manifest.csv"
    if sha256_file(external_manifest_path) != args.expected_external_manifest_sha256:
        raise ValueError("external saliency manifest SHA-256 mismatch")
    external_rows = {row["image_id"]: row for row in _read_csv(external_manifest_path)}
    if set(g1_rows) != set(split_by_id) or set(external_rows) != set(split_by_id):
        raise ValueError("G1/external cohorts differ from canonical validation")

    arms = ["legacy_U5", "legacy_U5_R7"]
    arms += [f"source_correct_{arm}" for arm in UPSTREAM_ARMS]
    arms += [f"source_correct_{arm}_R7" for arm in UPSTREAM_ARMS]
    choice_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    legacy_matches = 0

    for ordinal, split_row in enumerate(split_rows):
        image_id = str(split_row["image_id"])
        candidate_row = candidates[image_id]
        candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        g1_row = g1_rows[image_id]
        g1_path = args.g1_root / "descriptor_evidence" / g1_row["evidence_path"]
        if sha256_file(g1_path) != g1_row["evidence_sha256"]:
            raise ValueError(f"G1 evidence changed: {image_id}")

        with np.load(candidate_path, allow_pickle=False) as candidate, np.load(
            g1_path, allow_pickle=False
        ) as g1_payload:
            indices = g1_payload["candidate_indices"].astype(np.int64)
            g1 = g1_payload["candidate_logits"].astype(np.float64)
            legacy = g1_payload["selection_scores"].astype(np.float64)
            masks = candidate["sam_masks"][indices].astype(bool)
            sources = candidate["proposal_source_ids"][indices].astype(str)
            sam = candidate["sam_scores"][indices].astype(np.float64)
            components = candidate["component_ids"][indices]
            all_sources = candidate["proposal_source_ids"].astype(str)
            all_prompt_modes = candidate["prompt_modes"].astype(str)
            all_sam_scores = candidate["sam_scores"].astype(np.float64)
            if not (
                len(indices)
                == len(g1)
                == len(legacy)
                == len(masks)
                == len(sources)
                == len(sam)
                == len(components)
            ):
                raise ValueError(f"candidate/G1 arrays differ: {image_id}")
            shape = tuple(int(value) for value in masks.shape[1:])

            if int(split_row["tumor"]) == 1:
                if set(sources) != set(ALL_SOURCES):
                    raise ValueError(f"tumor source set differs: {image_id}: {set(sources)}")
                layercam_map = resize_float_map_bilinear(candidate["prompt_map"], shape)
                addition_path = args.addition_root / f"{Path(image_id).stem}.npz"
                expected_addition_sha = candidate_row["addition_diagnostic_sha256"]
                if sha256_file(addition_path) != expected_addition_sha:
                    raise ValueError(f"addition payload changed: {image_id}")
                with np.load(addition_path, allow_pickle=False) as addition:
                    classifier_map = resize_float_map_bilinear(
                        addition["prompt_map"], shape
                    )
                external_row = external_rows[image_id]
                external_path = args.external_saliency_root / external_row["map_path"]
                if sha256_file(external_path) != external_row["map_sha256"]:
                    raise ValueError(f"external map changed: {image_id}")
                external_map = resize_float_map_bilinear(np.load(external_path), shape)
                corrected = source_correct_upstream_components_by_source(
                    masks,
                    sources,
                    {
                        "layercam": layercam_map,
                        "classifier448:layercam": classifier_map,
                        "external_saliency": external_map,
                    },
                    sam,
                    components,
                )
            else:
                # Normal images have a single frozen empty fallback mask. They
                # never enter the tumor-Dice endpoint, but remain in the freeze
                # to verify the complete 371-image inference contract.
                corrected = None

            score_by_arm: dict[str, np.ndarray] = {
                "legacy_U5": legacy,
                "legacy_U5_R7": legacy,
            }
            if corrected is not None:
                for arm in UPSTREAM_ARMS:
                    score_by_arm[f"source_correct_{arm}"] = upstream_score(corrected, arm)
                    score_by_arm[f"source_correct_{arm}_R7"] = score_by_arm[
                        f"source_correct_{arm}"
                    ]
            else:
                for arm in UPSTREAM_ARMS:
                    score_by_arm[f"source_correct_{arm}"] = legacy
                    score_by_arm[f"source_correct_{arm}_R7"] = legacy

            selections: dict[str, int] = {}
            for arm in arms:
                score = score_by_arm[arm]
                if arm.endswith("_R7"):
                    selections[arm] = _select_fused(g1, score, indices)
                else:
                    selections[arm] = select_score_only(score, indices)
            if selections["legacy_U5_R7"] == int(g1_row["selected_candidate_index"]):
                legacy_matches += 1

            evidence_path = evidence_out / f"{ordinal:04d}_{Path(image_id).stem}.npz"
            arrays: dict[str, np.ndarray] = {
                "candidate_indices": indices.astype(np.int32),
                "g1_logits": g1.astype(np.float32),
                "legacy_U5": legacy.astype(np.float32),
                "sources": sources.astype("U96"),
            }
            if corrected is not None:
                arrays.update(
                    {
                        "sam_score": corrected.sam_score.astype(np.float32),
                        "cam_density": corrected.cam_density.astype(np.float32),
                        "cam_mass_coverage": corrected.cam_mass_coverage.astype(np.float32),
                        "sam_component_rank": corrected.sam_component_rank.astype(np.float32),
                        "sam_global_rank": corrected.sam_global_rank.astype(np.float32),
                    }
                )
            np.savez_compressed(evidence_path, **arrays)

        evidence_rows.append(
            {
                "image_id": image_id,
                "tumor": split_row["tumor"],
                "evidence_path": str(evidence_path.relative_to(args.output_dir)).replace("\\", "/"),
                "evidence_sha256": sha256_file(evidence_path),
            }
        )
        for arm in arms:
            selected = selections[arm]
            choice_rows.append(
                {
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "tumor": split_row["tumor"],
                    "arm": f"E7__{arm}",
                    "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                    "gallery_candidate_count": candidate_row["candidate_count"],
                    "g1_eligible_candidate_count": len(indices),
                    "eligible_candidate_count": len(indices),
                    "eligible_candidate_indices": ";".join(str(int(index)) for index in indices),
                    "selected_candidate_index": selected,
                    "selected_source": str(all_sources[selected]),
                    "selected_prompt_mode": str(all_prompt_modes[selected]),
                    "selected_sam_score": float(all_sam_scores[selected]),
                    "selected_upstream_score": float(score_by_arm[arm][np.flatnonzero(indices == selected)[0]]),
                    "selected_g1_logit": float(g1[np.flatnonzero(indices == selected)[0]]),
                }
            )

    if legacy_matches != 371:
        raise ValueError(f"legacy R7 reproduced {legacy_matches}/371 choices")
    choice_sha = _write_csv(args.output_dir / "g4_choices.csv", choice_rows)
    evidence_sha = _write_csv(
        args.output_dir / "component_evidence_manifest.csv", evidence_rows
    )
    freeze = {
        "schema_version": 1,
        "stage": "g4_e7_source_correct_upstream_choice_freeze_v1",
        "cohort_split": "val",
        "split_sha256": args.expected_split_sha256,
        "candidate_manifest_sha256": args.expected_candidate_manifest_sha256,
        "g1_freeze_sha256": args.expected_g1_freeze_sha256,
        "external_saliency_manifest_sha256": args.expected_external_manifest_sha256,
        "choices_sha256": choice_sha,
        "component_evidence_manifest_sha256": evidence_sha,
        "images": 371,
        "tumor_images": 184,
        "arms": [f"E7__{arm}" for arm in arms],
        "baseline_arm": "E7__legacy_U5_R7",
        "study": "G4 E7 source-correct upstream component and coefficient ablation",
        "selection_rows": len(choice_rows),
        "legacy_r7_exact_matches": legacy_matches,
        "source_map_contract": {
            "layercam": "merged anchor prompt_map on 320 grid",
            "classifier448:layercam": "addition prompt_map bilinear-resized to merged grid",
            "external_saliency": "frozen BiomedCLIP saliency map on merged grid",
            "component_rank_group": "proposal_source_id plus component_id",
        },
        "candidate_choices_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "g4_choice_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
