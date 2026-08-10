from __future__ import annotations

"""Unconditionally merge two frozen, GT-blind candidate galleries.

The anchor gallery supplies the prompt map and frozen output-mask diagnostics.
All additional candidates are evaluated against that same anchor prompt map by
the downstream geometry-v3 selector.  This prevents an inference-time source
router while preserving source identities for later family-balanced pooling.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from frozen_io import load_split_rows_without_annotations, sha256_file
from evaluation.frozen_test_guard import verify_frozen_test_config
from pseudo.candidate_diagnostics import (
    save_candidate_diagnostics,
    validate_candidate_diagnostics_manifest,
    write_candidate_diagnostics_manifest,
)


def _candidate_key(mask: np.ndarray) -> bytes:
    binary = np.asarray(mask) > 0
    return np.packbits(binary.reshape(-1)).tobytes()


def resize_binary_masks_nearest(
    masks: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Project a frozen gallery onto the anchor grid without using GT.

    The mapping is explicit ``floor(output_index * source / target)`` nearest
    sampling.  Keeping it here makes the 448->320 supply alignment immutable
    and independently testable rather than relying on a library default.
    """
    values = np.asarray(masks, dtype=np.uint8)
    if values.ndim != 3 or len(target_shape) != 2 or min(target_shape) <= 0:
        raise ValueError("Masks/target shape are invalid for nearest resizing")
    source_height, source_width = values.shape[1:]
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    if (source_height, source_width) == (target_height, target_width):
        return values.copy()
    y = np.floor(
        np.arange(target_height, dtype=np.float64) * source_height / target_height
    ).astype(np.int64)
    x = np.floor(
        np.arange(target_width, dtype=np.float64) * source_width / target_width
    ).astype(np.int64)
    y = np.clip(y, 0, source_height - 1)
    x = np.clip(x, 0, source_width - 1)
    return (values[:, y[:, None], x[None, :]] > 0).astype(np.uint8)


def merge_payloads(
    anchor: dict[str, np.ndarray],
    addition: dict[str, np.ndarray],
    *,
    addition_namespace: str,
    allow_missing_addition_provenance: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if not addition_namespace or ":" in addition_namespace:
        raise ValueError("addition_namespace must be a nonempty source prefix")
    anchor_masks = np.asarray(anchor["sam_masks"], dtype=np.uint8)
    addition_masks = np.asarray(addition["sam_masks"], dtype=np.uint8)
    if anchor_masks.ndim != 3 or addition_masks.ndim != 3:
        raise ValueError("Candidate galleries must contain [N,H,W] masks")
    if np.asarray(anchor["prompt_map"]).shape != anchor_masks.shape[1:]:
        raise ValueError("Anchor prompt map does not match candidate geometry")
    if np.asarray(addition["prompt_map"]).shape != addition_masks.shape[1:]:
        raise ValueError("Addition prompt map does not match candidate geometry")
    addition_input_shape = tuple(int(value) for value in addition_masks.shape[1:])
    anchor_shape = tuple(int(value) for value in anchor_masks.shape[1:])
    addition_masks = resize_binary_masks_nearest(addition_masks, anchor_shape)

    required = (
        "sam_scores",
        "selection_scores",
        "classifier_causal_scores",
        "component_ids",
        "prompt_modes",
        "proposal_source_ids",
    )
    for name, payload, count in (
        ("anchor", anchor, len(anchor_masks)),
        ("addition", addition, len(addition_masks)),
    ):
        for field in required:
            if len(np.asarray(payload[field]).reshape(-1)) != count:
                raise ValueError(f"{name} field {field} does not align with masks")
    provenance_fields = ("cam_levels", "prompt_ids", "multimask_indices")
    anchor_provenance_present = all(field in anchor for field in provenance_fields)
    addition_provenance_present = all(field in addition for field in provenance_fields)
    backfilled_addition_provenance = 0
    if anchor_provenance_present != addition_provenance_present:
        if not (
            allow_missing_addition_provenance
            and anchor_provenance_present
            and not addition_provenance_present
        ):
            raise ValueError("Anchor/addition exact provenance availability differs")
        addition = dict(addition)
        addition_count = len(addition_masks)
        addition["cam_levels"] = np.full(addition_count, np.nan, dtype=np.float32)
        addition["multimask_indices"] = np.full(
            addition_count, -1, dtype=np.int16
        )
        addition["prompt_ids"] = np.asarray(
            [
                "legacy_provenance_unavailable"
                f"|{str(np.asarray(addition['proposal_source_ids'])[index])}"
                f"|c{int(np.asarray(addition['component_ids'])[index])}"
                f"|{str(np.asarray(addition['prompt_modes'])[index])}"
                f"|i{index}"
                for index in range(addition_count)
            ],
            dtype="U192",
        )
        addition_provenance_present = True
        backfilled_addition_provenance = addition_count
    provenance_present = anchor_provenance_present and addition_provenance_present
    if provenance_present:
        for name, payload, count in (
            ("anchor", anchor, len(anchor_masks)),
            ("addition", addition, len(addition_masks)),
        ):
            for field in provenance_fields:
                if len(np.asarray(payload[field]).reshape(-1)) != count:
                    raise ValueError(f"{name} field {field} does not align with masks")

    seen: set[bytes] = set()
    anchor_keep: list[int] = []
    addition_keep: list[int] = []
    for index, mask in enumerate(anchor_masks):
        key = _candidate_key(mask)
        if key not in seen:
            seen.add(key)
            anchor_keep.append(index)
    for index, mask in enumerate(addition_masks):
        key = _candidate_key(mask)
        if key not in seen:
            seen.add(key)
            addition_keep.append(index)

    anchor_indices = np.asarray(anchor_keep, dtype=np.int64)
    addition_indices = np.asarray(addition_keep, dtype=np.int64)

    def source_features(payload: dict[str, np.ndarray], masks: np.ndarray) -> dict[str, np.ndarray]:
        count = len(masks)
        if "source_map_mean_scores" in payload:
            return {
                "source_map_mean_scores": np.asarray(payload["source_map_mean_scores"], dtype=np.float32),
                "source_map_mass_coverages": np.asarray(payload["source_map_mass_coverages"], dtype=np.float32),
                "source_score_densities": np.asarray(payload["source_score_densities"], dtype=np.float32),
                "source_class_ids": np.asarray(payload["source_class_ids"], dtype=np.int16),
                "source_class_probabilities": np.asarray(payload["source_class_probabilities"], dtype=np.float32),
                "source_class_ranks": np.asarray(payload["source_class_ranks"], dtype=np.int8),
            }
        prompt_map = np.asarray(payload["prompt_map"], dtype=np.float32)
        means = np.zeros(count, dtype=np.float32)
        mass = np.zeros(count, dtype=np.float32)
        density = np.zeros(count, dtype=np.float32)
        total_mass = max(float(prompt_map.sum()), 1.0e-8)
        for index, mask in enumerate(masks.astype(bool)):
            if not mask.any():
                continue
            values = prompt_map[mask]
            means[index] = float(values.mean())
            mass[index] = float(values.sum() / total_mass)
            density[index] = float((values > 0.5).mean())
        return {
            "source_map_mean_scores": means,
            "source_map_mass_coverages": mass,
            "source_score_densities": density,
            "source_class_ids": np.full(count, -4, dtype=np.int16),
            "source_class_probabilities": np.zeros(count, dtype=np.float32),
            "source_class_ranks": np.full(count, -1, dtype=np.int8),
        }

    anchor_features = source_features(anchor, anchor_masks)
    addition_features = source_features(addition, np.asarray(addition["sam_masks"], dtype=np.uint8))

    def joined(field: str, dtype: Any) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(anchor[field])[anchor_indices].astype(dtype),
                np.asarray(addition[field])[addition_indices].astype(dtype),
            ],
            axis=0,
        )

    addition_sources = np.asarray(addition["proposal_source_ids"], dtype="U64")
    addition_sources = np.asarray(
        [f"{addition_namespace}:{value}" for value in addition_sources], dtype="U96"
    )
    result = {
        "sam_masks": np.concatenate(
            [anchor_masks[anchor_indices], addition_masks[addition_indices]], axis=0
        ).astype(np.uint8),
        "sam_scores": joined("sam_scores", np.float32),
        "selection_scores": joined("selection_scores", np.float32),
        "classifier_causal_scores": joined(
            "classifier_causal_scores", np.float32
        ),
        "component_ids": joined("component_ids", np.int32),
        "prompt_modes": joined("prompt_modes", "U32"),
        "proposal_source_ids": np.concatenate(
            [
                np.asarray(anchor["proposal_source_ids"], dtype="U96")[anchor_indices],
                addition_sources[addition_indices],
            ]
        ),
        "prompt_map": np.asarray(anchor["prompt_map"], dtype=np.float32),
    }
    for feature_name, dtype in (
        ("source_map_mean_scores", np.float32),
        ("source_map_mass_coverages", np.float32),
        ("source_score_densities", np.float32),
        ("source_class_ids", np.int16),
        ("source_class_probabilities", np.float32),
        ("source_class_ranks", np.int8),
    ):
        result[feature_name] = np.concatenate(
            [
                np.asarray(anchor_features[feature_name])[anchor_indices].astype(dtype),
                np.asarray(addition_features[feature_name])[addition_indices].astype(dtype),
            ]
        )
    if "dsll_source_maps" in anchor:
        result["dsll_source_maps"] = np.asarray(anchor["dsll_source_maps"], dtype=np.float32)
        result["dsll_source_map_ids"] = np.asarray(anchor["dsll_source_map_ids"], dtype="U32")
    if provenance_present:
        addition_prompt_ids = np.asarray(addition["prompt_ids"], dtype="U192")
        addition_prompt_ids = np.asarray(
            [f"{addition_namespace}:{value}" for value in addition_prompt_ids],
            dtype="U192",
        )
        result.update(
            {
                "cam_levels": joined("cam_levels", np.float32),
                "prompt_ids": np.concatenate(
                    [
                        np.asarray(anchor["prompt_ids"], dtype="U192")[anchor_indices],
                        addition_prompt_ids[addition_indices],
                    ]
                ),
                "multimask_indices": joined("multimask_indices", np.int16),
            }
        )
    if len(result["sam_masks"]) == 0:
        raise ValueError("Merged gallery cannot be empty")
    return result, {
        "anchor_input": int(len(anchor_masks)),
        "addition_input": int(len(addition_masks)),
        "anchor_kept": int(len(anchor_indices)),
        "addition_kept": int(len(addition_indices)),
        "duplicates_removed": int(
            len(anchor_masks) + len(addition_masks) - len(result["sam_masks"])
        ),
        "merged_count": int(len(result["sam_masks"])),
        "addition_resized": int(addition_input_shape != anchor_shape),
        "addition_provenance_backfilled": int(backfilled_addition_provenance),
    }


def _read_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--anchor-candidate-manifest-sha256", required=True)
    parser.add_argument("--anchor-pseudo-manifest-sha256", required=True)
    parser.add_argument("--addition-root", type=Path, required=True)
    parser.add_argument("--addition-candidate-manifest-sha256", required=True)
    parser.add_argument("--addition-pseudo-manifest-sha256", required=True)
    parser.add_argument("--addition-namespace", required=True)
    parser.add_argument(
        "--allow-missing-addition-provenance",
        action="store_true",
        help=(
            "Preserve exact anchor provenance while marking a legacy addition "
            "with NaN/-1/stable unavailable sentinels. This changes metadata only."
        ),
    )
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
    )
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
        allow_test=args.split == "test",
    )
    expected = [str(row["image_id"]) for row in rows]
    anchor_rows, anchor_summary = validate_candidate_diagnostics_manifest(
        args.anchor_root,
        expected_image_names=expected,
        split=args.split,
        expected_pseudo_manifest_sha256=args.anchor_pseudo_manifest_sha256,
        expected_manifest_sha256=args.anchor_candidate_manifest_sha256,
    )
    addition_rows, addition_summary = validate_candidate_diagnostics_manifest(
        args.addition_root,
        expected_image_names=expected,
        split=args.split,
        expected_pseudo_manifest_sha256=args.addition_pseudo_manifest_sha256,
        expected_manifest_sha256=args.addition_candidate_manifest_sha256,
    )
    anchor_image_size = int(anchor_summary["image_size"])
    addition_image_size = int(addition_summary["image_size"])
    if anchor_image_size <= 0 or addition_image_size <= 0:
        raise ValueError("Candidate gallery output grids must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, object]] = []
    totals = {
        "anchor_input": 0,
        "addition_input": 0,
        "anchor_kept": 0,
        "addition_kept": 0,
        "duplicates_removed": 0,
        "merged_count": 0,
        "addition_resized": 0,
        "addition_provenance_backfilled": 0,
    }
    maximum_candidates = 0
    for row in rows:
        image_id = Path(str(row["image_id"])).stem
        anchor_row = anchor_rows[image_id]
        addition_row = addition_rows[image_id]
        anchor_path = args.anchor_root / anchor_row["diagnostic_path"]
        addition_path = args.addition_root / addition_row["diagnostic_path"]
        anchor = _read_payload(anchor_path)
        addition = _read_payload(addition_path)
        merged, stats = merge_payloads(
            anchor,
            addition,
            addition_namespace=args.addition_namespace,
            allow_missing_addition_provenance=(
                args.allow_missing_addition_provenance
            ),
        )
        for key in totals:
            totals[key] += stats[key]
        maximum_candidates = max(maximum_candidates, stats["merged_count"])
        diagnostic_path = (
            args.output_dir / "candidate_diagnostics" / f"{image_id}.npz"
        )
        result = save_candidate_diagnostics(
            diagnostic_path,
            sam_masks=merged["sam_masks"],
            refined_mask=anchor["refined_mask"],
            final_mask=anchor["final_mask"],
            bone_support=(
                anchor["bone_support"]
                if int(np.asarray(anchor["bone_support_present"])[0])
                else None
            ),
            prompt_map=merged["prompt_map"],
            positive_points=anchor["positive_points"],
            negative_points=anchor["negative_points"],
            boxes=anchor["boxes"],
            sam_scores=merged["sam_scores"],
            selection_scores=merged["selection_scores"],
            classifier_causal_scores=merged["classifier_causal_scores"],
            component_ids=merged["component_ids"],
            prompt_modes=merged["prompt_modes"],
            proposal_source_ids=merged["proposal_source_ids"],
            source_map_mean_scores=merged["source_map_mean_scores"],
            source_map_mass_coverages=merged["source_map_mass_coverages"],
            source_score_densities=merged["source_score_densities"],
            source_class_ids=merged["source_class_ids"],
            source_class_probabilities=merged["source_class_probabilities"],
            source_class_ranks=merged["source_class_ranks"],
            dsll_source_maps=merged.get("dsll_source_maps"),
            dsll_source_map_ids=merged.get("dsll_source_map_ids"),
            cam_levels=merged.get("cam_levels"),
            prompt_ids=merged.get("prompt_ids"),
            multimask_indices=merged.get("multimask_indices"),
        )
        output_rows.append(
            {
                "image_name": str(row["image_id"]),
                **result,
                **stats,
                "anchor_diagnostic_sha256": anchor_row["diagnostic_sha256"],
                "addition_diagnostic_sha256": addition_row["diagnostic_sha256"],
            }
        )
    summary = write_candidate_diagnostics_manifest(
        args.output_dir,
        output_rows,
        expected_image_names=expected,
        split=args.split,
        image_size=anchor_image_size,
        pseudo_manifest_sha256=args.anchor_pseudo_manifest_sha256,
        selection_method="geometry_v3_unconditional_gallery_union",
        support_clip_kernel=int(anchor_summary["support_clip_kernel"]),
        cam_percentile=float(anchor_summary["cam_percentile"]),
        cohort="all",
    )
    merge_contract = {
        "schema_version": 1,
        "split": args.split,
        "cohort": len(rows),
        "split_sha256": args.expected_split_sha256,
        "protocol_sha256": args.protocol_sha256,
        "anchor_candidate_manifest_sha256": args.anchor_candidate_manifest_sha256,
        "anchor_pseudo_manifest_sha256": args.anchor_pseudo_manifest_sha256,
        "addition_candidate_manifest_sha256": (
            args.addition_candidate_manifest_sha256
        ),
        "addition_pseudo_manifest_sha256": args.addition_pseudo_manifest_sha256,
        "addition_namespace": args.addition_namespace,
        "anchor_image_size": anchor_image_size,
        "addition_image_size": addition_image_size,
        "addition_alignment": (
            "identity"
            if anchor_image_size == addition_image_size
            else "fixed_nearest_neighbor_to_anchor_grid_before_dedup"
        ),
        "anchor_prompt_map_for_all_candidates": True,
        "exact_mask_deduplication": True,
        "missing_addition_provenance_policy": (
            "nan_level_minus1_multimask_stable_unavailable_prompt_id"
            if args.allow_missing_addition_provenance
            else "reject"
        ),
        "maximum_candidates": maximum_candidates,
        "totals": totals,
        "output_manifest_sha256": summary["manifest_sha256"],
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_images_read": len(rows) if args.split == "test" else 0,
        "test_evaluated": False,
    }
    path = args.output_dir / "gallery_merge_contract.json"
    path.write_text(
        json.dumps(merge_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**merge_contract, "merge_contract_sha256": sha256_file(path)},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
