from __future__ import annotations

"""Post-freeze G1 ranking/extent decomposition on validation tumors."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_ranking_diagnostics import (
    candidate_ranking_diagnostics,
    summarize_ranking_diagnostics,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


TOP_K = (1, 3, 5, 10, 20, 50)
SOURCE_NAMES = ("classifier448", "external_saliency", "layercam320")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--expected-diagnostic-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _size_group(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("Spearman inputs must be aligned vectors")
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if np.ptp(first_rank) == 0.0 or np.ptp(second_rank) == 0.0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _canonical_source(value: str) -> str:
    lowered = value.lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    return value


def _load_frozen_evidence(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    freeze_path = args.diagnostic_root / "diagnostic_freeze.json"
    if sha256_file(freeze_path) != args.expected_diagnostic_freeze_sha256:
        raise ValueError("G1 diagnostic freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_g1_all_candidate_score_freeze_v1"
        or freeze.get("source_commit") != args.expected_source_commit
        or freeze.get("protocol_sha256") != args.expected_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
        or freeze.get("validation_images") != 371
        or freeze.get("maximum_candidates") != 243
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1 diagnostic freeze provenance mismatch")
    manifest_path = args.diagnostic_root / "descriptor_evidence_manifest.csv"
    if sha256_file(manifest_path) != freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("descriptor evidence manifest SHA-256 mismatch")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    expected = {row["image_id"]: row for row in val_rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != set(expected):
        raise ValueError("descriptor evidence cohort mismatch")
    for image_id, row in indexed.items():
        expected_row = expected[image_id]
        if (
            row["group_id"] != expected_row["group_id"]
            or row["tumor"] != expected_row["tumor"]
        ):
            raise ValueError(f"descriptor evidence identity mismatch: {image_id}")
        evidence_path = args.diagnostic_root / "descriptor_evidence" / row["evidence_path"]
        if sha256_file(evidence_path) != row["evidence_sha256"]:
            raise ValueError(f"descriptor evidence payload mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            required = {
                "schema_version",
                "descriptors",
                "flipped_descriptors",
                "candidate_indices",
                "original_logits",
                "flipped_logits",
                "candidate_logits",
                "descriptor_metadata",
                "shape_features",
                "sam_scores",
                "selection_scores",
                "classifier_causal_scores",
                "component_ids",
                "prompt_modes",
                "proposal_source_ids",
            }
            if not required.issubset(payload.files) or int(payload["schema_version"]) != 1:
                raise ValueError(f"descriptor evidence schema mismatch: {image_id}")
            count = int(row["candidate_count"])
            indices = payload["candidate_indices"]
            logits = payload["candidate_logits"]
            if (
                indices.dtype != np.int32
                or logits.dtype != np.float32
                or indices.shape != (count,)
                or logits.shape != (count,)
                or np.any(np.diff(indices) <= 0)
                or not np.isfinite(logits).all()
                or int(indices[int(np.argmax(logits))])
                != int(row["selected_candidate_index"])
                or float(logits.max()) != float(row["selected_candidate_logit"])
            ):
                raise ValueError(f"descriptor evidence content mismatch: {image_id}")
    return freeze, indexed


def _summary_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _score_descriptors(
    model: RadDinoMaskBagMIL,
    descriptors: np.ndarray,
    flipped_descriptors: np.ndarray,
) -> np.ndarray:
    count = len(descriptors)
    valid = torch.ones((1, count), dtype=torch.bool)
    with torch.inference_mode():
        original, _ = model.score_descriptors(
            torch.from_numpy(descriptors.astype(np.float32))[None],
            valid,
        )
        flipped, _ = model.score_descriptors(
            torch.from_numpy(flipped_descriptors.astype(np.float32))[None],
            valid,
        )
    return (0.5 * (original + flipped))[0].numpy().astype(np.float64)


def _replace_candidate_block_with_bag_mean(
    values: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[:, start:stop] = result[:, start:stop].mean(axis=0, keepdims=True)
    return result


def _source_z_scores(scores: np.ndarray, sources: np.ndarray) -> np.ndarray:
    result = np.zeros(len(scores), dtype=np.float64)
    for source in np.unique(sources):
        selected = sources == source
        values = scores[selected]
        scale = float(values.std())
        result[selected] = (
            values - float(values.mean())
        ) / max(scale, 1.0e-6)
    return result


def _freeze_selector_variants(
    args: argparse.Namespace,
    freeze: dict[str, object],
    evidence_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, int]], str]:
    checkpoint_path = args.baseline_root / "rad_dino_mask_bag_mil.pt"
    if sha256_file(checkpoint_path) != args.expected_baseline_checkpoint_sha256:
        raise ValueError("G1 checkpoint differs before selector ablation")
    if freeze.get("baseline_checkpoint_sha256") != args.expected_baseline_checkpoint_sha256:
        raise ValueError("Stage-A freeze is not bound to the G1 checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MaskBagMILConfig(**checkpoint["config"])
    model = RadDinoMaskBagMIL(config).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    selections: dict[str, dict[str, int]] = {}
    for image_id, row in evidence_rows.items():
        path = args.diagnostic_root / "descriptor_evidence" / row["evidence_path"]
        with np.load(path, allow_pickle=False) as payload:
            descriptors = payload["descriptors"].astype(np.float32)
            flipped = payload["flipped_descriptors"].astype(np.float32)
            frozen_logits = payload["candidate_logits"].astype(np.float64)
            sam = payload["sam_scores"].astype(np.float64)
            upstream = payload["selection_scores"].astype(np.float64)
            causal = payload["classifier_causal_scores"].astype(np.float64)
            metadata = payload["descriptor_metadata"].astype(np.float64)
            sources = np.asarray(
                [_canonical_source(str(value)) for value in payload["proposal_source_ids"]]
            )
        if descriptors.shape[1] != 1156:
            raise ValueError("unexpected G1 descriptor dimension")
        variants = {
            "g1": frozen_logits,
            "upstream_selection": upstream,
            "sam_score": sam,
            "classifier_causal_score": causal,
            "anchor_prompt_mass": metadata[:, 2],
            "anchor_prompt_inside": metadata[:, 3],
            "source_zscore_g1": _source_z_scores(frozen_logits, sources),
        }
        for name, start, stop in (
            ("without_inside_discrimination", 0, 384),
            ("without_context_discrimination", 384, 768),
            ("without_contrast_discrimination", 768, 1152),
            ("without_metadata_discrimination", 1152, 1156),
        ):
            variants[name] = _score_descriptors(
                model,
                _replace_candidate_block_with_bag_mean(descriptors, start, stop),
                _replace_candidate_block_with_bag_mean(flipped, start, stop),
            )
        for name, values in variants.items():
            if values.shape != (len(descriptors),) or not np.isfinite(values).all():
                raise ValueError(f"invalid frozen selector variant: {image_id}/{name}")
        selections[image_id] = {
            name: int(np.argmax(values)) for name, values in variants.items()
        }
        if selections[image_id]["g1"] != int(np.argmax(frozen_logits)):
            raise RuntimeError("G1 variant selection does not reproduce frozen logits")
    variant_freeze = {
        "stage": "rich_gallery_g1_selector_variant_freeze_v1",
        "stage_a_diagnostic_freeze_sha256": args.expected_diagnostic_freeze_sha256,
        "checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "selection_variants": sorted(next(iter(selections.values()))),
        "validation_images": len(selections),
        "selections": selections,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    path = args.output_dir / "selector_variant_freeze.json"
    path.write_text(
        json.dumps(variant_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selections, sha256_file(path)


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    freeze, evidence_rows = _load_frozen_evidence(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("G1 ranking evaluator requires the complete candidate cohort")
    for image_id, row in evidence_rows.items():
        candidate = candidate_rows[Path(image_id).stem]
        if row["candidate_payload_sha256"] != candidate["diagnostic_sha256"]:
            raise ValueError(f"frozen evidence/candidate hash mismatch: {image_id}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    variant_selections, variant_freeze_sha256 = _freeze_selector_variants(
        args,
        freeze,
        evidence_rows,
    )

    # Annotation boundary: every continuous score, descriptor and candidate payload
    # and every selector-ablation choice are frozen above before validation
    # polygons are imported/opened.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    per_image: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_matrix: Counter[tuple[str, str]] = Counter()
    source_candidate_counts: Counter[str] = Counter()
    source_wins: Counter[str] = Counter()
    source_oracles: Counter[str] = Counter()
    variant_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index in range(len(dataset)):
        _image, mask_tensor, image_name = dataset[index]
        image_id = str(image_name)
        evidence_row = evidence_rows[image_id]
        if evidence_row["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        evidence_path = (
            args.diagnostic_root
            / "descriptor_evidence"
            / evidence_row["evidence_path"]
        )
        with np.load(evidence_path, allow_pickle=False) as evidence:
            kept = evidence["candidate_indices"].astype(np.int64)
            logits = evidence["candidate_logits"].astype(np.float64)
            original_logits = evidence["original_logits"].astype(np.float64)
            flipped_logits = evidence["flipped_logits"].astype(np.float64)
            metadata = evidence["descriptor_metadata"].astype(np.float64)
            shape = evidence["shape_features"].astype(np.float64)
            sam_scores = evidence["sam_scores"].astype(np.float64)
            selection_scores = evidence["selection_scores"].astype(np.float64)
            causal_scores = evidence["classifier_causal_scores"].astype(np.float64)
            sources = np.asarray(
                [_canonical_source(str(value)) for value in evidence["proposal_source_ids"]]
            )
        candidate_row = candidate_rows[Path(image_id).stem]
        with np.load(
            args.val_candidate_root / candidate_row["diagnostic_path"],
            allow_pickle=False,
        ) as candidate:
            masks = candidate["sam_masks"].astype(bool)[kept]
        if masks.shape[0] != len(logits):
            raise ValueError(f"candidate mask/score count mismatch: {image_id}")
        dice = np.asarray([_dice(mask, target) for mask in masks], dtype=np.float64)
        areas = masks.reshape(len(masks), -1).mean(axis=1)
        ranking = candidate_ranking_diagnostics(logits, dice, top_k=TOP_K)
        selected_local = int(ranking["selected_candidate_index"])
        oracle_local = int(ranking["oracle_candidate_index"])
        selected_source = str(sources[selected_local])
        oracle_source = str(sources[oracle_local])
        source_matrix[(selected_source, oracle_source)] += 1
        source_wins[selected_source] += 1
        source_oracles[oracle_source] += 1
        source_candidate_counts.update(str(source) for source in sources)
        selected_source_mask = sources == selected_source
        best_same_source = float(dice[selected_source_mask].max())
        source_choice_regret = float(ranking["oracle_quality"]) - best_same_source
        within_source_regret = best_same_source - float(ranking["selected_quality"])
        if abs(
            source_choice_regret
            + within_source_regret
            - float(ranking["selected_to_oracle_regret"])
        ) > 1.0e-10:
            raise RuntimeError("regret decomposition identity failed")
        temperature = 0.2
        shifted = (logits - logits.max()) / temperature
        mil_weights = np.exp(shifted) / np.exp(shifted).sum()
        selected_mask = masks[selected_local]
        target_area = float(target.mean())
        selected_area = float(areas[selected_local])
        subgroup = _size_group(target_area)
        record = {
            "image_id": image_id,
            "group_id": evidence_row["group_id"],
            "size_group": subgroup,
            **ranking,
            "selected_complete_miss": int(
                not np.logical_and(selected_mask, target).any()
            ),
            "baseline_complete_miss": int(
                not np.logical_and(selected_mask, target).any()
            ),
            "selected_source": selected_source,
            "oracle_source": oracle_source,
            "selected_source_matches_oracle": int(selected_source == oracle_source),
            "best_same_selected_source_quality": best_same_source,
            "source_choice_regret": source_choice_regret,
            "within_source_regret": within_source_regret,
            "gt_area_ratio": target_area,
            "selected_area_ratio": selected_area,
            "selected_to_gt_area_ratio": selected_area / max(target_area, 1.0e-12),
            "selected_log_area_error": abs(
                math.log(max(selected_area, 1.0e-12) / max(target_area, 1.0e-12))
            ),
            "mil_selected_weight": float(mil_weights[selected_local]),
            "mil_oracle_weight": float(mil_weights[oracle_local]),
            "mil_effective_candidate_count": float(1.0 / np.square(mil_weights).sum()),
            "original_flip_score_spearman": _spearman(original_logits, flipped_logits),
            "hard_positive_is_low_quality": int(float(ranking["selected_quality"]) < 0.1),
        }
        per_image.append(record)
        for variant, variant_local in variant_selections[image_id].items():
            variant_mask = masks[variant_local]
            variant_rows[variant].append(
                {
                    "size_group": subgroup,
                    "dice": float(dice[variant_local]),
                    "complete_miss": int(
                        not np.logical_and(variant_mask, target).any()
                    ),
                    "area_ratio": float(areas[variant_local]),
                    "source": str(sources[variant_local]),
                }
            )
        for position, source in enumerate(sources):
            bucket = pooled[str(source)]
            bucket["logit"].append(float(logits[position]))
            bucket["dice"].append(float(dice[position]))
            bucket["area"].append(float(areas[position]))
            bucket["sam_score"].append(float(sam_scores[position]))
            bucket["selection_score"].append(float(selection_scores[position]))
            bucket["causal_score"].append(float(causal_scores[position]))
            bucket["prompt_mass_fraction"].append(float(metadata[position, 2]))
            bucket["prompt_inside_mean"].append(float(metadata[position, 3]))
            bucket["compactness"].append(float(shape[position, 3]))
    if len(per_image) != 184:
        raise RuntimeError(f"expected 184 tumor images, got {len(per_image)}")
    subgroup_counts = Counter(str(row["size_group"]) for row in per_image)
    if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"subgroup mismatch: {subgroup_counts}")

    ranking_summary = summarize_ranking_diagnostics(
        per_image,
        top_k=TOP_K,
    )
    decomposition: dict[str, dict[str, Any]] = {}
    for subgroup in ("overall", "small", "medium", "large"):
        rows = [
            row
            for row in per_image
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        decomposition[subgroup] = {
            "n": len(rows),
            "selected_dice": float(np.mean([row["selected_quality"] for row in rows])),
            "oracle_dice": float(np.mean([row["oracle_quality"] for row in rows])),
            "total_regret": float(np.mean([row["selected_to_oracle_regret"] for row in rows])),
            "source_choice_regret": float(np.mean([row["source_choice_regret"] for row in rows])),
            "within_source_regret": float(np.mean([row["within_source_regret"] for row in rows])),
            "source_match_rate": float(np.mean([row["selected_source_matches_oracle"] for row in rows])),
            "oracle_rank": _summary_stats([float(row["oracle_best_rank"]) for row in rows]),
            "selected_to_gt_area_ratio": _summary_stats([row["selected_to_gt_area_ratio"] for row in rows]),
            "mil_selected_weight": _summary_stats([row["mil_selected_weight"] for row in rows]),
            "mil_oracle_weight": _summary_stats([row["mil_oracle_weight"] for row in rows]),
            "mil_effective_candidate_count": _summary_stats([row["mil_effective_candidate_count"] for row in rows]),
            "hard_positive_low_quality_rate": float(np.mean([row["hard_positive_is_low_quality"] for row in rows])),
            "dice_vs_log_area_error_spearman": _spearman(
                np.asarray([row["selected_quality"] for row in rows]),
                np.asarray([row["selected_log_area_error"] for row in rows]),
            ),
        }
    source_diagnostics: dict[str, dict[str, Any]] = {}
    for source, values in sorted(pooled.items()):
        logit = np.asarray(values["logit"])
        source_diagnostics[source] = {
            "candidate_count": len(logit),
            "winner_count": int(source_wins[source]),
            "oracle_count": int(source_oracles[source]),
            "mean_logit": float(logit.mean()),
            "logit_p90": float(np.percentile(logit, 90)),
            "pooled_score_dice_spearman": _spearman(logit, np.asarray(values["dice"])),
            "score_area_spearman": _spearman(logit, np.asarray(values["area"])),
            "score_sam_spearman": _spearman(logit, np.asarray(values["sam_score"])),
            "score_selection_spearman": _spearman(logit, np.asarray(values["selection_score"])),
            "score_causal_spearman": _spearman(logit, np.asarray(values["causal_score"])),
            "score_prompt_mass_spearman": _spearman(logit, np.asarray(values["prompt_mass_fraction"])),
            "score_prompt_inside_spearman": _spearman(logit, np.asarray(values["prompt_inside_mean"])),
            "score_compactness_spearman": _spearman(logit, np.asarray(values["compactness"])),
        }
    matrix = {
        selected: {
            oracle: int(source_matrix[(selected, oracle)])
            for oracle in sorted({key[1] for key in source_matrix})
        }
        for selected in sorted({key[0] for key in source_matrix})
    }
    selector_variants: dict[str, dict[str, Any]] = {}
    for variant, all_rows in sorted(variant_rows.items()):
        subgroup_result: dict[str, Any] = {}
        for subgroup in ("overall", "small", "medium", "large"):
            rows = [
                row
                for row in all_rows
                if subgroup == "overall" or row["size_group"] == subgroup
            ]
            subgroup_result[subgroup] = {
                "n": len(rows),
                "dice": float(np.mean([row["dice"] for row in rows])),
                "complete_misses": int(sum(row["complete_miss"] for row in rows)),
                "selected_area_ratio": _summary_stats(
                    [row["area_ratio"] for row in rows]
                ),
                "source_counts": dict(Counter(row["source"] for row in rows)),
            }
        selector_variants[variant] = subgroup_result
    summary = {
        "stage": "rich_gallery_g1_post_freeze_ranking_diagnostic_v1",
        "cohort": {"validation": 371, "tumor": 184, **dict(subgroup_counts)},
        "ranking": ranking_summary,
        "regret_decomposition": decomposition,
        "source_diagnostics": source_diagnostics,
        "selected_source_by_oracle_source": matrix,
        "source_candidate_counts": dict(source_candidate_counts),
        "selector_variant_actual_dice": selector_variants,
        "selector_variant_freeze_sha256": variant_freeze_sha256,
        "candidate_scores_frozen_before_gt": True,
        "complete_misses_included": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    per_image_path = args.output_dir / "per_image.jsonl"
    with per_image_path.open("w", encoding="utf-8") as handle:
        for row in per_image:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "audit_pass": True,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "diagnostic_freeze_sha256": args.expected_diagnostic_freeze_sha256,
        "selector_variant_freeze_sha256": variant_freeze_sha256,
        "candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "candidate_scores_frozen_before_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
