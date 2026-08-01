from __future__ import annotations

"""Exploratory post-freeze G1 source/extent selector diagnostic."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--expected-diagnostic-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    return str(value)


def _average_percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("rank values must be one finite nonempty vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / max(len(values) - 1, 1)


def _choose(scores: np.ndarray, g1: np.ndarray, eligible: np.ndarray) -> int:
    scores = np.asarray(scores, dtype=np.float64)
    g1 = np.asarray(g1, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if scores.shape != g1.shape or eligible.shape != scores.shape or not eligible.any():
        raise ValueError("selector arrays do not align")
    indices = np.flatnonzero(eligible)
    # Stable lexicographic maximum: primary score, original G1, then lower index.
    return int(max(indices.tolist(), key=lambda i: (scores[i], g1[i], -i)))


def _score_without_metadata(
    model: RadDinoMaskBagMIL,
    descriptors: np.ndarray,
    flipped: np.ndarray,
) -> np.ndarray:
    first = np.asarray(descriptors, dtype=np.float32).copy()
    second = np.asarray(flipped, dtype=np.float32).copy()
    first[:, 1152:1156] = first[:, 1152:1156].mean(axis=0, keepdims=True)
    second[:, 1152:1156] = second[:, 1152:1156].mean(axis=0, keepdims=True)
    valid = torch.ones((1, len(first)), dtype=torch.bool)
    with torch.inference_mode():
        first_logits, _ = model.score_descriptors(torch.from_numpy(first)[None], valid)
        second_logits, _ = model.score_descriptors(torch.from_numpy(second)[None], valid)
    return (0.5 * (first_logits + second_logits))[0].numpy().astype(np.float64)


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def _size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    freeze_path = args.diagnostic_root / "diagnostic_freeze.json"
    if sha256_file(freeze_path) != args.expected_diagnostic_freeze_sha256:
        raise ValueError("Stage-A diagnostic freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("validation_images") != 371
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Stage-A diagnostic provenance mismatch")
    manifest = args.diagnostic_root / "descriptor_evidence_manifest.csv"
    if sha256_file(manifest) != freeze["descriptor_evidence_manifest_sha256"]:
        raise ValueError("descriptor evidence manifest changed")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        evidence_rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in evidence_rows}
    if len(indexed) != 371 or set(indexed) != {row["image_id"] for row in rows}:
        raise ValueError("Stage-A validation cohort mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("G1 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RadDinoMaskBagMIL(MaskBagMILConfig(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)

    selections: dict[str, dict[str, int]] = {}
    sources_by_image: dict[str, np.ndarray] = {}
    kept_by_image: dict[str, np.ndarray] = {}
    for image_id, row in indexed.items():
        evidence_path = args.diagnostic_root / "descriptor_evidence" / row["evidence_path"]
        if sha256_file(evidence_path) != row["evidence_sha256"]:
            raise ValueError(f"descriptor evidence changed: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            kept = payload["candidate_indices"].astype(np.int64)
            g1 = payload["candidate_logits"].astype(np.float64)
            upstream = payload["selection_scores"].astype(np.float64)
            metadata = payload["descriptor_metadata"].astype(np.float64)
            sources = np.asarray(
                [_canonical_source(value) for value in payload["proposal_source_ids"]]
            )
            no_metadata = _score_without_metadata(
                model,
                payload["descriptors"],
                payload["flipped_descriptors"],
            )
        shared = sources != "external_saliency"
        rank_g1 = _average_percentile_rank(g1)
        rank_nometa = _average_percentile_rank(no_metadata)
        rank_upstream = _average_percentile_rank(upstream)
        purity = np.clip(metadata[:, 3], 0.0, 1.0)
        completeness = np.clip(metadata[:, 2], 0.0, 1.0)
        harmonic = 2.0 * purity * completeness / np.maximum(
            purity + completeness, 1.0e-12
        )
        all_candidates = np.ones(len(g1), dtype=bool)
        variants = {
            "g1": (g1, all_candidates),
            "g1_shared_only": (g1, shared),
            "rank_fusion_g1_upstream": (0.5 * (rank_g1 + rank_upstream), all_candidates),
            "rank_fusion_nometa_upstream": (
                0.5 * (rank_nometa + rank_upstream),
                all_candidates,
            ),
            "rank_fusion_nometa_upstream_shared_only": (
                0.5 * (rank_nometa + rank_upstream),
                shared,
            ),
            "purity_completeness_harmonic": (harmonic, all_candidates),
        }
        selections[image_id] = {
            name: _choose(score, g1, eligible)
            for name, (score, eligible) in variants.items()
        }
        sources_by_image[image_id] = sources
        kept_by_image[image_id] = kept
    choice_freeze = {
        "stage": "rich_gallery_g1_shortcut_extent_followup_choices_v1",
        "analysis_role": "post_stage_b_exploratory_not_promotion",
        "stage_a_diagnostic_freeze_sha256": args.expected_diagnostic_freeze_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "validation_images": 371,
        "selections": selections,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    choice_path = args.output_dir / "choice_freeze.json"
    choice_path.write_text(json.dumps(choice_freeze, indent=2, sort_keys=True) + "\n")

    # Annotation boundary: all candidate choices above are frozen first.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index in range(len(dataset)):
        _image, mask_tensor, image_id = dataset[index]
        image_id = str(image_id)
        if indexed[image_id]["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        path = args.candidate_root / f"{Path(image_id).stem}.npz"
        if sha256_file(path) != indexed[image_id]["candidate_payload_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            masks = payload["sam_masks"].astype(bool)[kept_by_image[image_id]]
        subgroup = _size_group(float(target.mean()))
        for name, local_index in selections[image_id].items():
            prediction = masks[local_index]
            results[name].append(
                {
                    "image_id": image_id,
                    "size_group": subgroup,
                    "dice": _dice(prediction, target),
                    "complete_miss": int(not np.logical_and(prediction, target).any()),
                    "area_ratio": float(prediction.mean()),
                    "source": str(sources_by_image[image_id][local_index]),
                }
            )
    summary: dict[str, Any] = {}
    for name, records in sorted(results.items()):
        summary[name] = {}
        for subgroup in ("overall", "small", "medium", "large"):
            selected = [
                row
                for row in records
                if subgroup == "overall" or row["size_group"] == subgroup
            ]
            summary[name][subgroup] = {
                "n": len(selected),
                "dice": float(np.mean([row["dice"] for row in selected])),
                "complete_misses": int(sum(row["complete_miss"] for row in selected)),
                "selected_area_mean": float(np.mean([row["area_ratio"] for row in selected])),
                "selected_area_median": float(np.median([row["area_ratio"] for row in selected])),
                "source_counts": dict(Counter(row["source"] for row in selected)),
            }
    payload = {
        "stage": "rich_gallery_g1_shortcut_extent_followup_v1",
        "analysis_role": "post_stage_b_exploratory_not_promotion",
        "choice_freeze_sha256": sha256_file(choice_path),
        "cohort": {"validation": 371, "tumor": 184, "small": 94, "medium": 72, "large": 18},
        "variants": summary,
        "candidate_choices_frozen_before_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit = {
        "audit_pass": True,
        "choice_freeze_sha256": sha256_file(choice_path),
        "summary_sha256": sha256_file(summary_path),
        "semantic_summary_sha256": _sha256_json(payload),
        "candidate_choices_frozen_before_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

