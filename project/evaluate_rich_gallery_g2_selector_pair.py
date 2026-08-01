from __future__ import annotations

"""Post-freeze actual Dice/IoU evaluator for the matched G2 selector pair."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_g2_objective import rank_fusion_scores, stable_select
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_g2_selector_pair import MODEL_NAMES, canonical_source


PRIMARY_GATES = {"small": 0.195607621, "medium": 0.479674337, "large": 0.513613009}
FROZEN_FUSION = {
    "overall": 0.28872948670665205,
    "small": 0.15772329637374455,
    "medium": 0.4352293348209193,
    "large": 0.38687353265476676,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = int(np.logical_or(prediction, target).sum())
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def size_group(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def verify_stage_a(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("G2 prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_variants = {
        f"{model}__{rule}"
        for model in MODEL_NAMES
        for rule in ("raw", "rank_fusion")
    }
    if (
        freeze.get("stage") != "rich_gallery_g2_selector_pair_stage_a_v1"
        or freeze.get("source_commit") != args.expected_source_commit
        or freeze.get("protocol_sha256") != args.expected_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
        or freeze.get("g1_reproduction_max_selected_index_delta") != 0
        or freeze.get("validation_images") != 371
        or set(freeze.get("variants", [])) != expected_variants
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G2 prediction freeze contract mismatch")
    manifest = args.prediction_root / "stage_a_selection_manifest.csv"
    if sha256_file(manifest) != freeze["selection_manifest_sha256"]:
        raise ValueError("G2 selection manifest changed")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 371 * len(expected_variants):
        raise ValueError("G2 selection row count mismatch")
    cohort = {row["image_id"] for row in val_rows}
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    score_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        key = (row["variant"], row["image_id"])
        if row["variant"] not in expected_variants or row["image_id"] not in cohort or key in indexed:
            raise ValueError("G2 selection identity mismatch")
        score_path = args.prediction_root / row["score_path"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"G2 score payload changed: {row['image_id']}")
        if row["score_path"] not in score_cache:
            with np.load(score_path, allow_pickle=False) as payload:
                score_cache[row["score_path"]] = {key: payload[key] for key in payload.files}
        payload = score_cache[row["score_path"]]
        model = row["model"]
        logits = np.asarray(payload[f"{model}_candidate_logits"], dtype=np.float64)
        upstream = np.asarray(payload["upstream_scores"], dtype=np.float64)
        candidates = np.asarray(payload["candidate_indices"], dtype=np.int64)
        sources = np.asarray(payload["source_ids"], dtype=np.int64)
        if not (len(logits) == len(upstream) == len(candidates) == len(sources)):
            raise ValueError("G2 score arrays are misaligned")
        if row["rule"] == "raw":
            local = stable_select(logits, logits)
        elif row["rule"] == "rank_fusion":
            local = stable_select(rank_fusion_scores(logits, upstream), logits)
        else:
            raise ValueError("unknown G2 selection rule")
        if (
            int(row["selected_local_index"]) != local
            or int(row["selected_candidate_index"]) != int(candidates[local])
            or canonical_source(row["selected_source"])
            != {0: "classifier448", 1: "layercam320", 2: "external_saliency"}[int(sources[local])]
        ):
            raise ValueError(f"G2 frozen choice does not reproduce: {key}")
        indexed[key] = row
    for variant in expected_variants:
        if {image_id for current, image_id in indexed if current == variant} != cohort:
            raise ValueError(f"G2 variant cohort incomplete: {variant}")
    return indexed, freeze


def main() -> None:
    args = parse_args()
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("canonical validation cohort mismatch")
    selections, freeze = verify_stage_a(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all":
        raise ValueError("G2 evaluator requires complete validation candidates")

    # Annotation boundary: every candidate choice and its score payload has
    # been hash-verified and exactly recomputed above.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    expected_variants = list(freeze["variants"])
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, mask_tensor, image_id = dataset[index]
        image_id = str(image_id)
        base = selections[(expected_variants[0], image_id)]
        if base["tumor"] != "1":
            continue
        target = mask_tensor[0].numpy() > 0.5
        candidate_row = candidate_rows[Path(image_id).stem]
        path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed after Stage A: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
        candidate_dice = np.asarray([dice(mask, target) for mask in proposals])
        oracle_index = int(candidate_dice.argmax())
        area = float(target.mean())
        subgroup = size_group(area)
        for variant in expected_variants:
            selection = selections[(variant, image_id)]
            selected_index = int(selection["selected_candidate_index"])
            if not 0 <= selected_index < len(proposals):
                raise ValueError(f"selected index outside candidate bag: {variant}/{image_id}")
            prediction = proposals[selected_index]
            selected_dice = dice(prediction, target)
            per_image.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": selection["group_id"],
                    "size_group": subgroup,
                    "gt_area_ratio": area,
                    "dice": selected_dice,
                    "iou": iou(prediction, target),
                    "complete_miss": int(not np.logical_and(prediction, target).any()),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_source": canonical_source(sources[selected_index]),
                    "oracle_dice": float(candidate_dice[oracle_index]),
                    "selector_regret": float(candidate_dice[oracle_index] - selected_dice),
                }
            )
    if len(per_image) != 184 * len(expected_variants):
        raise RuntimeError("G2 tumor evaluation count mismatch")
    subgroup_counts = Counter(
        row["size_group"]
        for row in per_image
        if row["variant"] == expected_variants[0]
    )
    if subgroup_counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"G2 subgroup mismatch: {subgroup_counts}")

    summary: dict[str, Any] = {}
    for variant in sorted(expected_variants):
        records = [row for row in per_image if row["variant"] == variant]
        summary[variant] = {}
        for subgroup in ("overall", "small", "medium", "large"):
            selected = [
                row
                for row in records
                if subgroup == "overall" or row["size_group"] == subgroup
            ]
            mean_dice = float(np.mean([row["dice"] for row in selected]))
            summary[variant][subgroup] = {
                "n": len(selected),
                "dice": mean_dice,
                "iou": float(np.mean([row["iou"] for row in selected])),
                "complete_misses": int(sum(row["complete_miss"] for row in selected)),
                "selected_area_mean": float(np.mean([row["selected_area_ratio"] for row in selected])),
                "selected_area_median": float(np.median([row["selected_area_ratio"] for row in selected])),
                "oracle_dice": float(np.mean([row["oracle_dice"] for row in selected])),
                "selector_regret": float(np.mean([row["selector_regret"] for row in selected])),
                "selected_source_counts": dict(sorted(Counter(row["selected_source"] for row in selected).items())),
                "delta_vs_frozen_g1_upstream_fusion": mean_dice - FROZEN_FUSION[subgroup],
                "primary_gate": (
                    None if subgroup == "overall" else mean_dice >= PRIMARY_GATES[subgroup]
                ),
            }
    label_manifest = args.prediction_root / "stage_a_selection_manifest.csv"
    with label_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        label_rows = list(csv.DictReader(handle))
    image_auroc: dict[str, float] = {}
    for model in MODEL_NAMES:
        rows = [row for row in label_rows if row["model"] == model and row["rule"] == "raw"]
        image_auroc[model] = float(
            roc_auc_score(
                [int(row["tumor"]) for row in rows],
                [float(row["bag_probability"]) for row in rows],
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    result = {
        "stage": "rich_gallery_g2_selector_pair_post_freeze_evaluation_v1",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187, "small": 94, "medium": 72, "large": 18},
        "image_level_auroc_diagnostic": image_auroc,
        "actual_binary_mask_metrics": summary,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "spatial_ground_truth_used_for_training": False,
        "complete_misses_included": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_pass": True,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": args.expected_prediction_freeze_sha256,
        "candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "candidate_choices_frozen_before_validation_gt": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
