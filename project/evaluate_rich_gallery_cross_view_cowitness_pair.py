from __future__ import annotations

"""Post-freeze actual-Dice evaluator for cross-view co-witness selectors."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.rich_gallery_g2_objective import rank_fusion_scores, stable_select
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rich_gallery_cross_view_cowitness_pair import (
    ARM_NAMES,
    RESIDUAL_MULTIPLIERS,
    frozen_variants,
    variant_spec,
)
from run_rich_gallery_g2_selector_pair import canonical_source


FROZEN_BASELINE = {
    "overall": 0.28872948670665205,
    "small": 0.15772329637374455,
    "medium": 0.4352293348209193,
    "large": 0.38687353265476676,
}
SOURCE_ID_TO_NAME = {0: "classifier448", 1: "layercam320", 2: "external_saliency"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-freeze-sha256", required=True)
    parser.add_argument("--stage-a-audit", type=Path, required=True)
    parser.add_argument("--expected-stage-a-audit-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / denominator) if denominator else 1.0


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


def _score_set_sha256(score_hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(score_hashes)).encode("utf-8")).hexdigest()


def verify_stage_a(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
    *,
    require_independent_audit: bool = True,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
    if require_independent_audit:
        if sha256_file(args.stage_a_audit) != args.expected_stage_a_audit_sha256:
            raise ValueError("cross-view Stage-A audit SHA-256 mismatch")
        stage_a_audit = json.loads(args.stage_a_audit.read_text(encoding="utf-8"))
        if (
            stage_a_audit.get("audit_pass") is not True
            or stage_a_audit.get("prediction_freeze_sha256")
            != args.expected_prediction_freeze_sha256
            or stage_a_audit.get("test_images_read") != 0
            or stage_a_audit.get("test_evaluated") is not False
        ):
            raise ValueError("cross-view Stage-A audit contract mismatch")
    freeze_path = args.prediction_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_prediction_freeze_sha256:
        raise ValueError("cross-view prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_variants = frozen_variants()
    if (
        freeze.get("stage") != "rich_gallery_cross_view_cowitness_pair_stage_a_v1"
        or freeze.get("source_commit") != args.expected_source_commit
        or freeze.get("protocol_sha256") != args.expected_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("pair_manifest_sha256") != args.expected_pair_manifest_sha256
        or freeze.get("g1_checkpoint_sha256") != args.expected_g1_checkpoint_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.expected_val_pseudo_manifest_sha256
        or freeze.get("validation_images") != 371
        or freeze.get("pair_rows") != 384
        or freeze.get("selection_rows") != 371 * len(expected_variants)
        or freeze.get("variants") != expected_variants
        or freeze.get("residual_multipliers") != list(RESIDUAL_MULTIPLIERS)
        or freeze.get("zero_residual_baseline_reproduced") is not True
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("cross-view prediction freeze contract mismatch")
    manifest_path = args.prediction_root / "stage_a_selection_manifest.csv"
    if sha256_file(manifest_path) != freeze["selection_manifest_sha256"]:
        raise ValueError("cross-view selection manifest changed")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 371 * len(expected_variants):
        raise ValueError("cross-view selection row count mismatch")
    val_by_id = {row["image_id"]: row for row in val_rows}
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    cache: dict[str, dict[str, np.ndarray]] = {}
    score_hashes: dict[str, str] = {}
    for row in rows:
        variant = row["variant"]
        image_id = row["image_id"]
        key = (variant, image_id)
        if variant not in expected_variants or image_id not in val_by_id or key in indexed:
            raise ValueError(f"cross-view selection identity mismatch: {key}")
        if row["group_id"] != val_by_id[image_id]["group_id"] or int(row["tumor"]) != int(val_by_id[image_id]["tumor"]):
            raise ValueError(f"cross-view selection label/group mismatch: {key}")
        score_path = args.prediction_root / row["score_path"]
        actual_hash = sha256_file(score_path)
        if actual_hash != row["score_sha256"]:
            raise ValueError(f"cross-view score payload changed: {image_id}")
        score_hashes[row["score_path"]] = actual_hash
        if row["score_path"] not in cache:
            with np.load(score_path, allow_pickle=False) as payload:
                cache[row["score_path"]] = {name: payload[name] for name in payload.files}
        payload = cache[row["score_path"]]
        required = {
            "candidate_indices", "source_ids", "g1_logits", "upstream_scores",
            "baseline_fusion", "baseline_scores", "control_residual", "full_residual",
        }
        if not required.issubset(payload):
            raise ValueError("cross-view score payload schema mismatch")
        arrays = {name: np.asarray(payload[name]).reshape(-1) for name in required}
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1 or lengths == {0} or not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("cross-view score arrays are empty, non-finite or misaligned")
        g1 = arrays["g1_logits"].astype(np.float64)
        upstream = arrays["upstream_scores"].astype(np.float64)
        expected_fusion = rank_fusion_scores(g1, upstream)
        if not np.allclose(arrays["baseline_fusion"], expected_fusion, rtol=0.0, atol=2e-7):
            raise ValueError("cross-view baseline fusion does not reproduce")
        if not np.allclose(arrays["baseline_scores"], 2.0 * expected_fusion - 1.0, rtol=0.0, atol=3e-7):
            raise ValueError("cross-view centered baseline does not reproduce")
        source_ids = arrays["source_ids"].astype(np.int64)
        if not set(source_ids.tolist()).issubset(SOURCE_ID_TO_NAME):
            raise ValueError("cross-view source IDs changed")
        arm, multiplier = variant_spec(variant)
        scores = arrays["baseline_scores"].astype(np.float64)
        if arm is not None:
            residual = arrays[f"{arm}_residual"].astype(np.float64)
            if np.any(np.abs(residual[source_ids == 2]) > 1.0e-7):
                raise ValueError("external-source residual is not frozen at zero")
            scores = scores + float(multiplier) * residual
        local = stable_select(scores, g1)
        candidate_indices = arrays["candidate_indices"].astype(np.int64)
        if (
            int(row["selected_local_index"]) != local
            or int(row["selected_candidate_index"]) != int(candidate_indices[local])
            or canonical_source(row["selected_source"]) != SOURCE_ID_TO_NAME[int(source_ids[local])]
        ):
            raise ValueError(f"cross-view frozen choice does not reproduce: {key}")
        indexed[key] = row
    if len(score_hashes) != 371 or _score_set_sha256(list(score_hashes.values())) != freeze["score_set_sha256"]:
        raise ValueError("cross-view score-set hash/count mismatch")
    for variant in expected_variants:
        if {image_id for current, image_id in indexed if current == variant} != set(val_by_id):
            raise ValueError(f"cross-view variant cohort incomplete: {variant}")
    return indexed, freeze


def _paired_group_bootstrap(
    per_image: list[dict[str, object]],
    variant_a: str,
    variant_b: str,
    *,
    subgroup: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    selected = [row for row in per_image if subgroup == "overall" or row["size_group"] == subgroup]
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_group[str(row["group_id"])].append(row)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("empty paired bootstrap cohort")
    deltas: dict[str, float] = {}
    for group in groups:
        rows = by_group[group]
        a = [float(row["dice"]) for row in rows if row["variant"] == variant_a]
        b = [float(row["dice"]) for row in rows if row["variant"] == variant_b]
        if len(a) != len(b) or not a:
            raise ValueError("paired bootstrap variants are incomplete")
        deltas[group] = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    values = np.asarray([deltas[group] for group in groups], dtype=np.float64)
    boot = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "groups": len(groups),
        "mean_delta": float(values.mean()),
        "ci95": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "replicates": replicates,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("cross-view Stage-B output must not exist")
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if len(val_rows) != 371:
        raise RuntimeError("cross-view evaluator requires canonical validation cohort")
    selections, freeze = verify_stage_a(args, val_rows)
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
    )
    if len(candidate_rows) != 371 or candidate_audit.get("cohort") != "all":
        raise ValueError("cross-view candidate cohort audit failed")

    # Annotation boundary: all variants have been independently audited and
    # hash-frozen above.  Only the 184 tumor samples are opened below.
    from datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    sample_index = {str(sample["image_id"]): index for index, sample in enumerate(dataset.samples)}
    tumor_rows = [row for row in val_rows if int(row["tumor"]) == 1]
    if len(tumor_rows) != 184:
        raise RuntimeError("canonical validation-tumor count mismatch")
    variants = freeze["variants"]
    per_image: list[dict[str, object]] = []
    for split_row in tumor_rows:
        image_id = split_row["image_id"]
        _image, mask_tensor, loaded_id = dataset[sample_index[image_id]]
        if str(loaded_id) != image_id:
            raise ValueError("validation annotation identity mismatch")
        target = mask_tensor[0].numpy() > 0.5
        area = float(target.mean())
        subgroup = size_group(area)
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed after Stage A: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = np.asarray(payload["sam_masks"], dtype=bool)
            all_sources = np.asarray([canonical_source(value) for value in payload["proposal_source_ids"]])
        candidate_dice = np.asarray([dice(mask, target) for mask in proposals], dtype=np.float64)
        all_oracle = float(candidate_dice.max())
        for variant in variants:
            selection = selections[(variant, image_id)]
            selected = int(selection["selected_candidate_index"])
            score_path = args.prediction_root / selection["score_path"]
            with np.load(score_path, allow_pickle=False) as score_payload:
                eligible_indices = np.asarray(score_payload["candidate_indices"], dtype=np.int64)
            if selected not in eligible_indices or not 0 <= selected < len(proposals):
                raise ValueError(f"cross-view selected index outside eligible gallery: {variant}/{image_id}")
            eligible_dice = candidate_dice[eligible_indices]
            eligible_sources = all_sources[eligible_indices]
            selected_source = canonical_source(all_sources[selected])
            source_values = eligible_dice[eligible_sources == selected_source]
            selected_dice = float(candidate_dice[selected])
            eligible_oracle = float(eligible_dice.max())
            source_oracle = float(source_values.max())
            prediction = proposals[selected]
            per_image.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "size_group": subgroup,
                    "gt_area_ratio": area,
                    "dice": selected_dice,
                    "iou": iou(prediction, target),
                    "complete_miss": int(not np.logical_and(prediction, target).any()),
                    "selected_area_ratio": float(prediction.mean()),
                    "selected_gt_area_ratio": float(prediction.mean() / max(area, 1.0e-12)),
                    "selected_source": selected_source,
                    "gallery_oracle_dice": all_oracle,
                    "eligible_oracle_dice": eligible_oracle,
                    "selected_source_oracle_dice": source_oracle,
                    "proposal_supply_regret": all_oracle - eligible_oracle,
                    "cross_source_regret": eligible_oracle - source_oracle,
                    "within_source_regret": source_oracle - selected_dice,
                    "eligible_selector_regret": eligible_oracle - selected_dice,
                    "oracle_source_match": int(selected_source == canonical_source(all_sources[int(candidate_dice.argmax())])),
                }
            )
    if len(per_image) != 184 * len(variants):
        raise RuntimeError("cross-view tumor evaluation count mismatch")
    counts = Counter(
        row["size_group"] for row in per_image if row["variant"] == "baseline"
    )
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"cross-view subgroup mismatch: {counts}")

    metrics: dict[str, Any] = {}
    for variant in variants:
        records = [row for row in per_image if row["variant"] == variant]
        metrics[variant] = {}
        for subgroup in ("overall", "small", "medium", "large"):
            selected = [row for row in records if subgroup == "overall" or row["size_group"] == subgroup]
            mean_dice = float(np.mean([row["dice"] for row in selected]))
            metrics[variant][subgroup] = {
                "n": len(selected),
                "dice": mean_dice,
                "iou": float(np.mean([row["iou"] for row in selected])),
                "complete_misses": int(sum(int(row["complete_miss"]) for row in selected)),
                "selected_gt_area_ratio_median": float(np.median([row["selected_gt_area_ratio"] for row in selected])),
                "gallery_oracle_dice": float(np.mean([row["gallery_oracle_dice"] for row in selected])),
                "eligible_oracle_dice": float(np.mean([row["eligible_oracle_dice"] for row in selected])),
                "proposal_supply_regret": float(np.mean([row["proposal_supply_regret"] for row in selected])),
                "cross_source_regret": float(np.mean([row["cross_source_regret"] for row in selected])),
                "within_source_regret": float(np.mean([row["within_source_regret"] for row in selected])),
                "eligible_selector_regret": float(np.mean([row["eligible_selector_regret"] for row in selected])),
                "wrong_oracle_source": int(sum(1 - int(row["oracle_source_match"]) for row in selected)),
                "selected_source_counts": dict(sorted(Counter(str(row["selected_source"]) for row in selected).items())),
                "delta_vs_frozen_baseline": mean_dice - FROZEN_BASELINE[subgroup],
            }
    for subgroup, expected in FROZEN_BASELINE.items():
        if not np.isclose(metrics["baseline"][subgroup]["dice"], expected, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"immutable baseline failed to reproduce for {subgroup}")

    full_variants = [name for name in variants if name.startswith("full__")]
    control_variants = [name for name in variants if name.startswith("control__")]
    best_full = max(full_variants, key=lambda name: metrics[name]["overall"]["dice"])
    best_control = max(control_variants, key=lambda name: metrics[name]["overall"]["dice"])
    _arm, best_multiplier = variant_spec(best_full)
    matched_control = next(
        name for name in control_variants if variant_spec(name)[1] == best_multiplier
    )
    paired: dict[str, Any] = {}
    for comparator in ("baseline", matched_control):
        paired[comparator] = {
            subgroup: _paired_group_bootstrap(
                per_image,
                best_full,
                comparator,
                subgroup=subgroup,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + (0 if comparator == "baseline" else 100) + index,
            )
            for index, subgroup in enumerate(("overall", "small", "medium", "large"))
        }
    full_dice = metrics[best_full]["overall"]["dice"]
    matched_control_dice = metrics[matched_control]["overall"]["dice"]
    decision = {
        "exploratory_best_full_variant": best_full,
        "exploratory_best_control_variant": best_control,
        "multiplier_matched_control": matched_control,
        "full_overall_dice": full_dice,
        "matched_control_overall_dice": matched_control_dice,
        "beats_immutable_baseline": full_dice > FROZEN_BASELINE["overall"],
        "beats_multiplier_matched_control": full_dice > matched_control_dice,
        "longer_run_supported": (
            full_dice > FROZEN_BASELINE["overall"] and full_dice > matched_control_dice
        ),
        "selection_status": "exploratory_validation_global_multiplier",
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = args.output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    result = {
        "stage": "rich_gallery_cross_view_cowitness_pair_post_freeze_evaluation_v1",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187, "small": 94, "medium": 72, "large": 18},
        "actual_binary_mask_metrics": metrics,
        "paired_group_bootstrap": paired,
        "decision": decision,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read_only_after_stage_a_audit": True,
        "spatial_ground_truth_used_for_training_or_selection": False,
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
        "stage_a_audit_sha256": args.expected_stage_a_audit_sha256,
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
