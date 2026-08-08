from __future__ import annotations

"""Independent fail-closed audit for the G4 E6 G1 feature/loss study."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SEEDS = (42, 43, 44)
UNIQUE_KEYS = (
    "feature_inside_only",
    "feature_inside_ring",
    "feature_inside_ring_contrast",
    "full",
    "loss_bag_only",
    "loss_bag_negative",
    "loss_bag_selfguided",
)
REPORTED_PREFIXES = (
    "E6F__inside_only",
    "E6F__inside_ring",
    "E6F__inside_ring_contrast",
    "E6F__full",
    "E6L__bag_only",
    "E6L__bag_negative",
    "E6L__bag_selfguided",
    "E6L__full",
)
BASELINE_ARM = "E8__R7"
EXPECTED_BASELINE_COMMON320 = 0.28872948670665205
EXPECTED_BASELINE_NATIVE = 0.28822402200654273


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_unique_models() -> set[str]:
    return {f"{key}__seed{seed}" for seed in SEEDS for key in UNIQUE_KEYS}


def expected_reported_arms() -> set[str]:
    return {
        f"{prefix}__seed{seed}" for seed in SEEDS for prefix in REPORTED_PREFIXES
    }


def _require_zero_test(payload: dict[str, object], name: str) -> None:
    if int(payload.get("test_images_read", -1)) != 0 or payload.get("test_evaluated") is not False:
        raise ValueError(f"{name} test contract differs")


def _require_matrix(
    rows: list[dict[str, str]], arms: set[str], *, name: str
) -> tuple[set[str], int]:
    if len(rows) != 371 * len(arms):
        raise ValueError(f"{name} row count differs")
    by_image: dict[str, set[str]] = defaultdict(set)
    tumor_by_image: dict[str, int] = {}
    for row in rows:
        image_id = row["image_id"]
        arm = row["arm"]
        if arm not in arms or arm in by_image[image_id]:
            raise ValueError(f"{name} arm matrix has a duplicate or unknown arm")
        by_image[image_id].add(arm)
        tumor = int(row["tumor"])
        if image_id in tumor_by_image and tumor_by_image[image_id] != tumor:
            raise ValueError(f"{name} tumor label differs across arms")
        tumor_by_image[image_id] = tumor
    if len(by_image) != 371 or any(value != arms for value in by_image.values()):
        raise ValueError(f"{name} arm matrix is incomplete")
    tumor_images = sum(tumor_by_image.values())
    if tumor_images != 184:
        raise ValueError(f"{name} tumor count differs")
    return set(by_image), tumor_images


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _aggregate(summary: dict[str, object], label_metrics: dict[str, object]) -> dict[str, object]:
    families = {
        "feature_inside_only": "E6F__inside_only",
        "feature_inside_ring": "E6F__inside_ring",
        "feature_inside_ring_contrast": "E6F__inside_ring_contrast",
        "feature_full": "E6F__full",
        "loss_bag_only": "E6L__bag_only",
        "loss_bag_negative": "E6L__bag_negative",
        "loss_bag_selfguided": "E6L__bag_selfguided",
        "loss_full": "E6L__full",
    }
    summaries = summary["summaries"]
    result: dict[str, object] = {}
    for family, prefix in families.items():
        arms = [f"{prefix}__seed{seed}" for seed in SEEDS]
        metrics = {
            "native_dice": [
                _finite(summaries[arm]["mean_tumor_dice"], f"{arm}/native Dice")
                for arm in arms
            ],
            "common320_dice": [
                _finite(summaries[arm]["selected_dice_common320"], f"{arm}/common Dice")
                for arm in arms
            ],
            "small_native_dice": [
                _finite(
                    summaries[arm]["native_subgroups"]["small_lt_1pct"]["mean_tumor_dice"],
                    f"{arm}/small Dice",
                )
                for arm in arms
            ],
            "medium_native_dice": [
                _finite(
                    summaries[arm]["native_subgroups"]["medium_1_to_5pct"]["mean_tumor_dice"],
                    f"{arm}/medium Dice",
                )
                for arm in arms
            ],
            "large_native_dice": [
                _finite(
                    summaries[arm]["native_subgroups"]["large_ge_5pct"]["mean_tumor_dice"],
                    f"{arm}/large Dice",
                )
                for arm in arms
            ],
            "image_auroc": [
                _finite(label_metrics[arm]["auroc"], f"{arm}/image AUROC") for arm in arms
            ],
        }
        result[family] = {
            key: {
                "values_by_seed": dict(zip((str(seed) for seed in SEEDS), values)),
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
            for key, values in metrics.items()
        }
    return result


def audit(
    root: Path,
    *,
    expected_source_commit: str,
    expected_protocol_sha256: str,
    expected_split_sha256: str,
    expected_candidate_manifest_sha256: str,
) -> dict[str, object]:
    required = {
        "freeze": root / "g4_choice_freeze.json",
        "choices": root / "g4_choices.csv",
        "histories": root / "training_histories.json",
        "label_metrics": root / "image_label_metrics.json",
        "manifest": root / "run_manifest.json",
        "summary": root / "evaluation" / "summary.json",
        "evaluation_audit": root / "evaluation" / "evaluation_audit.json",
        "per_image": root / "evaluation" / "per_image_all_arms.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E6 output files: {missing}")

    freeze = read_json(required["freeze"])
    reported_arms = expected_reported_arms()
    all_arms = {BASELINE_ARM, *reported_arms}
    if (
        freeze.get("stage") != "g4_offline_ablation_choice_freeze_v1"
        or freeze.get("study") != "G4 E6 matched G1 feature/loss ablations"
        or freeze.get("source_commit") != expected_source_commit
        or freeze.get("protocol_sha256") != expected_protocol_sha256
        or freeze.get("split_sha256") != expected_split_sha256
        or freeze.get("candidate_manifest_sha256") != expected_candidate_manifest_sha256
        or int(freeze.get("images", -1)) != 371
        or int(freeze.get("tumor_images", -1)) != 184
        or int(freeze.get("selection_rows", -1)) != 9275
        or freeze.get("seeds") != list(SEEDS)
        or int(freeze.get("unique_models_per_seed", -1)) != 7
        or int(freeze.get("reported_learned_arms_per_seed", -1)) != 8
        or freeze.get("full_feature_and_full_loss_alias_exact") is not True
        or int(freeze.get("baseline_r7_exact_matches", -1)) != 371
        or freeze.get("descriptor_cache_encoded_once") is not True
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or set(freeze.get("arms", [])) != all_arms
    ):
        raise ValueError("E6 freeze contract differs")
    _require_zero_test(freeze, "freeze")

    if (
        freeze.get("choices_sha256") != sha256(required["choices"])
        or freeze.get("training_histories_sha256") != sha256(required["histories"])
        or freeze.get("image_label_metrics_sha256") != sha256(required["label_metrics"])
    ):
        raise ValueError("E6 frozen artifact hash differs")

    checkpoint_hashes = freeze.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != expected_unique_models():
        raise ValueError("E6 checkpoint set differs")
    for name, digest in checkpoint_hashes.items():
        checkpoint = root / "checkpoints" / f"{name}.pt"
        if not checkpoint.is_file() or sha256(checkpoint) != digest:
            raise ValueError(f"E6 checkpoint receipt differs: {name}")

    histories = read_json(required["histories"])
    if set(histories) != expected_unique_models():
        raise ValueError("E6 training-history set differs")
    for name, rows in histories.items():
        if not isinstance(rows, list) or len(rows) != 16:
            raise ValueError(f"E6 history length differs: {name}")
        if [int(row["epoch"]) for row in rows] != list(range(1, 17)):
            raise ValueError(f"E6 history epochs differ: {name}")
        for row in rows:
            for key in ("image", "instance", "consistency", "total"):
                _finite(row[key], f"{name}/{key}")

    label_metrics = read_json(required["label_metrics"])
    if set(label_metrics) != reported_arms:
        raise ValueError("E6 image-label metric arm set differs")
    for arm, metrics in label_metrics.items():
        if (
            int(metrics.get("n", -1)) != 371
            or int(metrics.get("positive", -1)) != 184
            or int(metrics.get("negative", -1)) != 187
        ):
            raise ValueError(f"E6 image-label population differs: {arm}")
        for key in ("auroc", "average_precision_auprc", "f1", "brier_score"):
            _finite(metrics[key], f"{arm}/{key}")

    choices = read_csv(required["choices"])
    choice_ids, _ = _require_matrix(choices, all_arms, name="choice")
    choices_by_key = {(row["image_id"], row["arm"]): row for row in choices}
    for seed in SEEDS:
        feature = f"E6F__full__seed{seed}"
        loss = f"E6L__full__seed{seed}"
        for image_id in choice_ids:
            left, right = choices_by_key[(image_id, feature)], choices_by_key[(image_id, loss)]
            for key in (
                "candidate_payload_sha256",
                "eligible_candidate_indices",
                "selected_candidate_index",
                "selected_source",
                "selected_prompt_mode",
                "selected_g1_logit",
            ):
                if left[key] != right[key]:
                    raise ValueError(f"E6 full aliases differ: {seed}/{image_id}/{key}")

    manifest = read_json(required["manifest"])
    if (
        manifest.get("study") != freeze["study"]
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("protocol_sha256") != expected_protocol_sha256
        or manifest.get("split_sha256") != expected_split_sha256
        or manifest.get("choice_freeze_sha256") != sha256(required["freeze"])
        or manifest.get("validation_gt_read") is not False
        or manifest.get("training", {}).get("seeds") != list(SEEDS)
        or int(manifest.get("training", {}).get("epochs", -1)) != 16
    ):
        raise ValueError("E6 run manifest differs")
    _require_zero_test(manifest, "run manifest")

    evaluation_audit = read_json(required["evaluation_audit"])
    if (
        evaluation_audit.get("pass") is not True
        or evaluation_audit.get("choices_frozen_before_annotations") is not True
        or evaluation_audit.get("choice_freeze_sha256") != sha256(required["freeze"])
        or evaluation_audit.get("per_image_sha256") != sha256(required["per_image"])
        or evaluation_audit.get("summary_sha256") != sha256(required["summary"])
        or int(evaluation_audit.get("images", -1)) != 371
        or int(evaluation_audit.get("tumor_images", -1)) != 184
        or int(evaluation_audit.get("arms", -1)) != 25
        or int(evaluation_audit.get("per_image_rows", -1)) != 9275
        or int(evaluation_audit.get("validation_annotations_opened", -1)) != 184
    ):
        raise ValueError("E6 evaluator audit differs")
    _require_zero_test(evaluation_audit, "evaluation audit")

    summary = read_json(required["summary"])
    if (
        summary.get("study") != freeze["study"]
        or summary.get("primary_grid") != "native"
        or summary.get("baseline_arm") != BASELINE_ARM
        or summary.get("choice_freeze_sha256") != sha256(required["freeze"])
        or summary.get("split_sha256") != expected_split_sha256
        or int(summary.get("images", -1)) != 371
        or int(summary.get("tumor_images", -1)) != 184
        or int(summary.get("spatial_annotations_opened", -1)) != 184
        or set(summary.get("summaries", {})) != all_arms
    ):
        raise ValueError("E6 evaluation summary differs")
    _require_zero_test(summary, "evaluation summary")

    per_image = read_csv(required["per_image"])
    per_image_ids, _ = _require_matrix(per_image, all_arms, name="evaluation")
    if per_image_ids != choice_ids:
        raise ValueError("E6 choice/evaluation image IDs differ")
    for row in per_image:
        choice = choices_by_key[(row["image_id"], row["arm"])]
        if (
            row["selected_candidate_index"] != choice["selected_candidate_index"]
            or row["selected_source"] != choice["selected_source"]
        ):
            raise ValueError("E6 frozen choice changed during evaluation")

    baseline = summary["summaries"][BASELINE_ARM]
    if (
        abs(float(baseline["selected_dice_common320"]) - EXPECTED_BASELINE_COMMON320) > 1e-12
        or abs(float(baseline["mean_tumor_dice"]) - EXPECTED_BASELINE_NATIVE) > 1e-12
    ):
        raise ValueError("E6 baseline reproduction differs")

    aggregate = _aggregate(summary, label_metrics)
    feature_full = aggregate["feature_full"]
    loss_full = aggregate["loss_full"]
    if feature_full != loss_full:
        raise ValueError("E6 full feature/loss aggregate aliases differ")

    selected_source_counts = Counter(
        row["selected_source"] for row in choices if row["arm"] == BASELINE_ARM
    )
    return {
        "schema_version": 1,
        "study": "independent G4 E6 G1 feature/loss output audit",
        "pass": True,
        "source_commit": expected_source_commit,
        "protocol_sha256": expected_protocol_sha256,
        "split_sha256": expected_split_sha256,
        "candidate_manifest_sha256": expected_candidate_manifest_sha256,
        "choice_freeze_sha256": sha256(required["freeze"]),
        "choices_sha256": sha256(required["choices"]),
        "summary_sha256": sha256(required["summary"]),
        "per_image_sha256": sha256(required["per_image"]),
        "evaluation_audit_sha256": sha256(required["evaluation_audit"]),
        "checkpoints": 21,
        "reported_learned_arms": 24,
        "arms_including_baseline": 25,
        "seeds": list(SEEDS),
        "images": 371,
        "tumor_images": 184,
        "selection_rows": 9275,
        "baseline_native_dice": EXPECTED_BASELINE_NATIVE,
        "baseline_common320_dice": EXPECTED_BASELINE_COMMON320,
        "baseline_selected_source_counts": dict(sorted(selected_source_counts.items())),
        "aggregate_mean_sample_sd": aggregate,
        "validation_annotations_opened_after_freeze": 184,
        "validation_gt_read_during_training": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("E6 audit output already exists")
    result = audit(
        args.root.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_split_sha256=args.expected_split_sha256,
        expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "audit_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
