from __future__ import annotations

"""Run the frozen S6 coarse-versus-hierarchical image-label MIL pair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_label_granularity import (
    LabelGranularityConfig,
    LabelGranularityResidual,
)
from models.mask_bag_label_granularity_training import (
    LabelGranularityTrainingConfig,
    attach_frozen_base_logits,
    attach_tumor_type_labels,
    audit_zero_initialization,
    initial_residual_state,
    score_label_granularity_pair,
    train_label_granularity_arm,
)
from run_mask_bag_critical_relation_arm import (
    _absolute_spearman,
    _validate_descriptor_cache,
)
from run_mask_bag_normal_prototype_arm import (
    _load_baseline_model,
    _load_cache_records,
    _verify_cache_freeze,
    _write_validation_outputs,
)


EXPERIMENT_ID = "EXP-20260802-codex-s6-label-granularity-mil-v1"
RUN_ID = "btxrd_mask_bag_label_granularity_s6_pair_v1"
EXPECTED_SUBTYPE_COUNTS = (598, 211, 164, 74, 35, 41, 92, 237, 36)
ARMS = ("coarse_control", "hierarchical_entropy_routed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bag-temperature", type=float, default=0.20)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--residual-drift-weight", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_recipe(args: argparse.Namespace) -> None:
    actual = (
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.hidden_dim,
        args.bag_temperature,
        args.consistency_weight,
        args.residual_drift_weight,
        args.seed,
    )
    expected = (16, 16, 3.0e-4, 1.0e-4, 128, 0.20, 0.10, 1.0e-3, 42)
    if actual != expected:
        raise ValueError("S6 execution differs from the frozen one-shot recipe")


def _rank_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("binary metrics require aligned vectors")
    if not positive.any() or not negative.any() or not np.isfinite(probabilities).all():
        raise ValueError("binary AUROC requires finite scores and both classes")
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positives = int(positive.sum())
    negatives = int(negative.sum())
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(probabilities) >= 0.5
    positive = np.asarray(labels) == 1
    negative = ~positive
    return {
        "auroc": _rank_auc(labels, probabilities),
        "sensitivity": float(np.logical_and(predictions, positive).sum() / positive.sum()),
        "specificity": float(np.logical_and(~predictions, negative).sum() / negative.sum()),
    }


def _subtype_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if int(row["tumor"]) == 1]
    recalls: dict[str, float] = {}
    correct = 0
    for subtype in range(1, 10):
        subset = [row for row in positive if int(row["tumor_type"]) == subtype]
        if not subset:
            raise ValueError(f"validation split omits tumor subtype {subtype}")
        hits = sum(int(row["predicted_tumor_type"]) == subtype for row in subset)
        recalls[str(subtype)] = hits / len(subset)
        correct += hits
    return {
        "tumor_images": len(positive),
        "accuracy": correct / len(positive),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "per_subtype_recall": recalls,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("cannot write empty S6 evidence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["tumor"]) for row in rows], dtype=np.int64)
    counts = np.asarray([int(row["candidate_count"]) for row in rows], dtype=np.int64)
    control_probability = np.asarray(
        [float(row["control_bag_probability"]) for row in rows], dtype=np.float64
    )
    hierarchy_probability = np.asarray(
        [float(row["hierarchy_bag_probability"]) for row in rows], dtype=np.float64
    )
    route = np.asarray(
        [float(row["entropy_route_strength"]) for row in rows], dtype=np.float64
    )
    changed = sum(
        int(row["control_selected_local_index"])
        != int(row["hierarchy_selected_local_index"])
        for row in rows
    )
    return {
        "records": len(rows),
        "control_binary_metrics": _binary_metrics(labels, control_probability),
        "hierarchy_binary_metrics": _binary_metrics(labels, hierarchy_probability),
        "hierarchy_subtype_metrics": _subtype_metrics(rows),
        "control_absolute_candidate_count_probability_spearman": _absolute_spearman(
            counts, control_probability
        ),
        "hierarchy_absolute_candidate_count_probability_spearman": _absolute_spearman(
            counts, hierarchy_probability
        ),
        "control_original_flip_agreement": float(
            np.mean([int(row["control_original_flip_agreement"]) for row in rows])
        ),
        "hierarchy_original_flip_agreement": float(
            np.mean([int(row["hierarchy_original_flip_agreement"]) for row in rows])
        ),
        "changed_selection_count": changed,
        "changed_selection_fraction": changed / len(rows),
        "entropy_route_strength": {
            "mean": float(route.mean()),
            "median": float(np.median(route)),
            "q10": float(np.quantile(route, 0.10)),
            "q90": float(np.quantile(route, 0.90)),
            "minimum": float(route.min()),
            "maximum": float(route.max()),
        },
        "diagnostics_block_prediction_freeze": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def _score_device_shard(
    records: list[dict[str, Any]],
    control_state: Mapping[str, torch.Tensor],
    hierarchy_state: Mapping[str, torch.Tensor],
    *,
    model_config: LabelGranularityConfig,
    batch_size: int,
    device_index: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    control = LabelGranularityResidual(model_config).to(device)
    hierarchy = LabelGranularityResidual(model_config).to(device)
    control.load_state_dict(control_state, strict=True)
    hierarchy.load_state_dict(hierarchy_state, strict=True)
    # Explicit real CUDA work on each scoring device is retained in runtime evidence.
    smoke = torch.arange(4096, dtype=torch.float32, device=device).reshape(64, 64)
    smoke_checksum = float((smoke @ smoke.T).sum().item())
    arms, diagnostics = score_label_granularity_pair(
        records,
        control,
        hierarchy,
        model_config=model_config,
        batch_size=batch_size,
        device=device,
    )
    runtime = {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "records": len(records),
        "cuda_matmul_checksum": smoke_checksum,
    }
    del control, hierarchy, smoke
    torch.cuda.empty_cache()
    return arms, diagnostics, runtime


def main() -> None:
    args = parse_args()
    _validate_recipe(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S6 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S6 requires Tesla T4 x2, got {device_names}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)
    split_rows = {
        split: load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("S6 frozen train/validation cohort mismatch")
    cache, validated_cache_rows = _load_cache_records(
        args, split_rows, cache_manifest_rows
    )
    train_records = cache["train"]
    val_records = cache["val"]
    descriptor_dim = _validate_descriptor_cache(train_records + val_records)
    train_subtype_counts = tuple(
        attach_tumor_type_labels(train_records, split_rows["train"])
    )
    attach_tumor_type_labels(val_records, split_rows["val"])
    if train_subtype_counts != EXPECTED_SUBTYPE_COUNTS:
        raise RuntimeError(
            f"S6 train subtype counts differ: {train_subtype_counts}"
        )
    label_evidence = {
        "taxonomy": {
            "0": "normal",
            "1": "osteochondroma",
            "2": "multiple osteochondromas",
            "3": "simple bone cyst",
            "4": "giant cell tumor",
            "5": "osteofibroma",
            "6": "synovial osteochondroma",
            "7": "other bt",
            "8": "osteosarcoma",
            "9": "other mt",
        },
        "train_subtype_counts_1_to_9": list(train_subtype_counts),
        "training_label_scope": "image_level_tumor_benign_malignant_tumor_type_only",
        "validation_labels_used_for_training": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    label_path = args.output_dir / "image_label_evidence.json"
    label_path.write_text(
        json.dumps(label_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    training_device = torch.device("cuda:0")
    base, baseline_config = _load_baseline_model(args, device=training_device)
    if not math.isclose(
        baseline_config.bag_temperature,
        args.bag_temperature,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("S6 bag temperature differs from frozen baseline")
    attach_frozen_base_logits(
        train_records,
        base,
        batch_size=args.batch_size,
        device=training_device,
    )
    attach_frozen_base_logits(
        val_records,
        base,
        batch_size=args.batch_size,
        device=training_device,
    )
    del base
    torch.cuda.empty_cache()

    model_config = LabelGranularityConfig(
        descriptor_dim=descriptor_dim,
        hidden_dim=args.hidden_dim,
        bag_temperature=args.bag_temperature,
        consistency_weight=args.consistency_weight,
        residual_drift_weight=args.residual_drift_weight,
    )
    training_config = LabelGranularityTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    initial_state = initial_residual_state(model_config, seed=args.seed)
    pretraining_identity = {
        "train": audit_zero_initialization(
            train_records,
            model_config=model_config,
            batch_size=args.batch_size,
            device=training_device,
            initial_state=initial_state,
        ),
        "validation": audit_zero_initialization(
            val_records,
            model_config=model_config,
            batch_size=args.batch_size,
            device=training_device,
            initial_state=initial_state,
        ),
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    for cohort in ("train", "validation"):
        audit = pretraining_identity[cohort]
        if (
            audit["exact_control_candidate_score_records"] != audit["records"]
            or audit["exact_hierarchy_candidate_score_records"] != audit["records"]
            or audit["maximum_candidate_score_delta"] != 0.0
            or audit["maximum_zero_init_entropy_route_strength"] > 1.0e-6
        ):
            raise RuntimeError(f"S6 zero-initialization identity failed for {cohort}")
    identity_path = args.output_dir / "pretraining_identity_audit.json"
    identity_path.write_text(
        json.dumps(pretraining_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    control, control_history = train_label_granularity_arm(
        train_records,
        model_config=model_config,
        training_config=training_config,
        subtype_counts=train_subtype_counts,
        hierarchical=False,
        device=training_device,
        initial_state=initial_state,
    )
    hierarchy, hierarchy_history = train_label_granularity_arm(
        train_records,
        model_config=model_config,
        training_config=training_config,
        subtype_counts=train_subtype_counts,
        hierarchical=True,
        device=training_device,
        initial_state=initial_state,
    )
    control_state = {
        key: value.detach().cpu().clone() for key, value in control.state_dict().items()
    }
    hierarchy_state = {
        key: value.detach().cpu().clone()
        for key, value in hierarchy.state_dict().items()
    }
    del control, hierarchy
    torch.cuda.empty_cache()

    histories = {
        "coarse_control": control_history,
        "hierarchical_entropy_routed": hierarchy_history,
    }
    history_path = args.output_dir / "training_histories.json"
    history_path.write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_common = {
        "experiment_id": EXPERIMENT_ID,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    checkpoint_paths: dict[str, Path] = {}
    for arm, state in (
        ("coarse_control", control_state),
        ("hierarchical_entropy_routed", hierarchy_state),
    ):
        path = args.output_dir / f"{arm}_residual.pt"
        torch.save(
            {**checkpoint_common, "arm": arm, "model_state_dict": state}, path
        )
        checkpoint_paths[arm] = path

    shards = [val_records[::2], val_records[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _score_device_shard,
                shards[device_index],
                control_state,
                hierarchy_state,
                model_config=model_config,
                batch_size=args.batch_size,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        shard_outputs = [future.result() for future in futures]
    unordered_arms = {
        arm: [
            item
            for shard_arms, _diagnostics, _runtime in shard_outputs
            for item in shard_arms[arm]
        ]
        for arm in ARMS
    }
    unordered_diagnostics = [
        item
        for _arms, diagnostics, _runtime in shard_outputs
        for item in diagnostics
    ]
    diagnostic_by_image = {
        str(item["image_id"]): item for item in unordered_diagnostics
    }
    if len(diagnostic_by_image) != 371:
        raise RuntimeError("S6 T4x2 diagnostics do not cover 371 unique images")
    scored: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        by_image = {str(item["image_id"]): item for item in unordered_arms[arm]}
        if len(by_image) != 371:
            raise RuntimeError(f"S6 {arm} does not cover 371 unique images")
        scored[arm] = [by_image[str(record["image_id"])] for record in val_records]
    diagnostics = [
        diagnostic_by_image[str(record["image_id"])] for record in val_records
    ]
    diagnostic_csv = args.output_dir / "gt_blind_diagnostics.csv"
    diagnostic_csv_sha256 = _write_csv(diagnostic_csv, diagnostics)
    diagnostic_summary = _diagnostic_summary(diagnostics)
    diagnostic_summary_path = args.output_dir / "gt_blind_diagnostic_summary.json"
    diagnostic_summary_path.write_text(
        json.dumps(diagnostic_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arm_freezes: dict[str, str] = {}
    for arm in ARMS:
        arm_root = args.output_dir / arm
        arm_args = argparse.Namespace(output_dir=arm_root)
        prediction_sha, score_sha = _write_validation_outputs(
            arm_args, val_records, scored[arm]
        )
        freeze = {
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
            "selector_cache_manifest_sha256": cache_freeze[
                "selector_cache_manifest_sha256"
            ],
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "checkpoint_sha256": sha256_file(checkpoint_paths[arm]),
            "training_histories_sha256": sha256_file(history_path),
            "image_label_evidence_sha256": sha256_file(label_path),
            "pretraining_identity_audit_sha256": sha256_file(identity_path),
            "gt_blind_diagnostics_sha256": diagnostic_csv_sha256,
            "gt_blind_diagnostic_summary_sha256": sha256_file(
                diagnostic_summary_path
            ),
            "candidate_score_manifest_sha256": score_sha,
            "prediction_manifest_sha256": prediction_sha,
            "validation_predictions": 371,
            "training_labels": "image_level_only",
            "validation_subtype_label_used_for_routing": False,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        freeze_path.write_text(
            json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        arm_freezes[arm] = sha256_file(freeze_path)

    pair = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "arms": arm_freezes,
        "pair_physically_frozen_before_validation_gt": True,
        "diagnostics_block_prediction_freeze": False,
        "validation_subtype_label_used_for_routing": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    pair_path.write_text(
        json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "validated_cache_records": {
            "train": len(validated_cache_rows["train"]),
            "validation": len(validated_cache_rows["val"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "training_device": device_names[0],
            "validation_shards": [len(shards[0]), len(shards[1])],
            "scoring_device_evidence": [
                runtime for _arms, _diagnostics, runtime in shard_outputs
            ],
        },
        "prediction_pair_freeze_sha256": sha256_file(pair_path),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    run_path = args.output_dir / "run_manifest.json"
    run_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
