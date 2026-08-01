from __future__ import annotations

"""Run the gated count-controlled T1 self-paced confirmation selector arm."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_count_controlled_self_paced import (
    CountControlledResidual,
    CountControlledSelfPacedConfig,
    audit_consumer_residual_identity,
    audit_count_controlled_oof_producer,
    build_self_paced_targets,
    default_producer_model_config,
    fit_count_controlled_oof_fold,
    initial_consumer_state,
    initial_producer_state,
    score_self_paced_consumer,
    train_self_paced_consumer,
)
from models.mask_bag_crossfit import (
    assign_group_stratified_folds,
    crossfit_assignment_manifest,
)
from run_mask_bag_normal_prototype_arm import (
    _load_baseline_model,
    _load_cache_records,
    _verify_cache_freeze,
    _write_validation_outputs,
)


FOLD_COUNT = 5
EXPECTED_FOLD_SUMMARY = [
    {"fold": 0, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 1, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 2, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 3, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 4, "images": 597, "groups": 198, "normal_images": 299, "tumor_images": 298},
]


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
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--producer-epochs", type=int, default=16)
    parser.add_argument("--producer-batch-size", type=int, default=16)
    parser.add_argument("--producer-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--producer-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--view-consistency-weight", type=float, default=0.10)
    parser.add_argument("--count-independence-weight", type=float, default=1.0)
    parser.add_argument("--maximum-count-spearman", type=float, default=0.5013777759365411)
    parser.add_argument("--minimum-oof-auroc", type=float, default=0.75)
    parser.add_argument("--minimum-view-agreement", type=float, default=0.60)
    parser.add_argument("--pace-fractions", default="0.20,0.40,0.60")
    parser.add_argument("--consumer-epochs", type=int, default=12)
    parser.add_argument("--consumer-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--supervised-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _frozen_training_config(args: argparse.Namespace) -> CountControlledSelfPacedConfig:
    observed = {
        "fold_count": args.fold_count,
        "producer_epochs": args.producer_epochs,
        "producer_batch_size": args.producer_batch_size,
        "producer_learning_rate": args.producer_learning_rate,
        "producer_weight_decay": args.producer_weight_decay,
        "view_consistency_weight": args.view_consistency_weight,
        "count_independence_weight": args.count_independence_weight,
        "maximum_count_spearman": args.maximum_count_spearman,
        "minimum_oof_auroc": args.minimum_oof_auroc,
        "minimum_view_agreement": args.minimum_view_agreement,
        "pace_fractions": tuple(float(value) for value in args.pace_fractions.split(",")),
        "consumer_epochs": args.consumer_epochs,
        "consumer_learning_rate": args.consumer_learning_rate,
        "supervised_contrastive_weight": args.supervised_contrastive_weight,
        "contrastive_temperature": args.contrastive_temperature,
        "residual_hidden_dim": args.residual_hidden_dim,
        "seed": args.seed,
    }
    expected = asdict(CountControlledSelfPacedConfig())
    expected["pace_fractions"] = tuple(expected["pace_fractions"])
    if observed != expected:
        raise ValueError("T1 execution differs from the frozen finite contract")
    return CountControlledSelfPacedConfig(**observed)


def _validate_cache(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("T1 cache cannot be empty")
    for record in records:
        count = len(record["candidate_indices"])
        families = np.asarray(record["family_ids"])
        if count < 1 or families.shape != (count,):
            raise ValueError("T1 cache candidate/family alignment mismatch")
        if np.asarray(record["descriptors"]).shape != (count, 1156):
            raise ValueError("T1 requires the frozen 1,156-D descriptor")
        if np.asarray(record["flipped_descriptors"]).shape != (count, 1156):
            raise ValueError("T1 flipped descriptor alignment mismatch")


def _fit_oof_jobs(
    records: list[dict[str, Any]],
    fold_ids: np.ndarray,
    folds: Sequence[int],
    training_config: CountControlledSelfPacedConfig,
    initial_states: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    model_config = default_producer_model_config()
    output = [
        fit_count_controlled_oof_fold(
            records,
            fold_ids,
            heldout_fold=int(fold),
            model_config=model_config,
            training_config=training_config,
            device=device,
            initial_state=initial_states[int(fold)],
        )
        for fold in folds
    ]
    torch.cuda.empty_cache()
    return output


def _write_score_payloads(
    root: Path,
    scores: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        image_id = str(score["image_id"])
        record = records_by_id[image_id]
        path = root / f"{index:04d}_{Path(image_id).stem}.npz"
        np.savez_compressed(
            path,
            candidate_indices=np.asarray(record["candidate_indices"], dtype=np.int32),
            original_logits=np.asarray(score["original_logits"], dtype=np.float32),
            flipped_logits=np.asarray(score["flipped_logits"], dtype=np.float32),
        )
        rows.append(
            {
                "image_id": image_id,
                "group_id": score["group_id"],
                "image_label": score["image_label"],
                "heldout_fold": score["heldout_fold"],
                "candidate_count": score["candidate_count"],
                "bag_probability": score["bag_probability"],
                "payload_path": path.name,
                "payload_sha256": sha256_file(path),
            }
        )
    manifest = root.parent / "score_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(manifest)


def _write_oof_artifacts(
    output_dir: Path,
    fold_artifacts: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    training_config: CountControlledSelfPacedConfig,
) -> dict[str, Any]:
    root = output_dir / "oof_producers"
    root.mkdir(parents=True, exist_ok=False)
    records_by_id = {str(record["image_id"]): record for record in records}
    hashes: dict[str, Any] = {}
    for artifact in sorted(fold_artifacts, key=lambda row: int(row["heldout_fold"])):
        fold = int(artifact["heldout_fold"])
        fold_root = root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=False)
        checkpoint = fold_root / "producer.pt"
        torch.save(
            {
                "model_state_dict": artifact["producer_state_dict"],
                "model_config": asdict(default_producer_model_config()),
                "training_config": asdict(training_config),
                "heldout_fold": fold,
                "training_groups": artifact["training_groups"],
                "heldout_groups": artifact["heldout_groups"],
                "source_commit": args.source_commit,
                "protocol_sha256": args.protocol_sha256,
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            checkpoint,
        )
        history = fold_root / "training_history.json"
        history.write_text(
            json.dumps(artifact["training_history"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        score_manifest_sha = _write_score_payloads(
            fold_root / "scores", artifact["heldout_scores"], records_by_id
        )
        audit = fold_root / "fold_audit.json"
        audit.write_text(
            json.dumps(
                {
                    "heldout_fold": fold,
                    "derived_seed": artifact["derived_seed"],
                    "training_groups": artifact["training_groups"],
                    "heldout_groups": artifact["heldout_groups"],
                    "group_overlap": 0,
                    "heldout_records": len(artifact["heldout_scores"]),
                    "producer_checkpoint_sha256": sha256_file(checkpoint),
                    "training_history_sha256": sha256_file(history),
                    "score_manifest_sha256": score_manifest_sha,
                    "validation_segmentation_quality_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        hashes[f"fold_{fold}"] = {
            "producer_checkpoint_sha256": sha256_file(checkpoint),
            "training_history_sha256": sha256_file(history),
            "score_manifest_sha256": score_manifest_sha,
            "fold_audit_sha256": sha256_file(audit),
        }
    return hashes


def _write_target_bundle(
    output_dir: Path,
    target_bundle: Mapping[str, Any],
    producer_gate_path: Path,
) -> dict[str, Any]:
    root = output_dir / "self_paced_targets"
    root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, Any] = {}
    collections = [("negative_targets.csv", target_bundle["negative_targets"])]
    collections.extend(
        (f"positive_targets_stage_{stage['stage']}.csv", stage["positive_targets"])
        for stage in target_bundle["stages"]
    )
    for name, rows in collections:
        path = root / name
        if not rows:
            raise RuntimeError(f"T1 target collection is empty: {name}")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        hashes[name] = sha256_file(path)
    freeze = {
        "producer_gate_sha256": sha256_file(producer_gate_path),
        "eligible_positive_bags": target_bundle["eligible_positive_bags"],
        "pace_fractions": [stage["fraction"] for stage in target_bundle["stages"]],
        "target_hashes": hashes,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = root / "target_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"target_freeze_sha256": sha256_file(freeze_path), **freeze}


def _rankdata(values: np.ndarray) -> np.ndarray:
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


def _absolute_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = _rankdata(np.asarray(first, dtype=np.float64))
    right = _rankdata(np.asarray(second, dtype=np.float64))
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        raise ValueError("T1 Spearman inputs are invalid")
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _write_gt_blind_diagnostics(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    *,
    ceiling: float,
) -> tuple[str, float]:
    evidence_root = output_dir / "validation_residual_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index, (record, prediction) in enumerate(zip(records, scored, strict=True)):
        if str(record["image_id"]) != str(prediction["image_id"]):
            raise RuntimeError("T1 diagnostic order mismatch")
        image_id = str(record["image_id"])
        count = int(prediction["candidate_count"])
        arrays = {
            name: np.asarray(prediction[name], dtype=np.float32)
            for name in (
                "original_base_logits",
                "flipped_base_logits",
                "original_residual_logits",
                "flipped_residual_logits",
                "original_candidate_logits",
                "flipped_candidate_logits",
            )
        }
        averaged = np.asarray(prediction["candidate_logits"], dtype=np.float32)
        if (
            any(array.shape != (count,) for array in arrays.values())
            or averaged.shape != (count,)
            or not all(np.isfinite(array).all() for array in [*arrays.values(), averaged])
            or not np.array_equal(
                arrays["original_candidate_logits"],
                arrays["original_base_logits"] + arrays["original_residual_logits"],
            )
            or not np.array_equal(
                arrays["flipped_candidate_logits"],
                arrays["flipped_base_logits"] + arrays["flipped_residual_logits"],
            )
            or not np.array_equal(
                averaged,
                0.5
                * (
                    arrays["original_candidate_logits"]
                    + arrays["flipped_candidate_logits"]
                ),
            )
        ):
            raise RuntimeError("T1 physical residual evidence failed exact identity")
        path = evidence_root / f"{index:04d}_{Path(image_id).stem}.npz"
        np.savez_compressed(
            path,
            candidate_indices=np.asarray(record["candidate_indices"], dtype=np.int32),
            **arrays,
        )
        rows.append(
            {
                "image_id": image_id,
                "candidate_count": count,
                "bag_probability": prediction["bag_probability"],
                "selected_view_agreement": int(prediction["selected_view_agreement"]),
                "residual_evidence_path": path.name,
                "residual_evidence_sha256": sha256_file(path),
            }
        )
    association = _absolute_spearman(
        [row["candidate_count"] for row in rows],
        [row["bag_probability"] for row in rows],
    )
    if association > ceiling:
        raise RuntimeError("T1 final count/probability GT-blind gate failed")
    path = output_dir / "gt_blind_diagnostics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path), association


def _score_validation_shard(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    residual_state: Mapping[str, torch.Tensor],
    training_config: CountControlledSelfPacedConfig,
    *,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    base, model_config = _load_baseline_model(args, device=device)
    residual = CountControlledResidual(
        model_config.descriptor_dim, training_config.residual_hidden_dim
    ).to(device)
    residual.load_state_dict(residual_state, strict=True)
    output = score_self_paced_consumer(
        records,
        base,
        residual,
        model_config,
        batch_size=training_config.producer_batch_size,
        device=device,
    )
    del base, residual
    torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    training_config = _frozen_training_config(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("T1 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"T1 requires Tesla T4 x2, got {device_names}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)
    split_rows = {
        split: load_split_rows_without_annotations(
            args.split_manifest, expected_sha256=args.expected_split_sha256, split=split
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("T1 frozen cohort mismatch")
    cache, validated_cache_rows = _load_cache_records(args, split_rows, cache_manifest_rows)
    train_records = cache["train"]
    val_records = cache["val"]
    _validate_cache(train_records + val_records)
    train_groups = {str(record["group_id"]) for record in train_records}
    val_groups = {str(record["group_id"]) for record in val_records}
    if train_groups & val_groups:
        raise RuntimeError("T1 train and validation groups overlap")

    labels = np.asarray([record["label"] for record in train_records], dtype=np.int8)
    groups = np.asarray([record["group_id"] for record in train_records], dtype="U128")
    fold_ids = assign_group_stratified_folds(
        labels, groups, fold_count=training_config.fold_count, seed=training_config.seed
    )
    assignment = crossfit_assignment_manifest(
        [record["image_id"] for record in train_records], groups, labels, fold_ids
    )
    if assignment["fold_summary"] != EXPECTED_FOLD_SUMMARY:
        raise RuntimeError("T1 cross-fit assignment differs from frozen cohort")
    assignment_path = args.output_dir / "crossfit_assignment.json"
    assignment_path.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    producer_config = default_producer_model_config()
    initial_states = {
        fold: initial_producer_state(
            producer_config, seed=training_config.seed + 1000 + fold
        )
        for fold in range(FOLD_COUNT)
    }
    folds_by_device = [list(range(FOLD_COUNT))[::2], list(range(FOLD_COUNT))[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _fit_oof_jobs,
                train_records,
                fold_ids,
                folds_by_device[device_index],
                training_config,
                initial_states,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        fold_artifacts = [artifact for future in futures for artifact in future.result()]
    producer_gate = audit_count_controlled_oof_producer(
        train_records, fold_ids, fold_artifacts, training_config
    )
    producer_hashes = _write_oof_artifacts(
        args.output_dir, fold_artifacts, train_records, args, training_config
    )
    producer_gate_path = args.output_dir / "producer_gate_audit.json"
    producer_gate_path.write_text(
        json.dumps(
            {key: value for key, value in producer_gate.items() if key != "ordered_scores"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if producer_gate["producer_gate_pass"] is not True:
        raise RuntimeError("T1 producer-only operational gate failed")

    target_bundle = build_self_paced_targets(train_records, producer_gate, training_config)
    target_freeze = _write_target_bundle(args.output_dir, target_bundle, producer_gate_path)

    training_device = torch.device("cuda:0")
    torch.cuda.set_device(training_device)
    base, baseline_config = _load_baseline_model(args, device=training_device)
    consumer_initial_state = initial_consumer_state(
        baseline_config.descriptor_dim, training_config
    )
    initial_residual = CountControlledResidual(
        baseline_config.descriptor_dim, training_config.residual_hidden_dim
    ).to(training_device)
    initial_residual.load_state_dict(consumer_initial_state, strict=True)
    identity = {
        "train": audit_consumer_residual_identity(
            train_records,
            base,
            initial_residual,
            batch_size=training_config.producer_batch_size,
            device=training_device,
        ),
        "validation": audit_consumer_residual_identity(
            val_records,
            base,
            initial_residual,
            batch_size=training_config.producer_batch_size,
            device=training_device,
        ),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    identity_path = args.output_dir / "pretraining_identity_audit.json"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del initial_residual

    residual, history = train_self_paced_consumer(
        train_records,
        target_bundle,
        base,
        baseline_config,
        training_config,
        device=training_device,
        initial_state=consumer_initial_state,
    )
    residual_state = {
        key: value.detach().cpu().clone() for key, value in residual.state_dict().items()
    }
    residual_checkpoint = args.output_dir / "count_controlled_self_paced_residual.pt"
    torch.save(
        {
            "residual_state_dict": residual_state,
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "model_config": asdict(baseline_config),
            "training_config": asdict(training_config),
            "target_freeze_sha256": target_freeze["target_freeze_sha256"],
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        residual_checkpoint,
    )
    history_path = args.output_dir / "consumer_training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del base, residual
    torch.cuda.empty_cache()

    val_shards = [val_records[:186], val_records[186:]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _score_validation_shard,
                args,
                val_shards[device_index],
                residual_state,
                training_config,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        scored_val = [prediction for future in futures for prediction in future.result()]
    diagnostics_sha, final_count_spearman = _write_gt_blind_diagnostics(
        args.output_dir,
        val_records,
        scored_val,
        ceiling=training_config.maximum_count_spearman,
    )
    prediction_manifest_sha, score_manifest_sha = _write_validation_outputs(
        args, val_records, scored_val
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "selector_cache_manifest_sha256": cache_freeze["selector_cache_manifest_sha256"],
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "crossfit_assignment_sha256": sha256_file(assignment_path),
        "producer_gate_audit_sha256": sha256_file(producer_gate_path),
        "producer_hashes": producer_hashes,
        "target_freeze_sha256": target_freeze["target_freeze_sha256"],
        "pretraining_identity_audit_sha256": sha256_file(identity_path),
        "residual_checkpoint_sha256": sha256_file(residual_checkpoint),
        "consumer_training_history_sha256": sha256_file(history_path),
        "gt_blind_diagnostics_sha256": diagnostics_sha,
        "absolute_candidate_count_probability_spearman": final_count_spearman,
        "candidate_score_manifest_sha256": score_manifest_sha,
        "prediction_manifest_sha256": prediction_manifest_sha,
        "validation_predictions": 371,
        "training_labels": "image_level_only",
        "confirmation_residual_trained_after_producer_gate": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "run_id": "btxrd_mask_bag_count_controlled_self_paced_t1_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "crossfit": assignment,
        "producer_model_config": asdict(producer_config),
        "training_config": asdict(training_config),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "oof_parallel_workers": 2,
            "oof_folds_by_device": folds_by_device,
            "validation_shards": [186, 185],
        },
        "validated_cache_records": {
            "train": len(validated_cache_rows["train"]),
            "validation": len(validated_cache_rows["val"]),
        },
        "output_hashes": freeze,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "T1_PREDICTIONS_FROZEN_GT_BLIND",
                "prediction_freeze_sha256": sha256_file(freeze_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "producer_gate": producer_gate["checks"],
                "final_absolute_count_probability_spearman": final_count_spearman,
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
